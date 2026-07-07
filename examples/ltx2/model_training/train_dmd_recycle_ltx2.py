"""
LTX-2 19B port of the Wan DMD student training with SVI-style error recycle
(examples/wanvideo/model_training/train_dmd_recycle.py) — Stage B of the
cross-architecture generalization experiment. DMD2 + One-Forcing cls_branch
GAN + self-correcting recycle, on the sink+recent chunked long-video setup.

═══════════════════════════════════════════════════════════════════════════
MEMORY DESIGN (the "3×19B 显存账") — SHARED BASE + LoRA HOT-SWAP
═══════════════════════════════════════════════════════════════════════════
The Wan script builds 3 full pipes (teacher / student / critic). At 19B that
is 3 × 38 GB bf16 = 114 GB of weights before activations — does not fit.

Observation: all three models share the SAME base (base + fused Stage-A sink
LoRA); they differ only by their trainable LoRA. So we build ONE dit and
inject TWO peft adapters ("student", "critic"):

    teacher forward  = adapters disabled        (exactly base+sink)
    student forward  = "student" adapter active
    critic  forward  = "critic"  adapter active

This is semantically identical to the Wan 3-pipe setup (student/critic LoRAs
are zero-init-B, so at step 0 all three are the same model) and costs one
19B instead of three. Budget on H200 (140 GB):

    dit bf16                          ~38 GB
    student+critic LoRA + AdamW fp32  ~1 GB
    gemma-3-12b text encoder          ~24 GB   (offloadable: --offload_text_encoder
                                                + per-prompt embed cache; cartoon
                                                set has few unique prompts)
    video VAE encoder                 ~1 GB
    cls_branch (dim 4096)             ~0.5 GB
    activations (49f 480×832, ~2730   ~a few GB with gradient checkpointing
      video tokens, ckpt boundaries)
    ─────────────────────────────────
    total                             ~70-80 GB resident → fits, with headroom.

Adapter-switching correctness notes:
  - peft set_adapter()/enable_adapters() flip requires_grad on the inactive
    adapter's params as a side effect, and PyTorch checks the LEAF'S CURRENT
    requires_grad at BACKWARD time — a graph recorded under "student" would
    silently drop the student grads once we switch to "critic" for the GAN-G
    forward (verified empirically on a toy model: grad=None after switch).
    set_dit_role therefore re-enables requires_grad on BOTH adapters after
    every switch. The inactive adapter is not in the forward path, so this
    creates no graph/memory; critic grads picked up by the generator
    backward are discarded at the next critic zero_grad — same as Wan.
  - Rollout no-grad steps and the grad step run under the same active
    adapter; switching only happens at model-role boundaries.

═══════════════════════════════════════════════════════════════════════════
CONDITIONING MAPPING (same as Stage A, train_chunk_aware_recycle_ltx2.py)
═══════════════════════════════════════════════════════════════════════════
    recent → input_images index 0 (latent frame-0 replacement, t=0 there)
    sink   → input_images index --sink_index (t=0 reference token, default -49)
    pose   → OFF by default (in_context IC-LoRA path unverified for cartoon)

Because the recent is conditioned IN-FRAME (latent frame 0 of the denoised
chunk, unlike Wan where references are separate tokens), every loss on the
denoised chunk is weighted by denoise_mask_video: DMD gradient, critic MSE,
and self-correcting L1 all exclude the replaced frame-0 region.

SVI recycle, ported 1:1 from the Wan student (all the same gates):
  - inject: input_latents_video[:, :, 0:1] += α · buffer.sample()
            (sink ref token stays clean — global anchor must not drift)
  - clean rollout (no grad, shared noise/exit) → SC-loss target + clean
    buffer source; `need_clean_rollout` decoupling keeps the push fallback
    from ever reading a corrupt-ref rollout (the sc_weight=0 bug fixed in
    the Wan version is fixed here from the start)
  - collect: x_pred_clean[:, :, -1] - target_latents[:, :, -1]

CAVEATS (inherited from the Stage A port, still true here):
  - LATENT SPACE MISMATCH, worse than Wan: residual collected from the video
    VAE's last latent frame (temporal compression ×8 — one latent covers 8
    pixel frames), injected into a single-frame image encode (covers 1).
    Treated as a proxy; if err_norm looks wrong or sc_loss collapses, the
    strict decode→re-encode collect is the first thing to try.
  - Negative sink_index RoPE behavior must be validated by the chaining
    sanity script (examples/ltx2/model_inference/LTX-2-Chunk-Chain-Sanity.py)
    BEFORE spending GPU-weeks here.
  - Audio tower: never runs (audio_latents=None → model_fn_ltx2 skips it).
    The LoRA target modules ("to_q,to_k,to_v,to_out.0") also match the audio
    attention layers; those adapter params simply never receive gradients
    and AdamW skips them. Harmless, kept for consistency with Stage A.
  - lora_target_modules mirrors Stage A v2: attention + FFN (net.0.proj,
    net.2). Stage A v1's attention-only LoRA chained fine but failed to
    transfer the cartoon style — domain adaptation lives largely in FFN
    (Wan's Stage A adapted ffn too).
"""
import argparse, os, random, sys, time
import accelerate
import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm

# Make sibling files importable regardless of cwd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from diffsynth.core import load_state_dict
from diffsynth.pipelines.ltx2_audio_video import (
    LTX2AudioVideoPipeline,
    ModelConfig,
    model_fn_ltx2,
    LTX2AudioVideoUnit_PromptEmbedder,
    LTX2AudioVideoUnit_NoiseInitializer,
    LTX2AudioVideoUnit_InputVideoEmbedder,
    LTX2AudioVideoUnit_InputImagesEmbedder,
    LTX2AudioVideoUnit_InContextVideoEmbedder,
)
from peft import LoraConfig, inject_adapter_in_model
from peft.tuners.tuners_utils import BaseTunerLayer
from safetensors.torch import save_file

from dmd_utils_ltx2 import (
    timestep_to_sigma,
    add_noise_flow,
    velocity_to_x0,
    derive_denoising_step_list,
    sample_critic_timestep_ltx2,
    compute_critic_loss,
    compute_dmd_gradient,
    compute_dmd_loss,
    cfg_real_x0,
    masked_l1,
)
from train_chunk_aware_recycle_ltx2 import LTX2ChunkAwareDataset, ErrorBuffer
from cls_branch_ltx2 import ClsBranch, FeatureCapturer, gan_g_loss, gan_d_loss


# ===========================================================================
# Pipeline construction (ONE pipe — see memory design in module docstring)
# ===========================================================================
def parse_model_configs(model_paths, model_id_with_origin_paths):
    """Same formats as the training framework: --model_paths is a JSON list
    of local files (e.g. the official combined ltx-2-19b-dev.safetensors,
    from which DiffSynth auto-loads ALL submodules by hash);
    --model_id_with_origin_paths is 'model_id:pattern,...' (downloads)."""
    import json
    configs = []
    if model_paths:
        for path in json.loads(model_paths):
            configs.append(ModelConfig(path=path))
    if model_id_with_origin_paths:
        for pair in model_id_with_origin_paths.split(","):
            model_id, origin_pattern = pair.split(":", 1)
            configs.append(ModelConfig(model_id=model_id, origin_file_pattern=origin_pattern))
    return configs


def build_pipe(args, device, dtype=torch.bfloat16):
    tokenizer_config = (ModelConfig(args.tokenizer_path) if args.tokenizer_path
                        else ModelConfig(model_id="google/gemma-3-12b-it-qat-q4_0-unquantized"))
    pipe = LTX2AudioVideoPipeline.from_pretrained(
        torch_dtype=dtype,
        device=device,
        model_configs=parse_model_configs(args.model_paths, args.model_id_with_origin_paths),
        tokenizer_config=tokenizer_config,
    )
    # InputVideoEmbedder only emits target `input_latents` in training mode.
    pipe.scheduler.set_timesteps(1000, training=True)
    return pipe


def fuse_sink_lora_into_pipe(pipe, sink_lora_path, alpha=1.0):
    """Merge the Stage-A sink+recycle LoRA into the dit weights. After this
    the dit carries no LoRA modules — it IS the teacher base."""
    pipe.load_lora(pipe.dit, lora_config=sink_lora_path, alpha=alpha)
    return pipe


def add_adapter(dit, adapter_name, target_modules, rank):
    """Inject one named peft adapter (zero-init B → no-op at step 0)."""
    lora_config = LoraConfig(r=rank, lora_alpha=rank, target_modules=target_modules)
    return inject_adapter_in_model(lora_config, dit, adapter_name=adapter_name)


def set_dit_role(dit, role):
    """Switch the shared dit between teacher (adapters off) / student / critic.

    requires_grad is re-enabled on BOTH adapters after every switch: peft's
    set_adapter/enable_adapters disable it on inactive adapters, and autograd
    drops grads for leaves whose requires_grad is False at backward() time —
    which would silently kill the student update when the GAN-G forward
    switches the role to "critic" before gen_loss.backward()."""
    assert role in ("teacher", "student", "critic")
    for m in dit.modules():
        if isinstance(m, BaseTunerLayer):
            if role == "teacher":
                m.enable_adapters(False)
            else:
                m.enable_adapters(True)
                m.set_adapter(role)
    for n, p in dit.named_parameters():
        if ".lora_A." in n or ".lora_B." in n:
            p.requires_grad_(True)


def adapter_named_params(dit, adapter_name):
    return [(n, p) for n, p in dit.named_parameters()
            if f".lora_A.{adapter_name}." in n or f".lora_B.{adapter_name}." in n]


def all_reduce_grads(params, num_processes):
    """Manual DDP grad sync; skips params with no grad (e.g. audio-attention
    adapter params that never run)."""
    if num_processes <= 1:
        return
    for p in params:
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad.div_(num_processes)


class ResumableSampler:
    """Index-level skip on the resume epoch — no video decode for skipped
    samples (copy of the Wan version)."""
    def __init__(self, base_sampler, skip_first=0, skip_epoch=0):
        self.base = base_sampler
        self.skip_first = skip_first
        self.skip_epoch = skip_epoch
        self._epoch = 0

    def set_epoch(self, epoch):
        self._epoch = epoch
        if hasattr(self.base, "set_epoch"):
            self.base.set_epoch(epoch)

    def __iter__(self):
        it = iter(self.base)
        if self._epoch == self.skip_epoch and self.skip_first > 0:
            for _ in range(self.skip_first):
                next(it, None)
        return it

    def __len__(self):
        if self._epoch == self.skip_epoch:
            return max(0, len(self.base) - self.skip_first)
        return len(self.base)


# ===========================================================================
# LoRA checkpoint I/O (adapter-name normalization)
# ===========================================================================
def save_lora_state_dict(dit, out_path, adapter_name):
    """Save one adapter's params, normalized to standard 'lora_A.weight'
    naming (GeneralLoRALoader strips adapter segments anyway; normalized
    files are directly fusable at inference via pipe.load_lora)."""
    state_dict = {}
    for name, param in dit.state_dict().items():
        if f".lora_A.{adapter_name}." in name or f".lora_B.{adapter_name}." in name:
            name = name.replace(f".lora_A.{adapter_name}.", ".lora_A.")
            name = name.replace(f".lora_B.{adapter_name}.", ".lora_B.")
            state_dict[name] = param.detach().cpu().contiguous()
    save_file(state_dict, out_path)


def load_lora_into_adapter(dit, lora_state_dict, adapter_name):
    """Inverse of save_lora_state_dict: re-add the adapter segment."""
    fixed = {}
    for k, v in lora_state_dict.items():
        # tolerate files saved with 'default' adapter naming first
        k = k.replace(".lora_A.default.weight", ".lora_A.weight")
        k = k.replace(".lora_B.default.weight", ".lora_B.weight")
        k = k.replace(".lora_A.weight", f".lora_A.{adapter_name}.weight")
        k = k.replace(".lora_B.weight", f".lora_B.{adapter_name}.weight")
        fixed[k] = v
    return dit.load_state_dict(fixed, strict=False)


# ===========================================================================
# Conditioning encoders (run pipeline units manually, all no-grad)
# ===========================================================================
def get_unit(pipe, unit_cls):
    for unit in pipe.units:
        if isinstance(unit, unit_cls):
            return unit
    raise RuntimeError(f"unit {unit_cls.__name__} not found in pipe")


@torch.no_grad()
def encode_prompt_cached(pipe, prompt_unit, text, cache, offload_text_encoder):
    """video_context for one prompt, with a per-process embed cache. With
    --offload_text_encoder, gemma lives on CPU and is shuttled in only on a
    cache miss (~24 GB transfer ≈ seconds — fine for few unique prompts,
    thrashes if every sample has a fresh prompt)."""
    if text in cache:
        return cache[text]
    if offload_text_encoder:
        pipe.text_encoder.to(pipe.device)
    video_context, _, _ = prompt_unit.encode_prompt(pipe, text)
    if offload_text_encoder:
        pipe.text_encoder.to("cpu")
        torch.cuda.empty_cache()
    cache[text] = video_context
    return video_context


@torch.no_grad()
def encode_batch(pipe, batch, args, prompt_cache):
    """Encode one LTX2ChunkAwareDataset sample into model_fn_ltx2 inputs.

    Returns dict with:
        video_context             gemma embedding [1, L, 3840]
        video_positions           RoPE coords for the denoised chunk
        target_latents            video VAE latents [1, 128, f, h, w]
        ilv_aug / ilv_clean       input_latents_video (recent at frame 0,
                                  zeros elsewhere) for the augmented / clean
                                  recent ref
        denoise_mask              [1, 1, f, h, w]; 1-strength at frame 0
        ref_latents/ref_positions sink reference token (shared, always clean)
    """
    prompt_unit = get_unit(pipe, LTX2AudioVideoUnit_PromptEmbedder)
    noise_unit  = get_unit(pipe, LTX2AudioVideoUnit_NoiseInitializer)
    video_unit  = get_unit(pipe, LTX2AudioVideoUnit_InputVideoEmbedder)
    images_unit = get_unit(pipe, LTX2AudioVideoUnit_InputImagesEmbedder)

    video_context = encode_prompt_cached(
        pipe, prompt_unit, batch["prompt"], prompt_cache, args.offload_text_encoder)

    height, width, num_frames = args.height, args.width, args.num_frames
    noise_out = noise_unit.process(
        pipe, height=height, width=width, num_frames=num_frames,
        seed=None, rand_device=pipe.device, frame_rate=args.frame_rate,
    )
    if args.time_shift_frames > 0:
        # Positive-shift sink mapping (see Stage A install_time_shift_hook_ltx2):
        # chunk coords +N frames; sink sits at small positive index ≈ t=0.
        noise_out["video_positions"] = noise_out["video_positions"].clone()
        noise_out["video_positions"][:, 0, ...] += args.time_shift_frames / args.frame_rate

    video_out = video_unit.process(
        pipe, input_video=batch["video"], video_noise=noise_out["video_noise"],
        tiled=False, tile_size_in_pixels=None, tile_overlap_in_pixels=None,
    )
    target_latents = video_out["input_latents"]   # requires scheduler.training

    def _images_cond(recent_pil):
        if args.no_sink:
            input_images, input_indexes = [recent_pil], [0]
        else:
            input_images = [recent_pil, batch["sink_image"]]
            input_indexes = [0, args.sink_index]
        return images_unit.process(
            pipe, video_latents=target_latents,
            input_images=input_images,
            input_images_indexes=input_indexes,
            input_images_strength=args.input_images_strength,
            height=height, width=width, frame_rate=args.frame_rate,
            tiled=False, tile_size_in_pixels=None, tile_overlap_in_pixels=None,
            input_latents_video=None, denoise_mask_video=None,
        )

    if args.recent_augment_mode == "off":
        cond_clean = _images_cond(batch["recent_image_clean"])
        cond_aug = cond_clean
    elif args.recent_augment_mode == "asymmetric":
        cond_aug   = _images_cond(batch["recent_image"])
        cond_clean = _images_cond(batch["recent_image_clean"])
    else:  # symmetric — single encode, aug everywhere
        cond_aug = _images_cond(batch["recent_image"])
        cond_clean = cond_aug

    ic_latents, ic_positions = None, None
    if args.use_pose_control:
        ic_unit = get_unit(pipe, LTX2AudioVideoUnit_InContextVideoEmbedder)
        ic_out = ic_unit.process(
            pipe, in_context_videos=[batch["control_video"]],
            height=height, width=width, num_frames=num_frames,
            frame_rate=args.frame_rate,
            in_context_downsample_factor=args.pose_downsample_factor,
            tiled=False, tile_size_in_pixels=None, tile_overlap_in_pixels=None,
        )
        ic_latents = ic_out["in_context_video_latents"]
        ic_positions = ic_out["in_context_video_positions"]
        if args.time_shift_frames > 0:
            # pose tokens align with the (shifted) video time axis
            ic_positions = ic_positions.clone()
            ic_positions[:, 0, ...] += args.time_shift_frames / args.frame_rate

    return dict(
        video_context=video_context,
        video_positions=noise_out["video_positions"],
        target_latents=target_latents,
        ilv_aug=cond_aug["input_latents_video"],
        ilv_clean=cond_clean["input_latents_video"],
        denoise_mask=cond_aug["denoise_mask_video"],
        ref_latents=cond_aug["ref_frames_latents"],       # sink token (clean)
        ref_positions=cond_aug["ref_frames_positions"],
        ic_latents=ic_latents,                            # pose control tokens
        ic_positions=ic_positions,
    )


# ===========================================================================
# Single dit forward — wraps model_fn_ltx2, video-only
# ===========================================================================
def dit_forward(pipe, role, x_t, t_int, video_context, ilv, mask, cond,
                use_gradient_checkpointing=False):
    """One forward of the shared dit in the given role. Batch must be 1:
    model_fn_ltx2 builds per-token timesteps with a hardcoded batch of 1
    (fine for the official batch-1 inference, breaks for stacked batches —
    the GAN-D loop therefore runs fake/real as two separate forwards)."""
    set_dit_role(pipe.dit, role)
    device, dtype = pipe.device, pipe.torch_dtype
    timestep = torch.tensor([float(t_int)], dtype=dtype, device=device)

    video_positions = cond["video_positions"]
    ref_latents = cond["ref_latents"]
    ref_positions = cond["ref_positions"]
    ic_latents = cond.get("ic_latents")
    ic_positions = cond.get("ic_positions")

    vx, _ = model_fn_ltx2(
        dit=pipe.dit,
        video_latents=x_t.to(dtype=dtype, device=device),
        video_context=video_context,
        video_positions=video_positions,
        video_patchifier=pipe.video_patchifier,
        audio_latents=None,                # audio tower skipped entirely
        timestep=timestep,
        input_latents_video=ilv,
        denoise_mask_video=mask,
        # model_fn mutates these lists in place (`+=`) when in_context videos
        # are present; the list() copies keep cond's lists pristine.
        ref_frames_latents=list(ref_latents) if ref_latents is not None else None,
        ref_frames_positions=list(ref_positions) if ref_positions is not None else None,
        in_context_video_latents=ic_latents,
        in_context_video_positions=ic_positions,
        use_gradient_checkpointing=use_gradient_checkpointing,
    )
    return vx


# ===========================================================================
# Self-forcing rollout (1:1 port of the Wan rollout_student)
# ===========================================================================
def rollout_student(
    pipe, denoising_step_list, target_shape,
    video_context, ilv, mask, cond,
    use_gradient_checkpointing=False,
    initial_noisy=None, exit_idx=None, renoise_noises=None,
):
    """Roll the student through the few-step schedule from pure noise.
    Random exit step keeps grad; earlier steps run no-grad with linear-interp
    re-noising. Same NOTE as the Wan version: the re-noise branch is dead for
    1-step rollouts (exit_idx is always the only step); for 2/4-step rollouts
    the linear re-noise diverges from the scheduler's Euler step — switch to
    scheduler-matched re-noising before relying on multi-step."""
    device, dtype = pipe.device, pipe.torch_dtype
    N = len(denoising_step_list)
    if exit_idx is None:
        exit_idx = random.randrange(N)
    if initial_noisy is None:
        initial_noisy = torch.randn(target_shape, dtype=dtype, device=device)
    noisy = initial_noisy
    renoise_idx = 0

    for idx, t in enumerate(denoising_step_list):
        sigma_t = timestep_to_sigma(t)
        if idx == exit_idx:
            v = dit_forward(
                pipe, "student", noisy, t, video_context, ilv, mask, cond,
                use_gradient_checkpointing=use_gradient_checkpointing,
            )
            x_pred = velocity_to_x0(v, noisy, sigma_t)
            return x_pred, t
        with torch.no_grad():
            v = dit_forward(pipe, "student", noisy, t, video_context, ilv, mask, cond)
            x0 = velocity_to_x0(v, noisy, sigma_t)
            next_t = denoising_step_list[idx + 1]
            sigma_next = timestep_to_sigma(next_t)
            if renoise_noises is not None and renoise_idx < len(renoise_noises):
                fresh = renoise_noises[renoise_idx]
            else:
                fresh = torch.randn_like(x0)
            renoise_idx += 1
            noisy = sigma_next * fresh + (1 - sigma_next) * x0

    raise RuntimeError("rollout_student: exit step not consumed")


# ===========================================================================
# Argument parser (mirrors the Wan student args; LTX-2 deltas called out)
# ===========================================================================
def get_parser():
    p = argparse.ArgumentParser()
    # data
    p.add_argument("--dataset_metadata_path", required=True)
    p.add_argument("--height", type=int, default=832)
    p.add_argument("--width", type=int, default=480)
    p.add_argument("--num_frames", type=int, default=49)
    p.add_argument("--frame_rate", type=float, default=24)
    p.add_argument("--dataset_repeat", type=int, default=1)
    p.add_argument("--recent_aug_strength", type=float, default=0.5)
    # model
    p.add_argument("--model_paths", default=None,
                   help="JSON list of local weight files. The official combined "
                        "ltx-2-19b-dev.safetensors provides dit + VAEs + "
                        "text_encoder_post_modules in one file.")
    p.add_argument("--model_id_with_origin_paths", default=(
        "google/gemma-3-12b-it-qat-q4_0-unquantized:model-*.safetensors"))
    p.add_argument("--tokenizer_path", default=None)
    p.add_argument("--teacher_lora_path", required=True,
                   help="Stage-A sink(+recycle) LoRA, fused into the shared base.")
    p.add_argument("--lora_rank", type=int, default=32)
    p.add_argument("--lora_target_modules", default="to_k,to_q,to_v,to_out.0,net.0.proj,net.2",
                   help="Mirrors Stage A v2 (attention + FFN; v1's attention-"
                        "only failed to transfer cartoon style). Suffixes also "
                        "match the audio tower's modules — those adapters "
                        "never run and never get grads (harmless).")
    p.add_argument("--offload_text_encoder", action="store_true", default=False,
                   help="Keep gemma on CPU; shuttle in on prompt-cache miss. "
                        "Recommended when unique prompt count is small.")
    # sink+recent conditioning (mirrors Stage A — MUST match the teacher's)
    p.add_argument("--sink_index", type=int, default=-49)
    p.add_argument("--no_sink", default=False, action="store_true",
                   help="Recent-only conditioning. MUST match how the Stage-A "
                        "teacher was trained.")
    p.add_argument("--time_shift_frames", type=int, default=0,
                   help="Positive-shift sink mapping — MUST match the teacher "
                        "(48 with --sink_index 1).")
    p.add_argument("--input_images_strength", type=float, default=1.0)
    # pose control (in_context IC-LoRA path) — MUST match the teacher's config
    p.add_argument("--use_pose_control", default=False, action="store_true",
                   help="Drive motion with the dataset's skeleton videos via "
                        "in_context tokens. Requires --ic_lora_path; the "
                        "IC-LoRA is fused into the shared base BEFORE the "
                        "Stage-A teacher LoRA (same stacking as training).")
    p.add_argument("--ic_lora_path", default=(
        "models/Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control/"
        "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"))
    p.add_argument("--pose_downsample_factor", type=int, default=2)
    # recent-frame augmentation routing
    p.add_argument("--recent_augment_mode", choices=["off", "symmetric", "asymmetric"],
                   default="symmetric",
                   help="symmetric: augmented recent for all (the Stage-A "
                        "teacher trained with aug → robust). asymmetric: clean "
                        "→ teacher, augmented → student/critic. off: clean "
                        "everywhere.")
    # DMD specifics
    p.add_argument("--num_inference_steps", type=int, default=1,
                   help="Student's few-step count; 1 → denoise list [1000].")
    p.add_argument("--denoising_step_list", default="",
                   help="Optional manual override (comma-sep timesteps).")
    p.add_argument("--dfake_gen_update_ratio", type=int, default=5)
    p.add_argument("--critic_shift_len", type=int, default=4096,
                   help="dynamic_shift_len for the critic-timestep density "
                        "warp (4096 = pipeline inference default).")
    p.add_argument("--dmd_normalize_grad", action="store_true", default=True)
    p.add_argument("--real_guidance_scale", type=float, default=3.0)
    p.add_argument("--negative_prompt", default=None,
                   help="Teacher CFG uncond text. None → pipeline default "
                        "negative prompt for LTX-2.")
    # ─── GAN (One-Forcing) ───
    p.add_argument("--gan_g_weight", type=float, default=0.03)
    p.add_argument("--gan_d_weight", type=float, default=0.03)
    p.add_argument("--gan_feature_layers", default="21,34,47",
                   help="transformer_blocks indices feeding cls_branch. "
                        "Wan-1.3B used 13,21,29 of 30 → scaled to 48 layers.")
    p.add_argument("--gan_ffn_dim", type=int, default=4096)
    p.add_argument("--gan_num_heads", type=int, default=32,
                   help="4096 dim / 32 heads = 128 per head (Wan default 12 "
                        "does not divide 4096).")
    # ─── SVI-style error recycle + self-correcting loss ───
    p.add_argument("--error_alpha", type=float, default=0.25)
    p.add_argument("--error_buffer_size", type=int, default=500)
    p.add_argument("--error_warmup_count", type=int, default=50)
    p.add_argument("--error_alpha_ramp_steps", type=int, default=200)
    p.add_argument("--sc_weight", type=float, default=0.5)
    p.add_argument("--error_inject_prob", type=float, default=0.8)
    p.add_argument("--error_collect_start_step", type=int, default=100)
    # ─── EMA ───
    p.add_argument("--ema_decay", type=float, default=0.99)
    p.add_argument("--ema_start_step", type=int, default=200)
    # training
    p.add_argument("--learning_rate_student", type=float, default=5e-6)
    p.add_argument("--learning_rate_critic",  type=float, default=5e-6)
    p.add_argument("--grad_clip", type=float, default=0.0,
                   help="Max grad norm (clip_grad_norm_) before each optimizer "
                        "step. 0 disables. ~1.0 caps the explosive updates that "
                        "diverged v6/v7 (dmd_g ran to +11).")
    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--global_step_offset", type=int, default=0)
    # checkpointing
    p.add_argument("--output_path", required=True)
    p.add_argument("--resume_student_from", default=None)
    p.add_argument("--resume_critic_from", default=None)
    p.add_argument("--use_gradient_checkpointing", action="store_true", default=True)
    return p


# ===========================================================================
# Main
# ===========================================================================
def main():
    args = get_parser().parse_args()
    accelerator = accelerate.Accelerator()
    device = accelerator.device
    dtype = torch.bfloat16
    is_main = accelerator.is_main_process
    num_proc = accelerator.num_processes
    target_modules = args.lora_target_modules.split(",")
    if is_main:
        os.makedirs(args.output_path, exist_ok=True)
    accelerator.wait_for_everyone()

    rank = accelerator.process_index
    if is_main:
        print(f"[accel] num_processes={num_proc} (rank {rank}, device {device})")

    # Shared seed BEFORE adapter injection so peft's LoRA init is bit-identical
    # across ranks; per-rank seed set after.
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # ---------------- ONE pipe, shared base, two adapters ----------------
    if is_main: print("[setup] building shared 19B base (teacher = base + fused sink LoRA) ...")
    pipe = build_pipe(args, device, dtype)
    if args.use_pose_control:
        # IC-LoRA first, Stage-A LoRA second — same stacking order as Stage-A
        # training (preset LoRA fused before the trainable LoRA was added).
        pipe.load_lora(pipe.dit, lora_config=args.ic_lora_path, alpha=1.0)
        if is_main: print(f"[setup] fused IC-LoRA {args.ic_lora_path}")
    fuse_sink_lora_into_pipe(pipe, args.teacher_lora_path)
    pipe.dit.requires_grad_(False)

    if is_main: print("[setup] injecting student + critic adapters ...")
    pipe.dit = add_adapter(pipe.dit, "student", target_modules, args.lora_rank)
    pipe.dit = add_adapter(pipe.dit, "critic",  target_modules, args.lora_rank)
    pipe.dit.train()   # gradient_checkpoint_forward path; no dropout in LTXModel

    if args.resume_student_from:
        load_lora_into_adapter(pipe.dit, load_state_dict(args.resume_student_from), "student")
        if is_main: print(f"[resume] student LoRA from {args.resume_student_from}")
    if args.resume_critic_from:
        load_lora_into_adapter(pipe.dit, load_state_dict(args.resume_critic_from), "critic")
        if is_main: print(f"[resume] critic LoRA from {args.resume_critic_from}")

    # ---------------- ClsBranch discriminator ----------------
    gan_layers = [int(x) for x in args.gan_feature_layers.split(",")]
    dit_dim = pipe.dit.inner_dim
    cls_branch = ClsBranch(
        num_layers=len(gan_layers),
        dim=dit_dim,
        ffn_dim=args.gan_ffn_dim,
        num_heads=args.gan_num_heads,
        num_class=1,
    ).to(device=device, dtype=dtype)
    cls_branch.train()
    if is_main:
        n_cls = sum(p.numel() for p in cls_branch.parameters())
        print(f"[setup] cls_branch: {n_cls/1e6:.1f}M params, dim={dit_dim}, layers={gan_layers}")

    # ---------------- Optimizers ----------------
    student_named = adapter_named_params(pipe.dit, "student")
    critic_named  = adapter_named_params(pipe.dit, "critic")
    student_params = [p for _, p in student_named]
    critic_params  = [p for _, p in critic_named]
    cls_params     = list(cls_branch.parameters())
    if is_main:
        print(f"[setup] student adapter params: {sum(p.numel() for p in student_params)/1e6:.1f}M")
        print(f"[setup] critic  adapter params: {sum(p.numel() for p in critic_params )/1e6:.1f}M")

    student_optimizer = torch.optim.AdamW(student_params, lr=args.learning_rate_student, weight_decay=1e-2)
    # critic_optimizer drives BOTH the denoising critic AND the cls_branch.
    critic_optimizer  = torch.optim.AdamW(critic_params + cls_params,
                                          lr=args.learning_rate_critic, weight_decay=1e-2)

    # ---------------- Resume sidecars (optimizer / EMA / cls / buffer) ----------------
    resume_buffer = None
    resume_warmup_done_step = None
    if args.resume_student_from:
        cls_path = args.resume_student_from.replace(".safetensors", "_cls.pt")
        if os.path.isfile(cls_path):
            cls_branch.load_state_dict(torch.load(cls_path, map_location=device))
            if is_main: print(f"[resume] cls_branch state from {cls_path}")
        elif is_main:
            print(f"[resume] cls_branch NOT loaded (no {cls_path}) — fresh init.")

        ema_resume_path = args.resume_student_from.replace(".safetensors", "_ema.safetensors")
        if os.path.isfile(ema_resume_path):
            ema_sd = load_state_dict(ema_resume_path)
            main._ema_state = {}
            for n, p in student_named:
                key = n.replace(".lora_A.student.", ".lora_A.").replace(".lora_B.student.", ".lora_B.")
                if key in ema_sd:
                    main._ema_state[n] = ema_sd[key].to(device=p.device, dtype=p.dtype)
            if is_main:
                print(f"[resume] EMA state from {ema_resume_path} ({len(main._ema_state)} tensors)")
        elif is_main:
            print(f"[resume] EMA NOT loaded (no {ema_resume_path}) — will init when started.")

        state_path = args.resume_student_from.replace(".safetensors", "_state.pt")
        if os.path.isfile(state_path):
            ckpt = torch.load(state_path, map_location=device)
            student_optimizer.load_state_dict(ckpt["student_optimizer"])
            try:
                critic_optimizer.load_state_dict(ckpt["critic_optimizer"])
            except (ValueError, KeyError) as e:
                if is_main:
                    print(f"[resume] critic_optimizer state mismatch ({e}) — cold start critic moments.")
            resume_buffer = ckpt.get("error_buffer", None)
            resume_warmup_done_step = ckpt.get("error_warmup_done_step", None)
            if is_main:
                print(f"[resume] optimizer state from {state_path} "
                      f"(saved at step {ckpt.get('global_step', '?')}, "
                      f"buf={len(resume_buffer) if resume_buffer is not None else 0})")
        elif is_main:
            print(f"[resume] WARNING: no optimizer state at {state_path} — cold start.")

    # ---------------- Denoising schedule ----------------
    if args.denoising_step_list.strip():
        denoising_step_list = [float(x) for x in args.denoising_step_list.split(",")]
    else:
        denoising_step_list = derive_denoising_step_list(args.num_inference_steps)
    if is_main:
        print(f"[setup] denoising_step_list: {[round(t, 1) for t in denoising_step_list]}")

    # ---------------- Negative prompt embedding (teacher CFG uncond) ----------------
    prompt_cache = {}
    prompt_unit = get_unit(pipe, LTX2AudioVideoUnit_PromptEmbedder)
    neg_video_context = None
    if args.real_guidance_scale != 0:
        neg_text = (args.negative_prompt if args.negative_prompt is not None
                    else pipe.default_negative_prompt.get("LTX-2.3", pipe.default_negative_prompt["LTX-2"]))
        neg_video_context = encode_prompt_cached(
            pipe, prompt_unit, neg_text, prompt_cache, args.offload_text_encoder)
        if is_main:
            print(f"[setup] teacher CFG enabled (real_guidance_scale={args.real_guidance_scale})")

    # Per-rank seed (after identical model build), offset-advanced on resume.
    seed = 42 + rank * 10000 + args.global_step_offset
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ---------------- Dataset ----------------
    dataset = LTX2ChunkAwareDataset(
        csv_path=args.dataset_metadata_path,
        height=args.height,
        width=args.width,
        chunk_frames=args.num_frames,
        recent_aug_strength=args.recent_aug_strength,
        dataset_repeat=args.dataset_repeat,
    )
    steps_per_epoch = (len(dataset) // num_proc) if num_proc > 1 else len(dataset)
    steps_per_epoch = max(steps_per_epoch, 1)
    start_epoch  = args.global_step_offset // steps_per_epoch
    skip_batches = args.global_step_offset %  steps_per_epoch

    if num_proc > 1:
        base_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=num_proc, rank=rank, shuffle=True, drop_last=True,
        )
    else:
        base_sampler = torch.utils.data.RandomSampler(dataset)
    sampler = ResumableSampler(base_sampler, skip_first=skip_batches, skip_epoch=start_epoch)
    dataloader = torch.utils.data.DataLoader(
        dataset, sampler=sampler, shuffle=False, batch_size=None, num_workers=2,
    )

    # ---------------- Training log ----------------
    log_file = None
    if is_main:
        log_file = open(os.path.join(args.output_path, "dmd_train.log"), "a", buffering=1)
        log_file.write(f"=== run start {time.strftime('%Y-%m-%d %H:%M:%S')} | "
                       f"world_size={num_proc} "
                       f"lr_s={args.learning_rate_student} lr_c={args.learning_rate_critic} "
                       f"ratio={args.dfake_gen_update_ratio} steps={denoising_step_list} "
                       f"offset={args.global_step_offset} ===\n")

    if is_main and args.global_step_offset:
        print(f"[resume] offset={args.global_step_offset} → start at epoch "
              f"{start_epoch}, skip {skip_batches}/{steps_per_epoch} batches")

    # ---------------- Error buffer ----------------
    error_buffer = ErrorBuffer(max_size=args.error_buffer_size)
    error_warmup_done_step = None
    if resume_buffer is not None:
        error_buffer.buf = [
            t.detach().to(torch.bfloat16).cpu().contiguous()
            for t in list(resume_buffer)[-args.error_buffer_size:]
        ]
    if resume_warmup_done_step is not None:
        error_warmup_done_step = resume_warmup_done_step
    if is_main:
        print(f"[setup] error recycle: alpha={args.error_alpha}, buffer={args.error_buffer_size}, "
              f"warmup_count={args.error_warmup_count}, ramp_steps={args.error_alpha_ramp_steps}, "
              f"sc_weight={args.sc_weight}, initial_buf={len(error_buffer)}")

    # ---------------- Checkpoint saver (also used for the final save) ----------------
    def save_ckpt(global_step):
        accelerator.wait_for_everyone()
        if not is_main:
            return
        student_path = os.path.join(args.output_path, f"step-{global_step}.safetensors")
        critic_path  = os.path.join(args.output_path, f"step-{global_step}_critic.safetensors")
        state_path   = os.path.join(args.output_path, f"step-{global_step}_state.pt")
        save_lora_state_dict(pipe.dit, student_path, "student")
        save_lora_state_dict(pipe.dit, critic_path,  "critic")
        cls_path = os.path.join(args.output_path, f"step-{global_step}_cls.pt")
        torch.save(cls_branch.state_dict(), cls_path)
        if hasattr(main, "_ema_state"):
            ema_path = os.path.join(args.output_path, f"step-{global_step}_ema.safetensors")
            ema_sd = {}
            for n, v in main._ema_state.items():
                key = n.replace(".lora_A.student.", ".lora_A.").replace(".lora_B.student.", ".lora_B.")
                ema_sd[key] = v.detach().cpu().contiguous()
            save_file(ema_sd, ema_path)
        else:
            ema_path = "(not yet started)"
        torch.save({
            "student_optimizer": student_optimizer.state_dict(),
            "critic_optimizer":  critic_optimizer.state_dict(),
            "global_step": global_step,
            "error_buffer": list(error_buffer.buf),
            "error_warmup_done_step": error_warmup_done_step,
        }, state_path)
        print(f"[save] {student_path} + ema:{ema_path} + {cls_path} + {state_path} "
              f"(buf={len(error_buffer)})")

    # ---------------- Training loop ----------------
    local_step = 0
    for epoch in range(start_epoch, start_epoch + args.num_epochs):
        sampler.set_epoch(epoch)
        pbar = tqdm(dataloader, desc=f"epoch {epoch}", disable=not is_main)
        for batch in pbar:
            # ─── encode all conditioning once (no grad) ───
            cond = encode_batch(pipe, batch, args, prompt_cache)
            target_latents = cond["target_latents"]
            video_context  = cond["video_context"]
            mask           = cond["denoise_mask"]

            # Routing per augment mode (sink token is always clean & shared):
            #   off        → clean recent for everyone
            #   symmetric  → augmented recent for everyone
            #   asymmetric → clean → teacher, augmented → student/critic
            if args.recent_augment_mode == "asymmetric":
                ilv_student = cond["ilv_aug"]
                ilv_teacher = cond["ilv_clean"]
            elif args.recent_augment_mode == "off":
                ilv_student = ilv_teacher = cond["ilv_clean"]
            else:  # symmetric
                ilv_student = ilv_teacher = cond["ilv_aug"]

            # ─── SVI: clean vs corrupted recent latent ───
            ilv_student_clean = ilv_student
            alpha_eff = 0.0
            cur_global_step = args.global_step_offset + local_step
            this_step_inject = (random.random() < args.error_inject_prob)
            if (args.error_alpha > 0.0
                    and this_step_inject
                    and len(error_buffer) >= args.error_warmup_count):
                if error_warmup_done_step is None:
                    error_warmup_done_step = cur_global_step
                    if is_main:
                        print(f"[error-recycle] buffer warmup done at global_step={cur_global_step}, "
                              f"begin α ramp")
                if args.error_alpha_ramp_steps > 0:
                    ramp = min(1.0, (cur_global_step - error_warmup_done_step) / args.error_alpha_ramp_steps)
                else:
                    ramp = 1.0
                alpha_eff = args.error_alpha * max(0.0, ramp)

            if alpha_eff > 0.0:
                err = error_buffer.sample(device=ilv_student.device, dtype=ilv_student.dtype)
                ilv_student_corrupt = ilv_student.clone()
                ilv_student_corrupt[:, :, 0:1] = (
                    ilv_student_corrupt[:, :, 0:1]
                    + alpha_eff * err.view(1, err.shape[0], 1, *err.shape[1:])
                )
            else:
                ilv_student_corrupt = ilv_student

            # ─── Shared randomness for clean/corrupt rollouts ───
            sc_loss_active = (alpha_eff > 0.0 and args.sc_weight > 0.0)
            will_push = (cur_global_step >= args.error_collect_start_step)
            # Same invariant as the (fixed) Wan version: the buffer-push
            # fallback below only ever reads a clean-ref rollout.
            need_clean_rollout = sc_loss_active or (alpha_eff > 0.0 and will_push)
            N_steps = len(denoising_step_list)
            shared_exit_idx      = random.randrange(N_steps)
            shared_initial_noisy = torch.randn(target_latents.shape, dtype=dtype, device=device)
            shared_renoise_noises = [
                torch.randn(target_latents.shape, dtype=dtype, device=device)
                for _ in range(shared_exit_idx)
            ]

            # ─── CLEAN ROLLOUT (no grad) ───
            if need_clean_rollout:
                with torch.no_grad():
                    x_pred_clean_target, _ = rollout_student(
                        pipe, denoising_step_list, tuple(target_latents.shape),
                        video_context, ilv_student_clean, mask, cond,
                        use_gradient_checkpointing=args.use_gradient_checkpointing,
                        initial_noisy=shared_initial_noisy,
                        exit_idx=shared_exit_idx,
                        renoise_noises=shared_renoise_noises,
                    )
                    x_pred_clean_target = x_pred_clean_target.detach()
            else:
                x_pred_clean_target = None

            # ─── CORRUPT ROLLOUT (with grad) ───
            x_pred, t_gen = rollout_student(
                pipe, denoising_step_list, tuple(target_latents.shape),
                video_context, ilv_student_corrupt, mask, cond,
                use_gradient_checkpointing=args.use_gradient_checkpointing,
                initial_noisy=shared_initial_noisy,
                exit_idx=shared_exit_idx,
                renoise_noises=shared_renoise_noises,
            )

            # ─── critic update × N (x_pred.detach(), mask-weighted MSE) ───
            for _ in range(args.dfake_gen_update_ratio):
                t_c = sample_critic_timestep_ltx2(dynamic_shift_len=args.critic_shift_len)
                sigma_c = timestep_to_sigma(t_c)
                noise_c = torch.randn_like(x_pred)
                x_pred_noisy_c, _ = add_noise_flow(x_pred.detach(), sigma_c, noise_c)

                v_critic_pred = dit_forward(
                    pipe, "critic", x_pred_noisy_c, t_c,
                    video_context, ilv_student, mask, cond,
                    use_gradient_checkpointing=args.use_gradient_checkpointing,
                )
                critic_loss = compute_critic_loss(v_critic_pred, x_pred, noise_c, mask=mask)
                critic_optimizer.zero_grad()
                critic_loss.backward()
                all_reduce_grads(critic_params + cls_params, num_proc)
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(critic_params + cls_params, args.grad_clip)
                critic_optimizer.step()

            # ─── GAN-D update × N ───
            # Fake and real run as TWO batch-1 forwards with separate backwards
            # (the softplus D-loss is separable). The Wan-style 2×batch stacked
            # forward breaks model_fn_ltx2: it builds per-token timesteps with
            # a hardcoded batch of 1, which the batch-2 denoise mask broadcasts
            # up, and the ref-token timestep cat then mismatches. Separate
            # backwards also avoid holding both graphs alive at once.
            for _ in range(args.dfake_gen_update_ratio):
                t_d = sample_critic_timestep_ltx2(dynamic_shift_len=args.critic_shift_len)
                sigma_d = timestep_to_sigma(t_d)
                noise_d = torch.randn_like(x_pred)
                fake_noisy_d, _ = add_noise_flow(x_pred.detach(), sigma_d, noise_d)
                real_noisy_d, _ = add_noise_flow(target_latents,   sigma_d, noise_d)

                critic_optimizer.zero_grad()
                d_loss_total = 0.0
                for x_d, is_real in ((fake_noisy_d, False), (real_noisy_d, True)):
                    with torch.autograd.graph.save_on_cpu(pin_memory=False):
                        with FeatureCapturer(pipe.dit, gan_layers) as cap:
                            _ = dit_forward(
                                pipe, "critic", x_d, t_d,
                                video_context, ilv_student, mask, cond,
                                use_gradient_checkpointing=args.use_gradient_checkpointing,
                            )
                            feats = cap.features()
                        logit = cls_branch(feats)
                    part = torch.nn.functional.softplus(-logit.float() if is_real
                                                        else logit.float()).mean()
                    (part * args.gan_d_weight).backward()
                    d_loss_total += float(part) * args.gan_d_weight
                d_loss = torch.tensor(d_loss_total)   # logging only
                all_reduce_grads(critic_params + cls_params, num_proc)
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(critic_params + cls_params, args.grad_clip)
                critic_optimizer.step()

            # ─── generator update × 1 (DMD + GAN-G + SC) ───
            t_dmd = sample_critic_timestep_ltx2(dynamic_shift_len=args.critic_shift_len)
            sigma_dmd = timestep_to_sigma(t_dmd)
            noise_dmd = torch.randn_like(x_pred)
            x_pred_noisy_dmd, _ = add_noise_flow(x_pred, sigma_dmd, noise_dmd)

            with torch.no_grad():
                # fake score (critic adapter), student-side recent
                v_fake = dit_forward(
                    pipe, "critic", x_pred_noisy_dmd, t_dmd,
                    video_context, ilv_student, mask, cond,
                )
                pred_fake = velocity_to_x0(v_fake, x_pred_noisy_dmd, sigma_dmd)

                # real score (adapters OFF = teacher), CFG-enhanced, teacher-side recent
                v_real_cond = dit_forward(
                    pipe, "teacher", x_pred_noisy_dmd, t_dmd,
                    video_context, ilv_teacher, mask, cond,
                )
                pred_real_cond = velocity_to_x0(v_real_cond, x_pred_noisy_dmd, sigma_dmd)
                if neg_video_context is not None:
                    v_real_uncond = dit_forward(
                        pipe, "teacher", x_pred_noisy_dmd, t_dmd,
                        neg_video_context, ilv_teacher, mask, cond,
                    )
                    pred_real_uncond = velocity_to_x0(v_real_uncond, x_pred_noisy_dmd, sigma_dmd)
                else:
                    pred_real_uncond = None
                pred_real = cfg_real_x0(pred_real_cond, pred_real_uncond, args.real_guidance_scale)

            grad = compute_dmd_gradient(
                pred_fake, pred_real, x_pred, normalize=args.dmd_normalize_grad, mask=mask,
            )
            dmd_g_loss = compute_dmd_loss(x_pred, grad)

            # GAN-G: critic adapter active; its grads from this loss are
            # discarded at the next critic zero_grad (same as Wan).
            t_g = sample_critic_timestep_ltx2(dynamic_shift_len=args.critic_shift_len)
            sigma_g = timestep_to_sigma(t_g)
            noise_g = torch.randn_like(x_pred)
            fake_noisy_g, _ = add_noise_flow(x_pred, sigma_g, noise_g)
            with torch.autograd.graph.save_on_cpu(pin_memory=False):
                with FeatureCapturer(pipe.dit, gan_layers) as cap:
                    _ = dit_forward(
                        pipe, "critic", fake_noisy_g, t_g,
                        video_context, ilv_student, mask, cond,
                        use_gradient_checkpointing=args.use_gradient_checkpointing,
                    )
                    fake_feats_g = cap.features()
                fake_logit_g = cls_branch(fake_feats_g)
            g_loss = gan_g_loss(fake_logit_g) * args.gan_g_weight

            # SVI self-correcting loss (mask-weighted L1)
            if x_pred_clean_target is not None:
                sc_loss = masked_l1(x_pred, x_pred_clean_target, mask=mask)
            else:
                sc_loss = x_pred.new_zeros(())

            gen_loss = dmd_g_loss + g_loss + args.sc_weight * sc_loss
            student_optimizer.zero_grad()
            gen_loss.backward()
            all_reduce_grads(student_params, num_proc)
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(student_params, args.grad_clip)
            student_optimizer.step()

            # ─── SVI COLLECT (clean source only; see Wan-version invariant) ───
            err_norm = 0.0
            if cur_global_step >= args.error_collect_start_step:
                with torch.no_grad():
                    ref_x_pred = x_pred_clean_target if x_pred_clean_target is not None else x_pred
                    real_last    = target_latents[:, :, -1, :, :]
                    student_last = ref_x_pred[:,    :, -1, :, :].detach()
                    err = (student_last - real_last)
                    err_norm = float(err.abs().mean())
                    for b in range(err.shape[0]):
                        error_buffer.push(err[b])      # [C, h, w]

            # ─── EMA ───
            local_step += 1
            global_step = args.global_step_offset + local_step
            if global_step >= args.ema_start_step:
                with torch.no_grad():
                    if not hasattr(main, "_ema_state"):
                        main._ema_state = {n: p.data.clone() for n, p in student_named}
                        if is_main:
                            print(f"[ema] initialized at step {global_step}")
                    else:
                        d = args.ema_decay
                        for n, p in student_named:
                            if n in main._ema_state:
                                main._ema_state[n].mul_(d).add_(p.data, alpha=1 - d)

            if is_main:
                pbar.set_postfix(
                    g=f"{gen_loss.item():.4f}", c=f"{critic_loss.item():.4f}",
                    gg=f"{g_loss.item():.4f}", gd=f"{d_loss.item():.4f}",
                    sc=f"{float(sc_loss):.4f}", a=f"{alpha_eff:.2f}",
                    en=f"{err_norm:.4f}", buf=len(error_buffer),
                    t=t_gen,
                )
                log_file.write(
                    f"{time.strftime('%H:%M:%S')} epoch={epoch} step={global_step} "
                    f"t_gen={t_gen} gen_loss={gen_loss.item():.6f} "
                    f"dmd_g={dmd_g_loss.item():.6f} gan_g={g_loss.item():.6f} "
                    f"sc_loss={float(sc_loss):.6f} alpha_eff={alpha_eff:.4f} "
                    f"err_norm={err_norm:.6f} buf={len(error_buffer)} "
                    f"critic_loss={critic_loss.item():.6f} gan_d={d_loss.item():.6f}\n"
                )

            # ─── save ───
            if global_step % args.save_steps == 0:
                save_ckpt(global_step)

    # Final save: total steps (e.g. 870 = 174/epoch × 5) need not be a
    # multiple of save_steps — always keep the end-of-training state.
    final_step = args.global_step_offset + local_step
    if final_step % args.save_steps != 0:
        save_ckpt(final_step)

    if is_main:
        log_file.write(f"=== run end {time.strftime('%Y-%m-%d %H:%M:%S')} | "
                       f"total_steps={args.global_step_offset + local_step} ===\n")
        log_file.close()


if __name__ == "__main__":
    main()
