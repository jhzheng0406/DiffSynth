"""
DMD2 + SELF-FORCING ROLLOUT + FD-LOSS  v4  (distribution-alignment fixes vs v3):

  FD's role here = GAN substitute (One-Forcing rationale): it injects the
  REAL-DATA distribution signal beyond the teacher ceiling, while DMD handles
  mode-seeking toward the teacher.  Reference stats therefore STAY real data
  (VAE-roundtripped dataset frames).  v1-v3 had the right FD math but fed it
  features that made it align the wrong thing; v4 fixes the feeding protocol:

  1. FULL-CHUNK CAUSAL DECODE (fixes the v1-v3 domain-mismatch bug).
     v1-v3 subsampled SHUFFLED latent frames (randperm) and/or decoded time-
     chunks independently (--fd_vae_chunk).  Wan's VAE is temporally CAUSAL
     (first latent → 1 frame, rest → 4, each frame conditioned on preceding
     latents), so out-of-context decodes produce pixels of a *different*
     video.  Reference stats decode full 49-frame chunks → FD was measuring
     decode-protocol mismatch, not sample quality.  v4 always decodes the
     full latent chunk, then subsamples fd_num_frames in PIXEL space.
     --fd_vae_chunk is REMOVED (it was itself a causality break).

  2. QUEUE WARM-UP GATE (--fd_min_pop).
     The queue starts empty; with population n < feat_dim(768) the covariance
     is rank-deficient → hundreds of ~0 eigenvalues → 1/(2√λ) exploding
     gradients through sqrt.  Until the queue holds fd_min_pop features the
     FD term is skipped (features still extracted no_grad + enqueued), so we
     never backprop through a garbage covariance estimate.  This replaces the
     reference repo's fill_all_queues prefill: warm-up features come from the
     SAME gated rollout as training-time features (see 3), so cold-start and
     steady-state queue populations are the same distribution by construction.

  3. SINGLE-PHASE POPULATION (--fd_first_step_only, default on).
     FD (features + enqueue) only when the rollout exits at the FIRST
     denoising step = the pure 1-NFE output that deployment uses.  Mixing
     exit phases lets sharp late-exit samples dilute the t=1000 deficiency
     in the pooled stats.  exit_idx is drawn from a step-seeded RNG SHARED
     across ranks so the gate is collective-consistent (all ranks agree on
     whether to all_gather).  Trivially always-on at num_inference_steps=1;
     at N>1 note this synchronizes the exit phase across ranks per step.

  4. EIGENVALUE CLAMP (--fd_eval_clamp, default 1e-6).
     Safety net for near-singular covariances: eigendirections below the
     floor get zero gradient instead of an exploding one (fd_loss_v2.py
     eval_clamp_min).

Reference stats:  regenerate with the updated precompute_fd_stats.py
    (cross-video shuffled chunks + pixel-space frame stride → diverse Σ_ref):
    ./data/cartoon_15s/fd_stats_dinov2_v2.npz
    (old fd_stats_dinov2.npz remains valid in *domain* — full-chunk roundtrip
    decode — but its frames are consecutive/correlated; prefer the v2 file.)

  5. ENQUEUE THINNING (--fd_enqueue_per_chunk).
     Frames of one chunk are highly correlated; enqueueing all fd_num_frames
     per chunk means a 5000-slot queue holds few independent VIDEO-level
     units, so Σ_student estimates within-video variance of a handful of
     clips.  v4 keeps the full fd_num_frames in the GRADIENT batch but
     enqueues only fd_enqueue_per_chunk of them → more distinct videos per
     queue at the same size.

KNOWN LIMITATION (deliberate, not a bug): per-frame DINOv2 FD only constrains
the APPEARANCE marginal — it is blind to temporal artifacts (flicker, motion
realism) that a spatio-temporal GAN discriminator would also punish.  So this
replaces GAN's appearance-realism role, not all of it.  Diagnostic plan: watch
VBench temporal-flickering / motion-smoothness vs the GAN run; if those drop,
add an FD term in a temporal representation space (V-JEPA / InternVideo) —
that is a ready-made ablation for the paper, not a v4 blocker.

NOTE ON NAMING: run scripts lora/wan1.3b_dmd_fd_v4-v7.sh are RUN numbers on
the v2/v3 CODE; this file is CODE v4.  Next run = v8 (wan1.3b_dmd_fd_v8_*.sh).
"""
import argparse, gc, os, random, sys, time
import accelerate
import decord
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.utils.checkpoint
from PIL import Image
from tqdm import tqdm

# Make sibling files importable regardless of cwd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from diffsynth.pipelines.wan_video import (
    WanVideoPipeline,
    ModelConfig,
    model_fn_wan_video,
)
from diffsynth.core import load_state_dict
from peft import LoraConfig, inject_adapter_in_model
from safetensors.torch import save_file

from dmd_utils import (
    DEFAULT_DENOISING_STEPS,
    NUM_TRAIN_TIMESTEPS,
    timestep_to_sigma,
    add_noise_flow,
    velocity_to_x0,
    sample_generator_timestep,
    sample_critic_timestep,
    compute_critic_loss,
    compute_dmd_gradient,
    compute_dmd_loss,
    cfg_real_x0,
)
from train_chunk_aware import ChunkAwareDataset
from fd_loss_v2 import (
    FeatureQueue, VideoFeatureExtractor,
    compute_fd_loss_normalized, compute_mmd_loss_normalized,
    precompute_sigma_ref_sqrt, load_fd_stats, load_fd_reference_features,
    diff_all_gather, sample_crop_offsets, crop_frames,
)


# ===========================================================================
# Pipeline construction
# ===========================================================================
WAN_MODEL_CONFIGS = [
    ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-1.3B-Control",
                origin_file_pattern="diffusion_pytorch_model*.safetensors"),
    ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-1.3B-Control",
                origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
    ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-1.3B-Control",
                origin_file_pattern="Wan2.1_VAE.pth"),
    ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-1.3B-Control",
                origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
]
TOKENIZER_CONFIG = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B",
                               origin_file_pattern="google/umt5-xxl/")


def build_pipe(device, dtype=torch.bfloat16):
    return WanVideoPipeline.from_pretrained(
        torch_dtype=dtype,
        device=device,
        model_configs=WAN_MODEL_CONFIGS,
        tokenizer_config=TOKENIZER_CONFIG,
    )


def fuse_sink_lora_into_pipe(pipe, sink_lora_path, alpha=1.0):
    """Load the sink LoRA and merge into the DiT. After this the DiT weights
    have moved by alpha * (lora_B @ lora_A). The DiT no longer carries LoRA
    modules — it's just a slightly different base."""
    pipe.load_lora(pipe.dit, lora_config=sink_lora_path, alpha=alpha)
    return pipe


def add_trainable_lora(dit, target_modules, rank=32, init_zero_B=True):
    """Add a fresh trainable LoRA on top of an already-fused DiT.
    With init_zero_B=True (peft default), the LoRA's initial effect is 0 so
    the model behaves exactly like its current (sink-fused) state at step 0."""
    lora_config = LoraConfig(
        r=rank, lora_alpha=rank,
        target_modules=target_modules,
    )
    dit = inject_adapter_in_model(lora_config, dit)
    return dit


def trainable_params(dit):
    return [p for p in dit.parameters() if p.requires_grad]


def all_reduce_grads(params, num_processes):
    """Manually sync gradients across DDP ranks. Cheap for LoRA (~80MB).
    Skips ranks==1 (single-GPU) automatically."""
    if num_processes <= 1:
        return
    for p in params:
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad.div_(num_processes)


def _clear_cuda_cache(device):
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()


def release_pipe_modules(pipe, module_names, device, label="", is_main=False):
    """Drop modules that are never used from a pipe to free GPU memory."""
    released = []
    for name in module_names:
        module = getattr(pipe, name, None)
        if module is not None:
            setattr(pipe, name, None)
            released.append(name)
            del module
    if released:
        gc.collect()
        _clear_cuda_cache(device)
        if is_main:
            print(f"[memory] released {label}: {', '.join(released)}")


def move_pipe_modules(pipe, module_names, target_device, cache_device=None):
    """Move existing modules without changing object identity or optimizer refs."""
    moved = False
    for name in module_names:
        module = getattr(pipe, name, None)
        if module is not None:
            module.to(target_device)
            moved = True
    if moved and cache_device is not None:
        _clear_cuda_cache(cache_device)
    return moved


def offload_pipe_modules_to_cpu(pipe, module_names, device):
    """Move modules off GPU while keeping their Python/optimizer refs valid."""
    return move_pipe_modules(pipe, module_names, "cpu", cache_device=device)


def onload_pipe_modules_to_device(pipe, module_names, device):
    """Bring modules back to the active accelerator right before use."""
    _clear_cuda_cache(device)
    return move_pipe_modules(pipe, module_names, device)


def _unique_cuda_indices(devices):
    indices = []
    for dev in devices:
        if dev is None:
            continue
        dev = torch.device(dev)
        if dev.type != "cuda":
            continue
        idx = torch.cuda.current_device() if dev.index is None else dev.index
        if idx not in indices:
            indices.append(idx)
    return indices


def reset_cuda_memory_peaks(devices):
    if not torch.cuda.is_available():
        return
    for idx in _unique_cuda_indices(devices):
        torch.cuda.reset_peak_memory_stats(idx)


def log_cuda_memory(label, devices, enabled=True, is_main=True):
    if not enabled or not is_main or not torch.cuda.is_available():
        return
    parts = []
    for idx in _unique_cuda_indices(devices):
        torch.cuda.synchronize(idx)
        free, total = torch.cuda.mem_get_info(idx)
        alloc = torch.cuda.memory_allocated(idx)
        reserved = torch.cuda.memory_reserved(idx)
        peak = torch.cuda.max_memory_allocated(idx)
        gib = 1024 ** 3
        parts.append(
            f"cuda:{idx} used={(total - free) / gib:.2f}/{total / gib:.2f}G "
            f"alloc={alloc / gib:.2f}G reserved={reserved / gib:.2f}G "
            f"peak={peak / gib:.2f}G"
        )
    if parts:
        print(f"[mem] {label}: " + " | ".join(parts), flush=True)


class ResumableSampler:
    """Wraps a base sampler and, ONLY on epoch `skip_epoch`, drops the first
    `skip_first` indices — at the index level. The DataLoader never fetches the
    skipped samples, so resume does NOT decode their videos (the slow part).
    Just advancing integers, not 49-frame decords."""
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
                next(it, None)        # advance index only — no __getitem__/decode
        return it

    def __len__(self):
        if self._epoch == self.skip_epoch:
            return max(0, len(self.base) - self.skip_first)
        return len(self.base)


def save_lora_state_dict(dit, out_path, remove_prefix=None):
    """Save only the trainable LoRA params as a safetensors file."""
    state_dict = {}
    for name, param in dit.state_dict().items():
        if "lora_A" in name or "lora_B" in name:
            if remove_prefix and name.startswith(remove_prefix):
                name = name[len(remove_prefix):]
            state_dict[name] = param.detach().cpu().contiguous()
    save_file(state_dict, out_path)


def load_lora_into_trainable(dit, lora_state_dict):
    """Load a LoRA state dict into the trainable LoRA modules already injected
    via add_trainable_lora. Tolerant of missing keys (returns load_result)."""
    # peft uses 'lora_A.default.weight' naming, but the saved files may have
    # 'lora_A.weight'. Normalize.
    fixed = {}
    for k, v in lora_state_dict.items():
        if "lora_A.weight" in k:
            k = k.replace("lora_A.weight", "lora_A.default.weight")
        if "lora_B.weight" in k:
            k = k.replace("lora_B.weight", "lora_B.default.weight")
        fixed[k] = v
    return dit.load_state_dict(fixed, strict=False)


# ===========================================================================
# Conditioning encoder (shared across 3 models)
# ===========================================================================
@torch.no_grad()
def encode_prompt(pipe, prompt, dtype, device):
    """Replicates WanVideoUnit_PromptEmbedder.encode_prompt (wan_video.py:442)."""
    pipe.load_models_to_device(["text_encoder"])
    pipe.text_encoder.to(device)
    ids, mask = pipe.tokenizer(prompt, return_mask=True, add_special_tokens=True)
    ids = ids.to(device)
    mask = mask.to(device)
    seq_lens = mask.gt(0).sum(dim=1).long()
    prompt_emb = pipe.text_encoder(ids, mask).to(dtype=dtype)
    for i, v in enumerate(seq_lens):
        prompt_emb[:, v:] = 0
    return prompt_emb


@torch.no_grad()
def encode_batch(pipe, batch, dtype, device, no_recent=False, vae_device=None):
    """Use a pipe's encoders (T5/VAE/CLIP) to encode one batch from
    ChunkAwareDataset into the tensors model_fn_wan_video expects.

    Returns dict with: prompt_embed, target_latents, control_latents,
                       reference_latents (sink+recent on T-dim), clip_feature
    """
    vae_device = torch.device(vae_device or device)
    model_device = torch.device(device)

    # ---- T5 text ----
    prompt_emb = encode_prompt(pipe, batch["prompt"], dtype, model_device)

    # ---- VAE: target video -> latent ----
    pipe.vae.to(vae_device)
    target_video = pipe.preprocess_video(batch["video"]).to(dtype=dtype, device=vae_device)
    target_latents = pipe.vae.encode(target_video, device=vae_device).to(
        dtype=dtype, device=model_device,
    )

    # ---- VAE: control video (pose) -> latent ----
    # Wan-Fun-Control expects y to have (dit.in_dim - 2 * latent_channels) channels:
    # y = [control_latent(16) | zero_padding(16)] for V1.1 1.3B (in_dim=48).
    # See WanVideoUnit_FunControl.process (wan_video.py:540).
    control_video = pipe.preprocess_video(batch["control_video"]).to(dtype=dtype, device=vae_device)
    control_latents = pipe.vae.encode(control_video, device=vae_device).to(
        dtype=dtype, device=model_device,
    )
    y_dim = pipe.dit.in_dim - control_latents.shape[1] - target_latents.shape[1]
    if y_dim > 0:
        y_pad = torch.zeros(
            (control_latents.shape[0], y_dim, control_latents.shape[2],
             control_latents.shape[3], control_latents.shape[4]),
            dtype=dtype, device=model_device,
        )
        control_latents = torch.cat([control_latents, y_pad], dim=1)   # [B, in_dim - 16, T, h, w]

    # ---- VAE: sink + recent ref images → ref latent ----
    # Build TWO reference latents: one with the augmented recent (for the
    # student, drift-robustness) and one with the clean recent (for the teacher
    # in asymmetric mode — a reliable, in-distribution target).
    sink_pil = batch["sink_reference_image"][0]
    recent_aug_pil = batch["reference_image"][0]                  # augmented
    recent_clean_pil = batch.get("reference_image_clean", [recent_aug_pil])[0]
    w, h = sink_pil.size

    def _ref_latent(recent_pil):
        sink_video = pipe.preprocess_video([sink_pil.resize((w, h))]).to(dtype=dtype, device=vae_device)
        sink_latent = pipe.vae.encode(sink_video, device=vae_device).to(
            dtype=dtype, device=model_device,
        )
        if no_recent:
            return sink_latent           # sink-only (1 frame) for sinkonly teacher
        recent_video = pipe.preprocess_video([recent_pil.resize((w, h))]).to(dtype=dtype, device=vae_device)
        recent_latent = pipe.vae.encode(recent_video, device=vae_device).to(
            dtype=dtype, device=model_device,
        )
        # Concat sink at T=0, recent at T=1 (matches our patched FunReference)
        return torch.cat([sink_latent, recent_latent], dim=2)

    reference_latents_aug   = _ref_latent(recent_aug_pil)
    reference_latents_clean = _ref_latent(recent_clean_pil)

    # ---- CLIP: two versions so the loop can route per recent_augment_mode ----
    # no_recent → use sink for clip (matches WanVideoUnit_FunReference fallback).
    # Otherwise: one from aug recent (student side), one from clean (teacher).
    clip_feature_aug = None
    clip_feature_clean = None
    if pipe.image_encoder is not None:
        pipe.image_encoder.to(model_device)
        def _clip(pil):
            return pipe.image_encoder.encode_image([pipe.preprocess_image(pil)]).to(
                dtype=dtype, device=model_device,
            )
        if no_recent:
            clip_feature_aug = clip_feature_clean = _clip(sink_pil)
        else:
            clip_feature_aug = _clip(recent_aug_pil)
            clip_feature_clean = (clip_feature_aug if recent_aug_pil is recent_clean_pil
                                  else _clip(recent_clean_pil))

    return dict(
        prompt_embed=prompt_emb,
        target_latents=target_latents,
        control_latents=control_latents,
        reference_latents_aug=reference_latents_aug,
        reference_latents_clean=reference_latents_clean,
        clip_feature_aug=clip_feature_aug,
        clip_feature_clean=clip_feature_clean,
    )


# ===========================================================================
# Single DiT forward — wraps model_fn_wan_video to expose velocity output
# ===========================================================================
def dit_forward(dit, x_t, t_int, prompt_embed, control_latents,
                reference_latents, clip_feature, device, dtype,
                use_gradient_checkpointing=False):
    """Run a single dit forward with all our conditioning.

    Args:
        dit:               the WanModel (student / teacher / critic)
        x_t:               noisy target latent [B, 16, T, H, W]
        t_int:             integer timestep (0..1000)
        prompt_embed:      T5 text embedding
        control_latents:   pose latent [B, 16, T, H, W]  → passed as `y`
        reference_latents: sink+recent latent [B, 16, 2, h, w]
        clip_feature:      CLIP image feature
        use_gradient_checkpointing: enable activation checkpointing for the
            transformer blocks. Critical for fitting 3 DiTs in memory.

    Returns the velocity prediction (same shape as x_t).
    """
    # bf16 to match dit dtype (model_fn_wan_video calls dit.time_embedding on
    # sinusoidal_embedding_1d(timestep) without a dtype cast).
    timestep = torch.tensor([t_int], dtype=dtype, device=device)
    velocity = model_fn_wan_video(
        dit=dit,
        latents=x_t.to(dtype=dtype, device=device),
        timestep=timestep,
        context=prompt_embed.to(dtype=dtype, device=device),
        clip_feature=clip_feature,
        y=control_latents.to(dtype=dtype, device=device),
        reference_latents=reference_latents.to(dtype=dtype, device=device),
        use_gradient_checkpointing=use_gradient_checkpointing,
    )
    return velocity


# ===========================================================================
# Self-forcing rollout (Causal Forcing++ Stage-3 style)
# ===========================================================================
def rollout_student(
    dit, denoising_step_list, target_shape,
    prompt_embed, control_latents, reference_latents, clip_feature,
    device, dtype, use_gradient_checkpointing=False, exit_idx=None,
):
    """Roll the student through the full few-step denoising schedule, starting
    from PURE NOISE. Random exit step → only one forward keeps grad; everything
    earlier runs under no_grad with re-noising between steps. The exit step's
    x0 estimate is returned as x_pred (the model's "final" output at that step).

    exit_idx: caller-supplied exit step. v4 passes a step-seeded, rank-shared
    value so the FD gate downstream is collective-consistent; None keeps the
    v3 behavior (per-rank uniform draw).

    Returns: (x_pred, exit_t)  where exit_t is the timestep at the exit step.

    Mirrors pipeline/bidirectional_training.py:inference_with_trajectory in the
    upstream Causal-Forcing repo, simplified for our chunk-based setting.
    """
    N = len(denoising_step_list)
    if exit_idx is None:
        exit_idx = random.randrange(N)    # uniform over 0..N-1

    # Start from pure noise — KEY DIFFERENCE from current train_dmd.py which
    # noises GT. The whole rollout is data-free for the target latent.
    noisy = torch.randn(target_shape, dtype=dtype, device=device)

    for idx, t in enumerate(denoising_step_list):
        sigma_t = float(t) / NUM_TRAIN_TIMESTEPS
        if idx == exit_idx:
            # Grad step: one forward with autograd, then we stop the loop.
            v = dit_forward(
                dit, noisy, t, prompt_embed,
                control_latents, reference_latents, clip_feature, device, dtype,
                use_gradient_checkpointing=use_gradient_checkpointing,
            )
            x_pred = velocity_to_x0(v, noisy, sigma_t)
            return x_pred, t
        # No-grad step: forward → x0 → re-noise to NEXT timestep.
        with torch.no_grad():
            v = dit_forward(
                dit, noisy, t, prompt_embed,
                control_latents, reference_latents, clip_feature, device, dtype,
            )
            x0 = velocity_to_x0(v, noisy, sigma_t)
            next_t = denoising_step_list[idx + 1]
            sigma_next = float(next_t) / NUM_TRAIN_TIMESTEPS
            fresh = torch.randn_like(x0)
            # Re-noise the clean estimate to the next timestep (consistency-
            # sampler style, matches CF++ scheduler.add_noise).
            noisy = sigma_next * fresh + (1 - sigma_next) * x0

    # Should not reach here (exit_idx < N).
    raise RuntimeError("rollout_student: exit step not consumed")


# ===========================================================================
# Argument parser
# ===========================================================================
def get_parser():
    p = argparse.ArgumentParser()
    # data
    p.add_argument("--dataset_metadata_path", required=True)
    p.add_argument("--height", type=int, default=832)
    p.add_argument("--width", type=int, default=480)
    p.add_argument("--num_frames", type=int, default=49)
    p.add_argument("--dataset_repeat", type=int, default=1)
    p.add_argument("--recent_aug_strength", type=float, default=0.5)
    # model
    p.add_argument("--teacher_lora_path", default="",
                   help="Sink LoRA fused into all 3 models' base. Empty / 'none' "
                        "→ vanilla base teacher (student must learn sink itself; "
                        "without regression it can't — experimental).")
    p.add_argument("--lora_rank", type=int, default=32)
    p.add_argument("--lora_target_modules", default="q,k,v,o,ffn.0,ffn.2")
    p.add_argument("--no_recent", action="store_true",
                   help="Sink-only conditioning (no recent frame). Use when the "
                        "teacher is a sinkonly model trained without recent.")
    # recent-frame augmentation routing (sink/drift experiments)
    p.add_argument("--recent_augment_mode", choices=["off", "symmetric", "asymmetric"],
                   default="symmetric",
                   help="off: clean recent everywhere. symmetric: augmented recent "
                        "for all (needs augment-trained teacher). asymmetric: clean "
                        "recent → teacher, augmented → student/critic (teaches drift "
                        "robustness via DMD, no regression needed).")
    # DMD specifics
    p.add_argument("--num_inference_steps", type=int, default=4,
                   help="Student's few-step count. The denoising timesteps are "
                        "derived from the scheduler (shift=flow_shift) so they "
                        "match inference EXACTLY.")
    p.add_argument("--denoising_step_list", default="",
                   help="Optional manual override (comma-sep). Leave empty to "
                        "derive from the scheduler (recommended).")
    p.add_argument("--dfake_gen_update_ratio", type=int, default=5)
    p.add_argument("--flow_shift", type=float, default=5.0)
    p.add_argument("--dmd_normalize_grad", action="store_true", default=True)
    p.add_argument("--real_guidance_scale", type=float, default=3.0,
                   help="CFG scale on the teacher (real score). Helios default 3.0. "
                        "0 disables CFG (matches unguided teacher → blurrier).")
    p.add_argument("--negative_prompt", default=(
        "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
        "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，"
        "畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"),
        help="Used as the CFG uncond branch for the teacher (real score).")
    # ─── FD-loss ───
    p.add_argument("--fd_stats_path", default="",
                   help="Precomputed whole-frame reference stats .npz from "
                        "precompute_fd_stats.py. Required for frechet or "
                        "fd_weight > 0; crop-only runs (--fd_weight 0 "
                        "--fd_crop_weight > 0) don't need it.")
    p.add_argument("--fd_weight", type=float, default=1.0,
                   help="Weight on NORMALIZED FD loss (≈1 magnitude). v1 used "
                        "0.1 on un-normalized FD which was wrong; default 1.0 here.")
    p.add_argument("--fd_feature_model", default="dinov2_vitb14",
                   help="Frozen backbone for per-frame features.")
    p.add_argument("--fd_device", default="",
                   help="Optional fixed device for the FD/MMD path (student VAE, "
                        "DINO, reference features), e.g. cuda:1. For DDP, prefer "
                        "--fd_device_offset so ranks do not fight over one GPU.")
    p.add_argument("--fd_device_offset", type=int, default=-1,
                   help="DDP model-parallel pairing: if >=0 and main rank device is "
                        "cuda:i, place FD/MMD on cuda:i+offset. Example on 8 GPUs: "
                        "torchrun --nproc_per_node=4 with --fd_device_offset 4 uses "
                        "pairs 0/4, 1/5, 2/6, 3/7.")
    p.add_argument("--fd_objective", choices=["frechet", "mmd"], default="frechet",
                   help="frechet = existing FD covariance objective with queue; "
                        "mmd = RDM-lite exact generated repulsion + frozen "
                        "real-feature attraction, no generated queue.")
    # ─── crop-view MMD (second representation view; fixes blur-blindness) ───
    # The whole-frame path squash-resizes 832x480 → 224² (3.7× low-pass), so
    # its features barely see blur. Crop view: K random 224px crops per frame
    # at NATIVE resolution, matched against crop-protocol reference stats.
    # fd_weight controls the whole-frame (global) term; fd_crop_weight the
    # crop term. crop-only = --fd_weight 0 --fd_crop_weight w.
    p.add_argument("--fd_crop_stats_path", default="",
                   help="Reference stats .npz computed with "
                        "precompute_fd_stats.py --crops_per_frame K. Required "
                        "when --fd_crop_weight > 0.")
    p.add_argument("--fd_crop_weight", type=float, default=0.0,
                   help="Weight on the crop-view MMD term (mmd objective "
                        "only). 0 disables the crop view.")
    p.add_argument("--fd_crops_per_frame", type=int, default=4,
                   help="Random native-res crops per sampled frame. With "
                        "fd_num_frames=16 and 8 ranks: 16×4×8 = 512 crop "
                        "samples/step.")
    p.add_argument("--fd_crop_size", type=int, default=224)
    p.add_argument("--fd_crop_mmd_bandwidth", type=float, default=0.0,
                   help="Override RBF bandwidth for the crop view; 0 uses the "
                        "value saved in the crop stats file.")
    p.add_argument("--fd_mmd_max_ref_features", type=int, default=20000,
                   help="Max saved real features loaded from fd_stats for MMD.")
    p.add_argument("--fd_mmd_ref_chunk", type=int, default=4096,
                   help="Chunk size for MMD generated-to-reference kernel matmul.")
    p.add_argument("--fd_mmd_bandwidth", type=float, default=0.0,
                   help="Override RBF bandwidth for MMD; 0 uses value saved in stats "
                        "or median heuristic on the loaded feature bank.")
    p.add_argument("--fd_queue_size", type=int, default=5000,
                   help="Ring-buffer size for detached features (larger → "
                        "more stable statistics, more memory).")
    p.add_argument("--fd_num_frames", type=int, default=16,
                   help="How many PIXEL frames per chunk to sample for FD "
                        "(sampled AFTER the full-chunk causal decode; the "
                        "latent chunk is always decoded whole).")
    p.add_argument("--fd_window_tail_bias", type=float, default=0.0,
                   help="Tail-biased window sampling: P(start j) ∝ 1 + "
                        "bias·j/J. Real frame statistics are position-"
                        "stationary but the 1-NFE student's are not — blur "
                        "grows with distance from the recent anchor (sharp "
                        "chunk head, blurry tail), and uniform sampling lets "
                        "sharp head frames dilute the pooled MMD statistics. "
                        "Biasing windows toward the tail concentrates the "
                        "gradient where the blur is, using the SAME reference "
                        "bank (stationarity). 0 = uniform (previous "
                        "behavior); 3.0 ≈ tail window 4x likelier than head, "
                        "head still covered.")
    p.add_argument("--fd_window_latents", type=int, default=4,
                   help="Causal-window grad decode: decode only this many "
                        "latent steps WITH autograd at a random position per "
                        "step; the preceding latents prime the causal cache "
                        "under no_grad. Window pixels are IDENTICAL to the "
                        "full decode (causality), so the reference-stats "
                        "domain still matches, but grad memory scales with "
                        "the window instead of the whole chunk — no tiling, "
                        "no spare FD GPU needed. 0 = full-chunk grad decode "
                        "(needs --fd_vae_tiled and/or --fd_device_offset).")
    p.add_argument("--fd_vae_tiled", action="store_true", default=True,
                   help="Use tiled Wan VAE decode for FD/MMD. Much lower GPU "
                        "peak, slower. Enabled by default because pixel-space "
                        "FD/MMD otherwise OOMs easily. Only used when "
                        "--fd_window_latents is 0.")
    p.add_argument("--no_fd_vae_tiled", dest="fd_vae_tiled", action="store_false",
                   help="Disable tiled FD/MMD VAE decode.")
    p.add_argument("--fd_vae_tile_size", type=int, nargs=2, default=(16, 16),
                   metavar=("H", "W"),
                   help="Latent tile size for FD/MMD VAE decode when tiled.")
    p.add_argument("--fd_vae_tile_stride", type=int, nargs=2, default=(8, 8),
                   metavar=("H", "W"),
                   help="Latent tile stride for FD/MMD VAE decode when tiled.")
    p.add_argument("--fd_enqueue_per_chunk", type=int, default=8,
                   help="How many of the fd_num_frames features to ENQUEUE "
                        "per chunk (gradient batch still uses all). Fewer → "
                        "more distinct videos per queue slot → Σ_student "
                        "reflects cross-video variance, not within-video. "
                        "0 = enqueue all.")
    p.add_argument("--fd_min_pop", type=int, default=1536,
                   help="Warm-up gate: skip the FD term (but keep extracting "
                        "no_grad + enqueueing) until the queue holds this many "
                        "features. Default 2×768 (DINOv2-B dim) so the pooled "
                        "covariance is comfortably full-rank before any FD "
                        "gradient flows.")
    p.add_argument("--fd_first_step_only", action="store_true", default=True,
                   help="Restrict FD (features, loss AND enqueue) to rollouts "
                        "exiting at the FIRST denoising step — the pure 1-NFE "
                        "output. Keeps the queue population single-phase. "
                        "No-op at num_inference_steps=1.")
    p.add_argument("--fd_eval_clamp", type=float, default=1e-6,
                   help="Floor for eigenvalues of Σ·Σ_ref before sqrt; "
                        "directions below it get ZERO gradient instead of an "
                        "exploding 1/(2·sqrt(λ)) one.")
    # ─── EMA (One-Forcing default) ───
    p.add_argument("--ema_decay", type=float, default=0.99)
    p.add_argument("--ema_start_step", type=int, default=200)
    # training
    p.add_argument("--learning_rate_student", type=float, default=5e-6)
    p.add_argument("--learning_rate_critic",  type=float, default=5e-6)
    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--global_step_offset", type=int, default=0)
    # checkpointing
    p.add_argument("--output_path", required=True)
    p.add_argument("--resume_student_from", default=None)
    p.add_argument("--resume_critic_from", default=None)
    p.add_argument("--use_gradient_checkpointing", action="store_true", default=True)
    p.add_argument("--memory_debug", action="store_true",
                   help="Print per-GPU memory at setup and early training phases.")
    p.add_argument("--memory_debug_steps", type=int, default=1,
                   help="How many initial local steps get detailed memory prints.")
    return p


# ===========================================================================
# Main
# ===========================================================================
def main():
    args = get_parser().parse_args()
    accelerator = accelerate.Accelerator()
    device = accelerator.device
    if torch.device(device).type == "cuda":
        device = torch.device("cuda", torch.cuda.current_device())
    dtype = torch.bfloat16
    is_main = accelerator.is_main_process
    num_proc = accelerator.num_processes
    target_modules = args.lora_target_modules.split(",")
    if args.fd_device_offset >= 0:
        if args.fd_device:
            raise ValueError("Use either --fd_device or --fd_device_offset, not both.")
        model_device = torch.device(device)
        if model_device.type != "cuda":
            raise ValueError("--fd_device_offset requires CUDA model devices.")
        model_idx = torch.cuda.current_device() if model_device.index is None else model_device.index
        fd_idx = model_idx + args.fd_device_offset
        if fd_idx >= torch.cuda.device_count():
            raise ValueError(
                f"--fd_device_offset {args.fd_device_offset} maps model cuda:{model_idx} "
                f"to cuda:{fd_idx}, but only {torch.cuda.device_count()} visible CUDA devices."
            )
        fd_device = torch.device("cuda", fd_idx)
    else:
        fd_device = torch.device(args.fd_device) if args.fd_device else torch.device(device)
        if args.fd_device and num_proc > 1:
            raise ValueError(
                "Fixed --fd_device with DDP makes all ranks fight over one GPU. "
                "Use --fd_device_offset instead."
            )
    if fd_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"FD device {fd_device} requested but CUDA is unavailable")
    if is_main:
        os.makedirs(args.output_path, exist_ok=True)
    accelerator.wait_for_everyone()

    rank = accelerator.process_index
    if is_main:
        print(f"[accel] num_processes={num_proc} (rank {rank}, device {device}, fd_device={fd_device})")
    reset_cuda_memory_peaks([device, fd_device])
    log_cuda_memory("start", [device, fd_device], args.memory_debug, is_main)

    # Shared seed BEFORE model building so peft's LoRA kaiming_uniform_ init
    # is bit-identical across ranks. Per-rank seed gets set AFTER model build.
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Empty / "none" → vanilla base teacher (no sink fused). Student then has to
    # learn sink itself, which pure DMD can't do without a regression loss —
    # this is the experimental "one-shot from base" config.
    use_sink = bool(args.teacher_lora_path) and args.teacher_lora_path.lower() != "none"
    if is_main:
        print(f"[setup] sink LoRA fused into base: {use_sink} "
              f"({'teacher=base+sink' if use_sink else 'teacher=VANILLA base'})")

    # ---------------- Build 3 pipes (each rank gets its own 3 DiTs) ----------------
    if is_main: print("[setup] building teacher (frozen) ...")
    teacher_pipe = build_pipe(device, dtype)
    if use_sink:
        fuse_sink_lora_into_pipe(teacher_pipe, args.teacher_lora_path)
    teacher_pipe.dit.requires_grad_(False)
    teacher_pipe.dit.eval()
    release_pipe_modules(teacher_pipe, ["vae", "image_encoder"], device,
                         label="teacher unused encoders", is_main=is_main)
    log_cuda_memory("after teacher build/release", [device, fd_device], args.memory_debug, is_main)

    if is_main: print("[setup] building student (trainable LoRA) ...")
    student_pipe = build_pipe(device, dtype)
    if use_sink:
        fuse_sink_lora_into_pipe(student_pipe, args.teacher_lora_path)
    student_pipe.dit = add_trainable_lora(student_pipe.dit, target_modules, args.lora_rank)
    student_pipe.dit.train()   # gradient_checkpoint_forward checks self.training
    if args.resume_student_from:
        load_lora_into_trainable(student_pipe.dit, load_state_dict(args.resume_student_from))
        if is_main: print(f"[resume] student LoRA from {args.resume_student_from}")
    if fd_device != torch.device(device):
        move_pipe_modules(student_pipe, ["vae"], fd_device, cache_device=device)
        if is_main:
            print(f"[memory] student VAE pinned to FD device: {fd_device}")
    offload_pipe_modules_to_cpu(student_pipe, ["text_encoder", "image_encoder"], device)
    log_cuda_memory("after student build/VAE placement", [device, fd_device], args.memory_debug, is_main)

    if is_main: print("[setup] building critic (trainable LoRA) ...")
    critic_pipe = build_pipe(device, dtype)
    if use_sink:
        fuse_sink_lora_into_pipe(critic_pipe, args.teacher_lora_path)
    critic_pipe.dit = add_trainable_lora(critic_pipe.dit, target_modules, args.lora_rank)
    critic_pipe.dit.train()
    if args.resume_critic_from:
        load_lora_into_trainable(critic_pipe.dit, load_state_dict(args.resume_critic_from))
        if is_main: print(f"[resume] critic LoRA from {args.resume_critic_from}")
    release_pipe_modules(critic_pipe, ["vae", "text_encoder", "image_encoder"], device,
                         label="critic unused encoders", is_main=is_main)
    log_cuda_memory("after critic build/release", [device, fd_device], args.memory_debug, is_main)

    # ---------------- Optimizers ----------------
    student_params = trainable_params(student_pipe.dit)
    critic_params  = trainable_params(critic_pipe.dit)
    if is_main:
        print(f"[setup] student trainable params: {sum(p.numel() for p in student_params)/1e6:.1f}M")
        print(f"[setup] critic  trainable params: {sum(p.numel() for p in critic_params )/1e6:.1f}M")

    student_optimizer = torch.optim.AdamW(student_params, lr=args.learning_rate_student, weight_decay=1e-2)
    critic_optimizer  = torch.optim.AdamW(critic_params,  lr=args.learning_rate_critic,  weight_decay=1e-2)

    # ---------------- FD-loss setup ----------------
    if is_main: print(f"[setup] loading FD feature extractor: {args.fd_feature_model} on {fd_device}")
    # torch.hub download is NOT multi-process safe (concurrent zip extract +
    # rmtree in one cache dir). Rank 0 downloads first; others load from the
    # warm cache after the barrier.
    if num_proc > 1 and not is_main:
        accelerator.wait_for_everyone()
    fd_extractor = VideoFeatureExtractor(args.fd_feature_model, dtype=dtype).to(fd_device)
    if num_proc > 1 and is_main:
        accelerator.wait_for_everyone()
    needs_global_stats = args.fd_objective == "frechet" or args.fd_weight > 0
    if needs_global_stats and not args.fd_stats_path:
        raise ValueError("--fd_stats_path is required for --fd_objective "
                         "frechet or --fd_weight > 0")
    if is_main and needs_global_stats:
        print(f"[setup] loading FD/RDM reference stats: {args.fd_stats_path}")
    mu_ref = sigma_ref = sigma_ref_sqrt = None
    fd_queue = None
    mmd_ref_feats = mmd_bandwidth = None
    if args.fd_objective == "frechet":
        mu_ref, sigma_ref = load_fd_stats(args.fd_stats_path)
        mu_ref    = mu_ref.to(fd_device)
        sigma_ref = sigma_ref.to(fd_device)
        sigma_ref_sqrt = precompute_sigma_ref_sqrt(sigma_ref).to(fd_device)
        fd_queue = FeatureQueue(args.fd_queue_size, fd_extractor.feat_dim).to(fd_device)
    elif args.fd_weight > 0:
        # Global (whole-frame) MMD bank — only when the global term is active;
        # crop-only runs (--fd_weight 0) must not depend on this file's bank.
        mmd_ref_feats, mmd_bandwidth, g_proto = load_fd_reference_features(
            args.fd_stats_path, max_features=args.fd_mmd_max_ref_features,
        )
        if g_proto is None:
            if is_main:
                print("[setup] WARNING: global stats file has no protocol stamp "
                      "(legacy) — cannot verify it is whole-frame/resize.")
        else:
            if g_proto.get("crops_per_frame", 0) != 0:
                raise ValueError(
                    f"--fd_stats_path {args.fd_stats_path} was computed with the "
                    f"CROP protocol (crops_per_frame="
                    f"{g_proto['crops_per_frame']}) but is being used as the "
                    f"whole-frame/global bank. Wrong file?")
            if g_proto.get("feature_model") != args.fd_feature_model:
                raise ValueError(
                    f"global stats encoder {g_proto.get('feature_model')} != "
                    f"--fd_feature_model {args.fd_feature_model}")
            if (g_proto.get("height"), g_proto.get("width")) != (args.height, args.width):
                raise ValueError(
                    f"global stats resolution {g_proto.get('height')}x"
                    f"{g_proto.get('width')} != training {args.height}x{args.width}")
        mmd_ref_feats = mmd_ref_feats.to(fd_device)
        if args.fd_mmd_bandwidth > 0:
            mmd_bandwidth = torch.tensor(args.fd_mmd_bandwidth, dtype=torch.float32, device=fd_device)
        elif mmd_bandwidth is not None:
            mmd_bandwidth = mmd_bandwidth.to(fd_device)

    # Crop-view MMD reference (second representation view; blur-sensitive).
    crop_ref_feats = crop_bandwidth = None
    if args.fd_crop_weight > 0:
        if args.fd_objective != "mmd":
            raise ValueError("--fd_crop_weight requires --fd_objective mmd")
        if args.fd_crops_per_frame <= 0:
            raise ValueError("--fd_crop_weight > 0 requires "
                             "--fd_crops_per_frame > 0")
        if not args.fd_crop_stats_path:
            raise ValueError("--fd_crop_weight > 0 needs --fd_crop_stats_path "
                             "(precompute_fd_stats.py --crops_per_frame K)")
        crop_ref_feats, crop_bandwidth, c_proto = load_fd_reference_features(
            args.fd_crop_stats_path, max_features=args.fd_mmd_max_ref_features,
        )
        # Crop stats are always generated by protocol-stamping code, so a
        # missing/mismatched stamp is a hard error — a whole-frame bank
        # matched against crop features pulls toward the wrong distribution.
        if c_proto is None:
            raise ValueError(
                f"--fd_crop_stats_path {args.fd_crop_stats_path} has no "
                f"protocol stamp — regenerate with the current "
                f"precompute_fd_stats.py --crops_per_frame K.")
        if c_proto.get("crops_per_frame", 0) <= 0:
            raise ValueError(
                f"--fd_crop_stats_path {args.fd_crop_stats_path} was computed "
                f"WITHOUT crops (whole-frame protocol). Wrong file?")
        if c_proto.get("crop_size") != args.fd_crop_size:
            raise ValueError(
                f"crop stats crop_size={c_proto.get('crop_size')} != "
                f"--fd_crop_size {args.fd_crop_size}")
        if c_proto.get("feature_model") != args.fd_feature_model:
            raise ValueError(
                f"crop stats encoder {c_proto.get('feature_model')} != "
                f"--fd_feature_model {args.fd_feature_model}")
        if (c_proto.get("height"), c_proto.get("width")) != (args.height, args.width):
            # Same crop size at a different source resolution = different
            # object scale inside the crop → different feature distribution.
            raise ValueError(
                f"crop stats resolution {c_proto.get('height')}x"
                f"{c_proto.get('width')} != training {args.height}x{args.width}")
        if is_main and c_proto.get("vae_roundtrip") is False:
            print("[setup] WARNING: crop stats were computed WITHOUT the VAE "
                  "roundtrip — reference lives in raw-pixel domain, student "
                  "features in VAE-decode domain.")
        if is_main and c_proto.get("crops_per_frame") != args.fd_crops_per_frame:
            # K only changes within-frame correlation of the bank, not the
            # marginal crop distribution — warn, don't fail.
            print(f"[setup] note: crop stats K={c_proto.get('crops_per_frame')} "
                  f"!= --fd_crops_per_frame {args.fd_crops_per_frame} (ok).")
        crop_ref_feats = crop_ref_feats.to(fd_device)
        if args.fd_crop_mmd_bandwidth > 0:
            crop_bandwidth = torch.tensor(args.fd_crop_mmd_bandwidth,
                                          dtype=torch.float32, device=fd_device)
        elif crop_bandwidth is not None:
            crop_bandwidth = crop_bandwidth.to(fd_device)
        if is_main:
            bw_c = 'median' if crop_bandwidth is None else f"{float(crop_bandwidth):.4f}"
            print(f"[setup] crop-view MMD: ref={tuple(crop_ref_feats.shape)}, "
                  f"bandwidth={bw_c}, crops/frame={args.fd_crops_per_frame}, "
                  f"crop={args.fd_crop_size}px, weight={args.fd_crop_weight}")
    log_cuda_memory("after FD/MMD setup", [device, fd_device], args.memory_debug, is_main)
    if is_main:
        if args.fd_objective == "frechet":
            print(f"[setup] FD objective=frechet queue: size={args.fd_queue_size}, "
                  f"feat_dim={fd_extractor.feat_dim}, frames/chunk={args.fd_num_frames} "
                  f"(enqueue {args.fd_enqueue_per_chunk or 'all'}/chunk), "
                  f"weight={args.fd_weight}, min_pop={args.fd_min_pop}, "
                  f"first_step_only={args.fd_first_step_only}, "
                  f"eval_clamp={args.fd_eval_clamp}")
        else:
            bw = 'median' if mmd_bandwidth is None else f"{float(mmd_bandwidth):.4f}"
            g_ref = 'OFF (crop-only)' if mmd_ref_feats is None else tuple(mmd_ref_feats.shape)
            print(f"[setup] FD objective=mmd global_ref={g_ref}, "
                  f"bandwidth={bw}, ref_chunk={args.fd_mmd_ref_chunk}, "
                  f"frames/chunk={args.fd_num_frames}, weight={args.fd_weight}, "
                  f"first_step_only={args.fd_first_step_only}, "
                  f"vae_tiled={args.fd_vae_tiled}, "
                  f"tile={tuple(args.fd_vae_tile_size)}/{tuple(args.fd_vae_tile_stride)}")

    # ---------------- Resume optimizer state (AdamW moments) ----------------
    # The state sidecar sits next to the student LoRA: step-N.safetensors →
    # step-N_state.pt. Restoring it avoids the cold-start momentum re-warm.
    if args.resume_student_from:
        # EMA (LoRA-format safetensors)
        ema_resume_path = args.resume_student_from.replace(".safetensors", "_ema.safetensors")
        if os.path.isfile(ema_resume_path):
            from diffsynth.core import load_state_dict as _lsd
            ema_sd = _lsd(ema_resume_path)
            main._ema_state = {}
            for n, p in student_pipe.dit.named_parameters():
                if p.requires_grad and n in ema_sd:
                    main._ema_state[n] = ema_sd[n].to(device=p.device, dtype=p.dtype)
            if is_main:
                print(f"[resume] EMA state from {ema_resume_path} ({len(main._ema_state)} tensors)")
        elif is_main:
            print(f"[resume] EMA NOT loaded (no {ema_resume_path}) — will init when started.")

        state_path = args.resume_student_from.replace(".safetensors", "_state.pt")
        if os.path.isfile(state_path):
            ckpt = torch.load(state_path, map_location=device)
            student_optimizer.load_state_dict(ckpt["student_optimizer"])
            critic_optimizer.load_state_dict(ckpt["critic_optimizer"])
            if is_main:
                print(f"[resume] optimizer state from {state_path} "
                      f"(saved at step {ckpt.get('global_step', '?')})")
        elif is_main:
            print(f"[resume] WARNING: no optimizer state at {state_path} — "
                  f"AdamW moments start cold (expect a brief re-warm).")

    # ---------------- Derive denoising timesteps from the scheduler ----------------
    # Guarantees train timesteps == inference timesteps (sigma = t/1000).
    if args.denoising_step_list.strip():
        denoising_step_list = [float(x) for x in args.denoising_step_list.split(",")]
    else:
        student_pipe.scheduler.set_timesteps(
            num_inference_steps=args.num_inference_steps, shift=args.flow_shift,
        )
        denoising_step_list = [float(t) for t in student_pipe.scheduler.timesteps]
    if is_main:
        print(f"[setup] denoising_step_list (shift={args.flow_shift}): "
              f"{[round(t, 1) for t in denoising_step_list]}")

    # Negative prompt embedding for the teacher's CFG uncond branch. Fixed text
    # → encode ONCE and reuse every step.
    neg_prompt_embed = None
    if args.real_guidance_scale != 0:
        neg_prompt_embed = encode_prompt(teacher_pipe, args.negative_prompt, dtype, device)
        if is_main:
            print(f"[setup] teacher CFG enabled (real_guidance_scale={args.real_guidance_scale})")
    release_pipe_modules(teacher_pipe, ["text_encoder"], device,
                         label="teacher text encoder after CFG setup", is_main=is_main)
    # Keep non-student DiTs off GPU outside their score calls. FD/MMD needs
    # student VAE + DINO activations, and two extra DiTs leave almost no headroom.
    offload_pipe_modules_to_cpu(teacher_pipe, ["dit"], device)
    offload_pipe_modules_to_cpu(critic_pipe, ["dit"], device)
    if is_main:
        print("[memory] teacher/critic DiTs are onloaded only for score calls")
    log_cuda_memory("after initial teacher/critic DiT offload", [device, fd_device], args.memory_debug, is_main)

    # Models are built with identical weights across ranks. Now switch to a
    # per-rank seed so noise / timestep / aug sampling differs per rank — more
    # diverse training signal at the same wall-clock step. The +global_step_offset
    # term advances the RNG on resume so we don't replay the same noise/timestep
    # stream we already trained on.
    seed = 42 + rank * 10000 + args.global_step_offset
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ---------------- Dataset ----------------
    dataset = ChunkAwareDataset(
        csv_path=args.dataset_metadata_path,
        height=args.height,
        width=args.width,
        chunk_frames=args.num_frames,
        recent_aug_strength=args.recent_aug_strength,
        dataset_repeat=args.dataset_repeat,
    )
    # Resume position: map global_step_offset back to (epoch, batch-in-epoch)
    # so we CONTINUE the stream instead of replaying epoch 0 from the top.
    steps_per_epoch = (len(dataset) // num_proc) if num_proc > 1 else len(dataset)
    steps_per_epoch = max(steps_per_epoch, 1)
    start_epoch  = args.global_step_offset // steps_per_epoch
    skip_batches = args.global_step_offset %  steps_per_epoch

    # Base sampler: DistributedSampler (multi-GPU) or RandomSampler (single).
    if num_proc > 1:
        base_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=num_proc, rank=rank, shuffle=True, drop_last=True,
        )
    else:
        base_sampler = torch.utils.data.RandomSampler(dataset)
    # Wrap so the skipped batches are dropped at the index level (no decode).
    sampler = ResumableSampler(base_sampler, skip_first=skip_batches, skip_epoch=start_epoch)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        shuffle=False,            # sampler handles shuffling
        batch_size=None,
        num_workers=2,
    )

    # ---------------- Training log (main process only) ----------------
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
              f"{start_epoch}, skip {skip_batches}/{steps_per_epoch} batches "
              f"(index-level, no decode)")

    # ---------------- Training loop ----------------
    local_step = 0
    for epoch in range(start_epoch, start_epoch + args.num_epochs):
        sampler.set_epoch(epoch)   # reshuffle + apply skip on the start epoch only
        pbar = tqdm(dataloader, desc=f"epoch {epoch}", disable=not is_main)
        for batch in pbar:
            mem_debug_step = args.memory_debug and local_step < args.memory_debug_steps
            log_cuda_memory(f"step {local_step} start", [device, fd_device], mem_debug_step, is_main)
            # ─── encode all conditioning once ───
            cond = encode_batch(
                student_pipe, batch, dtype, device, no_recent=args.no_recent,
                vae_device=fd_device,
            )
            target_latents    = cond["target_latents"]
            control_latents   = cond["control_latents"]
            prompt_embed      = cond["prompt_embed"]
            # These heavy encoders are only needed to materialize conditioning
            # tensors. Keep VAE on GPU for the MMD/FD decode path, but move
            # T5/CLIP away before DiT + VAE-grad work.
            offload_pipe_modules_to_cpu(student_pipe, ["text_encoder", "image_encoder"], device)
            log_cuda_memory(f"step {local_step} after encode_batch", [device, fd_device], mem_debug_step, is_main)

            # Route BOTH ref latent and CLIP feature per augment mode:
            #   off        → clean for everyone
            #   symmetric  → augmented for everyone (teacher must be augment-robust)
            #   asymmetric → clean to teacher, augmented to student/critic
            if args.recent_augment_mode == "off":
                ref_student = ref_teacher = cond["reference_latents_clean"]
                clip_student = clip_teacher = cond["clip_feature_clean"]
            elif args.recent_augment_mode == "asymmetric":
                ref_student  = cond["reference_latents_aug"]
                ref_teacher  = cond["reference_latents_clean"]
                clip_student = cond["clip_feature_aug"]
                clip_teacher = cond["clip_feature_clean"]
            else:  # symmetric
                ref_student = ref_teacher = cond["reference_latents_aug"]
                clip_student = clip_teacher = cond["clip_feature_aug"]

            B = target_latents.shape[0]

            # ─── ROLLOUT student through the full schedule (random exit) ───
            # Replaces train_dmd.py's GT-noising + single forward. The student
            # sees its OWN intermediate states; only the exit step has grad.
            # x_pred is the exit-step x0 estimate (used for both critic update
            # and the DMD generator update below).
            # exit_idx comes from a step-seeded RNG shared across ranks (per-
            # rank seeds diverge, so random.randrange would differ per rank):
            # the FD first-step gate below must be collective-consistent or
            # the all_gather deadlocks. At num_inference_steps=1 always 0.
            exit_idx = random.Random(
                10007 * (args.global_step_offset + local_step + 1)
            ).randrange(len(denoising_step_list))
            x_pred, t_gen = rollout_student(
                student_pipe.dit, denoising_step_list, tuple(target_latents.shape),
                prompt_embed, control_latents, ref_student, clip_student,
                device, dtype,
                use_gradient_checkpointing=args.use_gradient_checkpointing,
                exit_idx=exit_idx,
            )
            log_cuda_memory(f"step {local_step} after student rollout", [device, fd_device], mem_debug_step, is_main)

            # ─── critic update × N (use x_pred.detach() — no grad to student) ───
            onload_pipe_modules_to_device(critic_pipe, ["dit"], device)
            for _ in range(args.dfake_gen_update_ratio):
                t_c = sample_critic_timestep(shift=args.flow_shift)
                sigma_c = timestep_to_sigma(t_c)
                noise_c = torch.randn_like(x_pred)
                x_pred_noisy_c, _ = add_noise_flow(x_pred.detach(), sigma_c, noise_c)

                v_critic_pred = dit_forward(
                    critic_pipe.dit, x_pred_noisy_c, t_c, prompt_embed,
                    control_latents, ref_student, clip_student, device, dtype,
                    use_gradient_checkpointing=args.use_gradient_checkpointing,
                )
                critic_loss = compute_critic_loss(v_critic_pred, x_pred, noise_c)
                critic_optimizer.zero_grad(set_to_none=True)
                critic_loss.backward()
                all_reduce_grads(critic_params, num_proc)   # sync across ranks
                critic_optimizer.step()
                critic_optimizer.zero_grad(set_to_none=True)
            log_cuda_memory(f"step {local_step} after critic update", [device, fd_device], mem_debug_step, is_main)

            # ─── generator update × 1 (reuse x_pred, with grad) ───
            t_dmd = sample_critic_timestep(shift=args.flow_shift)
            sigma_dmd = timestep_to_sigma(t_dmd)
            noise_dmd = torch.randn_like(x_pred)
            x_pred_noisy_dmd, _ = add_noise_flow(x_pred, sigma_dmd, noise_dmd)

            with torch.no_grad():
                # fake score (critic), conditional only — student-side recent
                v_fake = dit_forward(
                    critic_pipe.dit, x_pred_noisy_dmd, t_dmd, prompt_embed,
                    control_latents, ref_student, clip_student, device, dtype,
                )
                pred_fake = velocity_to_x0(v_fake, x_pred_noisy_dmd, sigma_dmd)
                offload_pipe_modules_to_cpu(critic_pipe, ["dit"], device)
                onload_pipe_modules_to_device(teacher_pipe, ["dit"], device)

                # real score (teacher), CFG-enhanced — teacher-side (clean) recent
                v_real_cond = dit_forward(
                    teacher_pipe.dit, x_pred_noisy_dmd, t_dmd, prompt_embed,
                    control_latents, ref_teacher, clip_teacher, device, dtype,
                )
                pred_real_cond = velocity_to_x0(v_real_cond, x_pred_noisy_dmd, sigma_dmd)
                if neg_prompt_embed is not None:
                    v_real_uncond = dit_forward(
                        teacher_pipe.dit, x_pred_noisy_dmd, t_dmd, neg_prompt_embed,
                        control_latents, ref_teacher, clip_teacher, device, dtype,
                    )
                    pred_real_uncond = velocity_to_x0(v_real_uncond, x_pred_noisy_dmd, sigma_dmd)
                else:
                    pred_real_uncond = None
                pred_real = cfg_real_x0(pred_real_cond, pred_real_uncond, args.real_guidance_scale)
                offload_pipe_modules_to_cpu(teacher_pipe, ["dit"], device)

            grad = compute_dmd_gradient(
                pred_fake, pred_real, x_pred, normalize=args.dmd_normalize_grad,
            )
            dmd_g_loss = compute_dmd_loss(x_pred, grad)
            log_cuda_memory(f"step {local_step} after DMD score", [device, fd_device], mem_debug_step, is_main)

            # ─── FD-loss v4: full-chunk causal decode → pixel-space frame
            # subsample → DINOv2 → Frechet vs REAL-data stats ───
            # Gate 1 (phase purity): only when the rollout exited at the FIRST
            #   denoising step — the pure 1-NFE output deployment uses. Loss
            #   AND enqueue both sit behind this gate, so the queue population
            #   (warm-up included) is single-phase by construction.
            # Gate 2 (warm-up): until the queue holds fd_min_pop features the
            #   FD term is skipped; features are extracted no_grad + enqueued
            #   only. Never backprop through a rank-deficient covariance.
            # Both gates are rank-consistent (shared exit_idx; queue count is
            # identical across ranks since gathered feats are enqueued
            # everywhere), so the all_gather below cannot deadlock.
            fd_active = (not args.fd_first_step_only) or exit_idx == 0
            fd_in_loss = False
            fd_g_loss = None
            fd_raw_val = float("nan")
            if fd_active:
                # Wan causal VAE: T_lat latents → 4·(T_lat−1)+1 pixel frames.
                # Decode the WHOLE chunk — partial or shuffled latent decodes
                # go through the causal VAE out of temporal context and yield
                # pixels of a *different* video (the v1-v3 domain-mismatch
                # bug). Random frame subsample happens AFTER decode, in pixel
                # space, matching the reference-stats protocol.
                T_lat = x_pred.shape[2]
                fd_w = args.fd_window_latents
                if 0 < fd_w < T_lat:
                    # Random causal window: the no_grad prefix primes the
                    # causal cache, so window pixels are IDENTICAL to the full
                    # decode while grad memory scales with the window. j=0 IS
                    # included — the chunk's first frames are the continuity
                    # anchor to the previous chunk and need FD gradient too.
                    # Latent 0 decodes to 1 frame instead of 4, so the j=0
                    # draw widens the window by one latent (4·fd_w+1 frames);
                    # n_fr is capped by the SMALLEST branch (4·fd_w), keeping
                    # per-rank feature counts — and the all_gather shape —
                    # uniform no matter which j each rank draws.
                    J = T_lat - fd_w
                    if args.fd_window_tail_bias > 0:
                        # P(j) ∝ 1 + bias·j/J — soft tilt toward the blurry
                        # chunk tail; j=0 (continuity anchor) stays covered.
                        wts = [1.0 + args.fd_window_tail_bias * (j / max(J, 1))
                               for j in range(J + 1)]
                        fd_j = random.choices(range(J + 1), weights=wts)[0]
                    else:
                        fd_j = random.randint(0, J)
                    w_eff = min(fd_w + 1, T_lat) if fd_j == 0 else fd_w
                    T_win = 4 * w_eff - (3 if fd_j == 0 else 0)
                    n_fr = min(args.fd_num_frames, 4 * fd_w)
                else:
                    fd_j, fd_w, w_eff = 0, 0, 0
                    T_win = 4 * (T_lat - 1) + 1
                    n_fr = min(args.fd_num_frames, T_win)
                fidx = torch.randperm(T_win, device="cpu")[:n_fr]

                # Two representation views:
                #   global: squash-resized whole frame (semantic/layout; blind
                #           to blur — the 3.7x resize low-passes it away)
                #   crop:   K random 224px crops at NATIVE resolution per
                #           frame (texture/blur-sensitive; blind to layout)
                # fd_weight gates global, fd_crop_weight gates crop. Offsets
                # sampled OUTSIDE the checkpoint → identical crops in the
                # backward recompute.
                use_global = args.fd_weight > 0 or args.fd_objective == "frechet"
                use_crop = crop_ref_feats is not None
                n_g = (B * n_fr) if use_global else 0
                K_crop = args.fd_crops_per_frame if use_crop else 0
                crop_off = None
                if use_crop:
                    crop_off = sample_crop_offsets(
                        B, n_fr, K_crop, args.height, args.width,
                        args.fd_crop_size,
                    )

                def _vae_to_feats(x_lat, frame_idx, c_off):
                    # [B,C,T_lat,H_lat,W_lat] -> [B,3,T_win,H,W] in [-1,1].
                    log_cuda_memory("FD before VAE decode", [device, fd_device], mem_debug_step, is_main)
                    x_lat = x_lat.to(device=fd_device, dtype=dtype)
                    if fd_w:
                        decoded = student_pipe.vae.decode_window(
                            x_lat, device=fd_device, t_start=fd_j, t_len=w_eff,
                        ).to(dtype)
                    else:
                        # Full-chunk grad decode (tiled): heavy fallback.
                        decoded = student_pipe.vae.decode(
                            x_lat, device=fd_device, tiled=args.fd_vae_tiled,
                            tile_size=tuple(args.fd_vae_tile_size),
                            tile_stride=tuple(args.fd_vae_tile_stride),
                        ).to(dtype)
                    log_cuda_memory("FD after VAE decode", [device, fd_device], mem_debug_step, is_main)
                    pix = ((decoded.clamp(-1, 1) + 1) / 2)      # [B,3,T_win,H,W]
                    pix = pix.permute(0, 2, 1, 3, 4)            # [B,T_win,3,H,W]
                    pix = pix.index_select(1, frame_idx.to(pix.device))
                    pix = pix.to(device=fd_device, dtype=dtype) # [B,n_fr,3,H,W]
                    # Per-rank layout: [global rows (B*n_fr); crop rows
                    # (B*n_fr*K)] — fixed sizes, so the gathered tensor can be
                    # split back per rank.
                    outs = []
                    if use_global:
                        outs.append(fd_extractor(pix))          # [B*n_fr, D]
                    if use_crop:
                        crops = crop_frames(pix, c_off.to(pix.device),
                                            args.fd_crop_size)  # [B,n_fr*K,3,c,c]
                        outs.append(fd_extractor(crops))        # [B*n_fr*K, D]
                    return torch.cat(outs, dim=0)

                fd_in_loss = (
                    args.fd_objective == "mmd"
                    or int(fd_queue.count.item()) >= args.fd_min_pop
                )
                if fd_in_loss:
                    # checkpoint: VAE/DINOv2 activations recomputed in backward
                    # (same memory rationale as v3, now on the full chunk).
                    new_feats_local = torch.utils.checkpoint.checkpoint(
                        _vae_to_feats, x_pred, fidx, crop_off, use_reentrant=False,
                    )
                else:
                    with torch.no_grad():
                        new_feats_local = _vae_to_feats(x_pred.detach(), fidx, crop_off)
                log_cuda_memory(f"step {local_step} after FD feature extract", [device, fd_device], mem_debug_step, is_main)
                # Pool features across ranks (differentiable: grad routes back
                # to the local portion only), then split the per-rank blocks
                # back into the two views.
                new_feats = diff_all_gather(new_feats_local)
                n_local = new_feats_local.shape[0]
                blocks = new_feats.reshape(-1, n_local, new_feats.shape[-1])
                g_feats = blocks[:, :n_g].reshape(-1, blocks.shape[-1]) if use_global else None
                c_feats = blocks[:, n_g:].reshape(-1, blocks.shape[-1]) if use_crop else None

            fd_c_loss = None
            fd_c_raw = float("nan")
            if fd_in_loss:
                fd_terms = dmd_g_loss.new_zeros(())
                if use_global:
                    if args.fd_objective == "frechet":
                        all_feats = torch.cat([fd_queue.get_valid(), g_feats], dim=0)
                        fd_g_loss, fd_raw_val = compute_fd_loss_normalized(
                            mu_ref, sigma_ref, all_feats, sigma_ref_sqrt=sigma_ref_sqrt,
                            eval_clamp_min=args.fd_eval_clamp,
                        )                                        # ≈ 1
                    else:
                        fd_g_loss, fd_raw_val = compute_mmd_loss_normalized(
                            mmd_ref_feats, g_feats, bandwidth=mmd_bandwidth,
                            ref_chunk=args.fd_mmd_ref_chunk,
                        )                                        # ≈ ±1
                    fd_terms = fd_terms + args.fd_weight * fd_g_loss.to(device)
                if use_crop:
                    fd_c_loss, fd_c_raw = compute_mmd_loss_normalized(
                        crop_ref_feats, c_feats, bandwidth=crop_bandwidth,
                        ref_chunk=args.fd_mmd_ref_chunk,
                    )                                            # ≈ ±1
                    fd_terms = fd_terms + args.fd_crop_weight * fd_c_loss.to(device)
                # diff_all_gather(scale_backward=True) already compensates for
                # the manual all_reduce(mean) below, so fd_weight is stable
                # across world sizes without an extra num_proc factor here.
                gen_loss = dmd_g_loss + fd_terms
            else:
                # FD gated off this step (warm-up, or non-first-step exit).
                gen_loss = dmd_g_loss

            student_optimizer.zero_grad(set_to_none=True)
            gen_loss.backward()
            all_reduce_grads(student_params, num_proc)   # sync across ranks
            student_optimizer.step()
            student_optimizer.zero_grad(set_to_none=True)
            log_cuda_memory(f"step {local_step} after generator backward/step", [device, fd_device], mem_debug_step, is_main)

            if fd_active and args.fd_objective == "frechet":
                # Thin the enqueue to fd_enqueue_per_chunk frames per chunk so
                # the queue spans more distinct videos at the same size. fidx
                # is a random permutation, so the first n_enq rows of each
                # chunk block are already a uniform random frame subset.
                enq = g_feats.detach()
                n_enq = args.fd_enqueue_per_chunk
                if 0 < n_enq < n_fr:
                    enq = enq.reshape(-1, n_fr, enq.shape[-1])[:, :n_enq]
                    enq = enq.reshape(-1, enq.shape[-1])
                fd_queue.enqueue(enq)

            local_step += 1
            global_step = args.global_step_offset + local_step
            # ─── EMA update on student LoRA ───
            if global_step >= args.ema_start_step:
                with torch.no_grad():
                    if not hasattr(main, "_ema_state"):
                        main._ema_state = {n: p.data.clone()
                                           for n, p in student_pipe.dit.named_parameters()
                                           if p.requires_grad}
                        if is_main:
                            print(f"[ema] initialized at step {global_step}")
                    else:
                        d = args.ema_decay
                        for n, p in student_pipe.dit.named_parameters():
                            if p.requires_grad and n in main._ema_state:
                                main._ema_state[n].mul_(d).add_(p.data, alpha=1 - d)
            if is_main:
                fd_str  = "-" if fd_g_loss is None else f"{fd_g_loss.item():.3f}"
                # .4f: MMD raw sits near -0.5 and moves in the 2nd-4th decimal;
                # .1f made it look frozen.
                fdr_str = "-" if fd_g_loss is None else f"{float(fd_raw_val):.4f}"
                fdc_str = "-" if fd_c_loss is None else f"{float(fd_c_raw):.4f}"
                pbar.set_postfix(
                    g=f"{gen_loss.item():.4f}", c=f"{critic_loss.item():.4f}",
                    fd=fd_str, fdR=fdr_str, fdC=fdc_str,
                    q=(int(fd_queue.count.item()) if fd_queue is not None else 0), t=t_gen,
                )
                log_file.write(
                    f"{time.strftime('%H:%M:%S')} epoch={epoch} step={global_step} "
                    f"t_gen={t_gen} gen_loss={gen_loss.item():.6f} "
                    f"dmd_g={dmd_g_loss.item():.6f} fd_norm={fd_str} fd_raw={fdr_str} "
                    f"fd_crop_raw={fdc_str} "
                    f"fd_active={int(fd_active)} "
                    f"q={(int(fd_queue.count.item()) if fd_queue is not None else 0)} "
                    f"critic_loss={critic_loss.item():.6f}\n"
                )

            # ─── save (main process only — params are sync'd across ranks anyway) ───
            if global_step % args.save_steps == 0:
                accelerator.wait_for_everyone()
                if is_main:
                    student_path = os.path.join(args.output_path, f"step-{global_step}.safetensors")
                    critic_path  = os.path.join(args.output_path, f"step-{global_step}_critic.safetensors")
                    state_path   = os.path.join(args.output_path, f"step-{global_step}_state.pt")
                    save_lora_state_dict(student_pipe.dit, student_path, remove_prefix="dit.")
                    save_lora_state_dict(critic_pipe.dit,  critic_path,  remove_prefix="dit.")
                    # EMA LoRA (preferred for inference)
                    if hasattr(main, "_ema_state"):
                        ema_path = os.path.join(args.output_path, f"step-{global_step}_ema.safetensors")
                        ema_sd = {n: v.detach().cpu().contiguous()
                                  for n, v in main._ema_state.items()
                                  if "lora_A" in n or "lora_B" in n}
                        save_file(ema_sd, ema_path)
                    else:
                        ema_path = "(not yet started)"
                    # Optimizer moments + step counter for seamless resume.
                    torch.save({
                        "student_optimizer": student_optimizer.state_dict(),
                        "critic_optimizer":  critic_optimizer.state_dict(),
                        "global_step": global_step,
                    }, state_path)
                    print(f"[save] {student_path} + ema:{ema_path} + {critic_path} + {state_path}")

    if is_main:
        log_file.write(f"=== run end {time.strftime('%Y-%m-%d %H:%M:%S')} | "
                       f"total_steps={args.global_step_offset + local_step} ===\n")
        log_file.close()


if __name__ == "__main__":
    main()
