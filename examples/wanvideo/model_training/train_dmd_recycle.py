"""
DMD2 + ONE-FORCING (cls_branch GAN) + SVI-STYLE SELF-CORRECTING ERROR RECYCLE.

Keeps DMD + cls_branch GAN intact, adds SVI's core mechanism:

  Per step:
    1. CLEAN rollout (no grad):
         x_pred_clean = student(noise, ref_clean)         ← no error injected
    2. INJECT error from buffer:
         ref_corrupt = ref_clean + α · sampled_error
    3. CORRUPT rollout (with grad):
         x_pred      = student(noise, ref_corrupt)
    4. Critic + GAN-D updates on x_pred.detach()           (unchanged)
    5. Generator update:
         dmd_g_loss + gan_g_loss + λ_sc · L1(x_pred, x_pred_clean.detach())
                                          ↑↑↑ SELF-CORRECTING LOSS
         student must produce same output regardless of error in recent_ref
    6. COLLECT error (drift direction):
         err = x_pred_clean[last] - target_latents[last]   ← (student - real)
         buffer.push(err)
       Sign matters: this is the direction student's autoregressive recent_ref
       drifts away from real at inference. Injecting this direction during
       training simulates what student will see at inference.

Key difference vs the naive recycle (replay only): the self-correcting L1
trains student to be INVARIANT to recent_ref perturbation. This is the
mechanism that makes SVI work — not just replaying errors, but training
the network to output the same answer under input corruption.

α schedule: 0 during buffer warmup, then linearly ramps to error_alpha.

Reference: SVI (Stable-Video-Infinity, vita-epfl). Adapted from base-model
finetuning to DMD distillation regime by anchoring the self-correcting
target to the clean-input rollout.
"""
import argparse, os, random, sys, time
import accelerate
import decord
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.distributed as dist
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
from cls_branch import ClsBranch, FeatureCapturer, gan_g_loss, gan_d_loss


# ===========================================================================
# Error replay buffer (FIFO ring buffer of latent residuals)
# ===========================================================================
class ErrorBuffer:
    """Per-rank FIFO buffer of single-latent-frame residuals [C, h, w]."""
    def __init__(self, max_size: int = 500):
        self.max_size = max_size
        self.buf = []     # list of CPU bf16 tensors

    def push(self, err: torch.Tensor):
        self.buf.append(err.detach().to(torch.bfloat16).cpu().contiguous())
        if len(self.buf) > self.max_size:
            self.buf.pop(0)

    def sample(self, device=None, dtype=None) -> torch.Tensor:
        """Returns a single error of shape [C, h, w]. Raises if empty."""
        assert len(self.buf) > 0, "ErrorBuffer is empty"
        idx = torch.randint(0, len(self.buf), (1,)).item()
        out = self.buf[idx]
        if device is not None: out = out.to(device)
        if dtype is not None:  out = out.to(dtype)
        return out

    def __len__(self):
        return len(self.buf)


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
    ids, mask = pipe.tokenizer(prompt, return_mask=True, add_special_tokens=True)
    ids = ids.to(device)
    mask = mask.to(device)
    seq_lens = mask.gt(0).sum(dim=1).long()
    prompt_emb = pipe.text_encoder(ids, mask).to(dtype=dtype)
    for i, v in enumerate(seq_lens):
        prompt_emb[:, v:] = 0
    return prompt_emb


@torch.no_grad()
def encode_batch(pipe, batch, dtype, device, no_recent=False):
    """Use a pipe's encoders (T5/VAE/CLIP) to encode one batch from
    ChunkAwareDataset into the tensors model_fn_wan_video expects.

    Returns dict with: prompt_embed, target_latents, control_latents,
                       reference_latents (sink+recent on T-dim), clip_feature
    """
    # ---- T5 text ----
    prompt_emb = encode_prompt(pipe, batch["prompt"], dtype, device)

    # ---- VAE: target video → latent ----
    pipe.vae.to(device)
    target_video = pipe.preprocess_video(batch["video"]).to(dtype=dtype, device=device)
    target_latents = pipe.vae.encode(target_video, device=device).to(dtype=dtype)

    # ---- VAE: control video (pose) → latent ----
    # Wan-Fun-Control expects y to have (dit.in_dim - 2 * latent_channels) channels:
    # y = [control_latent(16) | zero_padding(16)] for V1.1 1.3B (in_dim=48).
    # See WanVideoUnit_FunControl.process (wan_video.py:540).
    control_video = pipe.preprocess_video(batch["control_video"]).to(dtype=dtype, device=device)
    control_latents = pipe.vae.encode(control_video, device=device).to(dtype=dtype)
    y_dim = pipe.dit.in_dim - control_latents.shape[1] - target_latents.shape[1]
    if y_dim > 0:
        y_pad = torch.zeros(
            (control_latents.shape[0], y_dim, control_latents.shape[2],
             control_latents.shape[3], control_latents.shape[4]),
            dtype=dtype, device=device,
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
        sink_video = pipe.preprocess_video([sink_pil.resize((w, h))]).to(dtype=dtype, device=device)
        sink_latent = pipe.vae.encode(sink_video, device=device).to(dtype=dtype)
        if no_recent:
            return sink_latent           # sink-only (1 frame) for sinkonly teacher
        recent_video = pipe.preprocess_video([recent_pil.resize((w, h))]).to(dtype=dtype, device=device)
        recent_latent = pipe.vae.encode(recent_video, device=device).to(dtype=dtype)
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
        pipe.image_encoder.to(device)
        def _clip(pil):
            return pipe.image_encoder.encode_image([pipe.preprocess_image(pil)]).to(dtype=dtype)
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
    device, dtype, use_gradient_checkpointing=False,
    initial_noisy=None, exit_idx=None, renoise_noises=None,
):
    """Roll the student through the full few-step denoising schedule, starting
    from PURE NOISE. Random exit step → only one forward keeps grad; everything
    earlier runs under no_grad with re-noising between steps.

    Args (new, for SVI self-correcting loss — shared randomness across two rollouts):
        initial_noisy : pre-generated noise tensor [target_shape]. If None, a new
                        one is sampled. Pass the same tensor to both clean+corrupt
                        rollouts so the only difference is the reference.
        exit_idx      : int in [0, N), the step at which to take the grad forward.
                        If None, sampled uniformly. Pass same value to both rollouts.
        renoise_noises: list of fresh noise tensors used for re-noising between
                        no-grad steps. Length = exit_idx. If None, sampled fresh.
    """
    N = len(denoising_step_list)
    if exit_idx is None:
        exit_idx = random.randrange(N)        # uniform over 0..N-1
    if initial_noisy is None:
        initial_noisy = torch.randn(target_shape, dtype=dtype, device=device)
    noisy = initial_noisy
    renoise_idx = 0

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
            if renoise_noises is not None and renoise_idx < len(renoise_noises):
                fresh = renoise_noises[renoise_idx]
            else:
                fresh = torch.randn_like(x0)
            renoise_idx += 1
            # Re-noise the clean estimate to the next timestep (consistency-
            # sampler style, matches CF++ scheduler.add_noise).
            # NOTE: This is a LINEAR INTERPOLATION re-noise. The actual inference
            # pipeline uses the scheduler's Euler step, not this. For 1-step
            # rollout this branch is dead (exit_idx is always the only step).
            # If/when extending to 2/4-step rollouts, this train-vs-inference
            # mismatch becomes real — consider switching to scheduler.add_noise.
            noisy = sigma_next * fresh + (1 - sigma_next) * x0

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
    # ─── GAN (One-Forcing) ───
    p.add_argument("--gan_g_weight", type=float, default=0.03,
                   help="Weight of gan_g_loss added to generator's total loss.")
    p.add_argument("--gan_d_weight", type=float, default=0.03,
                   help="Weight on the discriminator (cls_branch) update.")
    p.add_argument("--gan_feature_layers", default="13,21,29",
                   help="WanModel block indices whose outputs feed the cls_branch.")
    p.add_argument("--gan_ffn_dim", type=int, default=4096,
                   help="FFN hidden dim inside each GanCrossAttnBlock (One-Forcing uses 8192).")
    p.add_argument("--gan_num_heads", type=int, default=12)
    # ─── SVI-style error recycle + self-correcting loss ───
    p.add_argument("--error_alpha", type=float, default=0.5,
                   help="Final scale on injected error (after warmup ramp). 0=off (sc loss still active).")
    p.add_argument("--error_buffer_size", type=int, default=500,
                   help="Max entries in error replay buffer (per rank, FIFO). SVI uses 500.")
    p.add_argument("--error_warmup_count", type=int, default=50,
                   help="Wait until buffer has >= this many entries before injecting (and ramping α).")
    p.add_argument("--error_alpha_ramp_steps", type=int, default=200,
                   help="Linearly ramp α from 0 to --error_alpha over this many steps after warmup. "
                        "0 = apply --error_alpha immediately.")
    p.add_argument("--sc_weight", type=float, default=0.5,
                   help="Weight on self-correcting L1 loss between corrupted x_pred and clean x_pred.")
    p.add_argument("--error_inject_prob", type=float, default=0.8,
                   help="Probability of injecting error on each step (after warmup). "
                        "1.0 = always inject, 0.8 = 80% inject + 20% clean (SVI-style). "
                        "When skipped, alpha_eff=0 and student sees clean ref.")
    p.add_argument("--error_collect_start_step", type=int, default=0,
                   help="Global step before which error collection is skipped. "
                        "Use 100-300 if you want DMD to stabilize before collecting "
                        "(early residuals may include untrained-LoRA noise, not just drift).")
    # ─── EMA (One-Forcing default) ───
    p.add_argument("--ema_decay", type=float, default=0.99)
    p.add_argument("--ema_start_step", type=int, default=200,
                   help="Start EMA tracking after this many global steps.")
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

    if is_main: print("[setup] building student (trainable LoRA) ...")
    student_pipe = build_pipe(device, dtype)
    if use_sink:
        fuse_sink_lora_into_pipe(student_pipe, args.teacher_lora_path)
    student_pipe.dit = add_trainable_lora(student_pipe.dit, target_modules, args.lora_rank)
    student_pipe.dit.train()   # gradient_checkpoint_forward checks self.training
    if args.resume_student_from:
        load_lora_into_trainable(student_pipe.dit, load_state_dict(args.resume_student_from))
        if is_main: print(f"[resume] student LoRA from {args.resume_student_from}")

    if is_main: print("[setup] building critic (trainable LoRA) ...")
    critic_pipe = build_pipe(device, dtype)
    if use_sink:
        fuse_sink_lora_into_pipe(critic_pipe, args.teacher_lora_path)
    critic_pipe.dit = add_trainable_lora(critic_pipe.dit, target_modules, args.lora_rank)
    critic_pipe.dit.train()
    if args.resume_critic_from:
        load_lora_into_trainable(critic_pipe.dit, load_state_dict(args.resume_critic_from))
        if is_main: print(f"[resume] critic LoRA from {args.resume_critic_from}")

    # ---------------- ClsBranch discriminator (hangs off the critic) ----------------
    gan_layers = [int(x) for x in args.gan_feature_layers.split(",")]
    dit_dim = critic_pipe.dit.dim
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
        print(f"[setup] cls_branch: {n_cls/1e6:.1f}M params, layers={gan_layers}")

    # ---------------- Optimizers ----------------
    student_params = trainable_params(student_pipe.dit)
    critic_params  = trainable_params(critic_pipe.dit)
    cls_params     = list(cls_branch.parameters())
    if is_main:
        print(f"[setup] student trainable params: {sum(p.numel() for p in student_params)/1e6:.1f}M")
        print(f"[setup] critic  trainable params: {sum(p.numel() for p in critic_params )/1e6:.1f}M")

    student_optimizer = torch.optim.AdamW(student_params, lr=args.learning_rate_student, weight_decay=1e-2)
    # critic_optimizer drives BOTH the denoising critic AND the cls_branch — they
    # are updated together by the GAN-D step and by critic_loss.
    critic_optimizer  = torch.optim.AdamW(critic_params + cls_params,
                                          lr=args.learning_rate_critic, weight_decay=1e-2)

    # ---------------- Resume optimizer state (AdamW moments) ----------------
    # The state sidecar sits next to the student LoRA: step-N.safetensors →
    # step-N_state.pt. Restoring it avoids the cold-start momentum re-warm.
    if args.resume_student_from:
        # cls_branch state (also next to student LoRA: ..._cls.pt)
        cls_path = args.resume_student_from.replace(".safetensors", "_cls.pt")
        if os.path.isfile(cls_path):
            cls_branch.load_state_dict(torch.load(cls_path, map_location=device))
            if is_main: print(f"[resume] cls_branch state from {cls_path}")
        elif is_main:
            print(f"[resume] cls_branch NOT loaded (no {cls_path}) — fresh init.")

        # EMA state (saved as LoRA-format safetensors)
        ema_resume_path = args.resume_student_from.replace(".safetensors", "_ema.safetensors")
        if os.path.isfile(ema_resume_path):
            from diffsynth.core import load_state_dict as _lsd
            ema_sd = _lsd(ema_resume_path)
            # Match against current student LoRA param names
            main._ema_state = {}
            for n, p in student_pipe.dit.named_parameters():
                if p.requires_grad and n in ema_sd:
                    main._ema_state[n] = ema_sd[n].to(device=p.device, dtype=p.dtype)
            if is_main:
                print(f"[resume] EMA state from {ema_resume_path} ({len(main._ema_state)} tensors)")
        elif is_main:
            print(f"[resume] EMA NOT loaded (no {ema_resume_path}) — will init when started.")

        # Optimizer states (+ SVI error buffer state)
        state_path = args.resume_student_from.replace(".safetensors", "_state.pt")
        if os.path.isfile(state_path):
            ckpt = torch.load(state_path, map_location=device)
            student_optimizer.load_state_dict(ckpt["student_optimizer"])
            try:
                critic_optimizer.load_state_dict(ckpt["critic_optimizer"])
            except (ValueError, KeyError) as e:
                if is_main:
                    print(f"[resume] critic_optimizer state mismatch ({e}) — cold start critic moments.")
            if is_main:
                print(f"[resume] student optimizer state from {state_path} "
                      f"(saved at step {ckpt.get('global_step', '?')})")
            # SVI buffer state (may not exist in old checkpoints — falls back to empty)
            resume_buffer = ckpt.get("error_buffer", None)
            resume_warmup_done_step = ckpt.get("error_warmup_done_step", None)
            if is_main and resume_buffer is not None:
                print(f"[resume] error buffer with {len(resume_buffer)} entries, "
                      f"warmup_done_step={resume_warmup_done_step}")
        else:
            resume_buffer = None
            resume_warmup_done_step = None
            if is_main:
                print(f"[resume] WARNING: no optimizer state at {state_path} — "
                      f"AdamW moments + error buffer start cold.")
    else:
        resume_buffer = None
        resume_warmup_done_step = None

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

    # ---------------- SVI Error Buffer + state ----------------
    error_buffer = ErrorBuffer(max_size=args.error_buffer_size)
    error_warmup_done_step = None
    # Restore from resume checkpoint if available
    if resume_buffer is not None:
        # torch.load(..., map_location=device) puts them on GPU; force back to CPU bf16
        # to match the buffer's runtime invariant.
        error_buffer.buf = [
            t.detach().to(torch.bfloat16).cpu().contiguous()
            for t in list(resume_buffer)[-args.error_buffer_size:]
        ]
    if resume_warmup_done_step is not None:
        error_warmup_done_step = resume_warmup_done_step
    if is_main:
        print(f"[setup] error recycle: alpha={args.error_alpha}, buffer={args.error_buffer_size}, "
              f"warmup_count={args.error_warmup_count}, ramp_steps={args.error_alpha_ramp_steps}, "
              f"sc_weight={args.sc_weight}")
        print(f"[setup] error_buffer initial size: {len(error_buffer)}, "
              f"warmup_done_step: {error_warmup_done_step}")

    # ---------------- Training loop ----------------
    local_step = 0
    for epoch in range(start_epoch, start_epoch + args.num_epochs):
        sampler.set_epoch(epoch)   # reshuffle + apply skip on the start epoch only
        pbar = tqdm(dataloader, desc=f"epoch {epoch}", disable=not is_main)
        for batch in pbar:
            # ─── encode all conditioning once ───
            cond = encode_batch(student_pipe, batch, dtype, device, no_recent=args.no_recent)
            target_latents    = cond["target_latents"]
            control_latents   = cond["control_latents"]
            prompt_embed      = cond["prompt_embed"]

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

            # ─── SVI: prepare clean/corrupted reference latents for two rollouts ───
            # Clean reference (no error) is the target for self-correcting loss.
            # Corrupted reference (clean + α·sampled_error in recent slice) is what
            # student actually sees during the gradient-bearing rollout.
            ref_student_clean = ref_student      # baseline (may already have random aug)
            alpha_eff = 0.0
            # Use GLOBAL step for ramp calc so resume picks up where it left off.
            # (local_step resets to 0 on resume; global_step = offset + local_step.)
            cur_global_step = args.global_step_offset + local_step
            # Probabilistic injection: with prob `error_inject_prob`, do error
            # injection; otherwise use clean ref (SVI's clean_prob mechanism).
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
                # SVI injection targets the "recent" slice (T=1) of reference_latents.
                # Under --no_recent, the reference is sink-only (T=1) and there is
                # no recent slice → skip injection entirely.
                if args.no_recent or ref_student.shape[2] < 2:
                    if local_step == 0 and is_main:
                        print("[error-recycle] --no_recent or single-frame ref detected; "
                              "skipping error injection.")
                    ref_student_corrupt = ref_student
                    alpha_eff = 0.0
                else:
                    err = error_buffer.sample(device=ref_student.device, dtype=ref_student.dtype)
                    ref_student_corrupt = ref_student.clone()
                    ref_student_corrupt[:, :, 1, :, :] = ref_student_corrupt[:, :, 1, :, :] + alpha_eff * err
            else:
                ref_student_corrupt = ref_student

            # ─── Shared randomness for two rollouts (clean vs corrupt) ─────
            # The two rollouts MUST differ only by reference_latents. Otherwise
            # sc_loss = L1(corrupt, clean) ends up regressing two different
            # random noise inputs to the same output, which pressures student
            # toward noise-invariance (= mode collapse).
            sc_loss_active = (alpha_eff > 0.0 and args.sc_weight > 0.0)
            N_steps = len(denoising_step_list)
            shared_exit_idx     = random.randrange(N_steps)
            shared_initial_noisy = torch.randn(target_latents.shape, dtype=dtype, device=device)
            shared_renoise_noises = [
                torch.randn(target_latents.shape, dtype=dtype, device=device)
                for _ in range(shared_exit_idx)        # only need this many fresh re-noises
            ]

            # ─── CLEAN ROLLOUT (no grad) — target for self-correcting loss ───
            if sc_loss_active:
                with torch.no_grad():
                    x_pred_clean_target, _ = rollout_student(
                        student_pipe.dit, denoising_step_list, tuple(target_latents.shape),
                        prompt_embed, control_latents, ref_student_clean, clip_student,
                        device, dtype,
                        use_gradient_checkpointing=args.use_gradient_checkpointing,
                        initial_noisy=shared_initial_noisy,
                        exit_idx=shared_exit_idx,
                        renoise_noises=shared_renoise_noises,
                    )
                    x_pred_clean_target = x_pred_clean_target.detach()
            else:
                x_pred_clean_target = None

            # ─── CORRUPT ROLLOUT (with grad) — main path ───
            # Uses the SAME initial noise + exit_idx as the clean rollout above.
            x_pred, t_gen = rollout_student(
                student_pipe.dit, denoising_step_list, tuple(target_latents.shape),
                prompt_embed, control_latents, ref_student_corrupt, clip_student,
                device, dtype,
                use_gradient_checkpointing=args.use_gradient_checkpointing,
                initial_noisy=shared_initial_noisy,
                exit_idx=shared_exit_idx,
                renoise_noises=shared_renoise_noises,
            )

            # ─── critic update × N (use x_pred.detach() — no grad to student) ───
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
                critic_optimizer.zero_grad()
                critic_loss.backward()
                all_reduce_grads(critic_params + cls_params, num_proc)   # sync
                critic_optimizer.step()

            # ─── GAN-D update × N (One-Forcing's discriminator step) ───
            # Train critic + cls_branch to distinguish noised fake vs noised real.
            # MEMORY: stack fake+real into ONE forward (2× batch instead of 2 sequential
            # forwards keeping both graphs alive), and CPU-offload activations.
            for _ in range(args.dfake_gen_update_ratio):
                t_d = sample_critic_timestep(shift=args.flow_shift)
                sigma_d = timestep_to_sigma(t_d)
                noise_d = torch.randn_like(x_pred)
                fake_noisy_d, _ = add_noise_flow(x_pred.detach(), sigma_d, noise_d)
                real_noisy_d, _ = add_noise_flow(target_latents,   sigma_d, noise_d)

                stacked = torch.cat([fake_noisy_d, real_noisy_d], dim=0)
                # T5/CLIP/control conditioning: also stack 2× along batch dim
                prompt_2 = torch.cat([prompt_embed, prompt_embed], dim=0)
                control_2 = torch.cat([control_latents, control_latents], dim=0)
                ref_2 = torch.cat([ref_student, ref_student], dim=0)
                clip_2 = torch.cat([clip_student, clip_student], dim=0) if clip_student is not None else None

                with torch.autograd.graph.save_on_cpu(pin_memory=False):
                    with FeatureCapturer(critic_pipe.dit, gan_layers) as cap:
                        _ = dit_forward(
                            critic_pipe.dit, stacked, t_d, prompt_2,
                            control_2, ref_2, clip_2, device, dtype,
                            use_gradient_checkpointing=args.use_gradient_checkpointing,
                        )
                        stacked_feats = cap.features()
                    stacked_logit = cls_branch(stacked_feats)
                fake_logit_d, real_logit_d = stacked_logit.chunk(2, dim=0)

                d_loss = gan_d_loss(real_logit_d, fake_logit_d) * args.gan_d_weight
                critic_optimizer.zero_grad()
                d_loss.backward()
                all_reduce_grads(critic_params + cls_params, num_proc)
                critic_optimizer.step()

            # ─── generator update × 1 (DMD + GAN-G, with grad through x_pred) ───
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

            grad = compute_dmd_gradient(
                pred_fake, pred_real, x_pred, normalize=args.dmd_normalize_grad,
            )
            dmd_g_loss = compute_dmd_loss(x_pred, grad)

            # GAN-G: push x_pred toward "real" via the (frozen-this-step)
            # critic+cls_branch discriminator. Critic/cls grads from this loss
            # are computed but discarded — they get zeroed before the next
            # critic_loss step (`critic_optimizer.zero_grad()`).
            # MEMORY: CPU-offload activations (One-Forcing's gan_activation_cpu_offload).
            t_g = sample_critic_timestep(shift=args.flow_shift)
            sigma_g = timestep_to_sigma(t_g)
            noise_g = torch.randn_like(x_pred)
            fake_noisy_g, _ = add_noise_flow(x_pred, sigma_g, noise_g)
            with torch.autograd.graph.save_on_cpu(pin_memory=False):
                with FeatureCapturer(critic_pipe.dit, gan_layers) as cap:
                    _ = dit_forward(
                        critic_pipe.dit, fake_noisy_g, t_g, prompt_embed,
                        control_latents, ref_student, clip_student, device, dtype,
                        use_gradient_checkpointing=args.use_gradient_checkpointing,
                    )
                    fake_feats_g = cap.features()
                fake_logit_g = cls_branch(fake_feats_g)
            g_loss = gan_g_loss(fake_logit_g) * args.gan_g_weight

            # ─── SVI Self-correcting loss ───
            # L1 between corrupted x_pred (with grad) and clean x_pred (detached).
            # Trains student to be invariant to error in recent_ref.
            if x_pred_clean_target is not None:
                sc_loss = F.l1_loss(x_pred, x_pred_clean_target)
            else:
                sc_loss = x_pred.new_zeros(())

            gen_loss = dmd_g_loss + g_loss + args.sc_weight * sc_loss
            student_optimizer.zero_grad()
            gen_loss.backward()
            all_reduce_grads(student_params, num_proc)   # sync across ranks
            student_optimizer.step()

            # ─── SVI COLLECT: push student's drift direction to buffer ───
            # error = student - real  (drift direction; matches SVI sign)
            # Skipped until cur_global_step >= error_collect_start_step to avoid
            # contaminating buffer with cold-start undertrained-LoRA noise.
            #
            # CAVEAT (latent space mismatch):
            #   We collect from target_latents[:, :, -1] which is a slice of
            #   the 13-frame VIDEO latent (Wan VAE's temporal convs see context).
            #   We inject into reference_latents[:, :, 1] which is a SINGLE-FRAME
            #   image VAE encode (no temporal context). The two latents have
            #   similar but not identical distributions.
            #   A strictly correct collect would be:
            #     student_last_pix = vae.decode(x_pred[:, :, -1:])    # 1 frame
            #     student_enc      = vae.encode(student_last_pix)     # single-frame latent
            #     err              = student_enc - real_single_frame_enc
            #   That adds 1 VAE decode + 1 VAE encode per step (~20% overhead).
            #   First-pass uses video-latent-slice as a proxy — direction should be
            #   approximately right; if err_norm looks wrong or sc_loss collapses,
            #   switching to the strict version is the first thing to try.
            err_norm = 0.0
            if cur_global_step >= args.error_collect_start_step:
                with torch.no_grad():
                    ref_x_pred = x_pred_clean_target if x_pred_clean_target is not None else x_pred
                    real_last    = target_latents[:, :, -1, :, :]
                    student_last = ref_x_pred[:,    :, -1, :, :].detach()
                    err = (student_last - real_last)
                    err_norm = float(err.abs().mean())          # for logging
                    for b in range(err.shape[0]):
                        error_buffer.push(err[b])      # [C, h, w]

            # ─── EMA update (One-Forcing: ema_weight=0.99, start after step 200) ───
            local_step += 1
            global_step = args.global_step_offset + local_step
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

            # ─── save (main process only — params are sync'd across ranks anyway) ───
            if global_step % args.save_steps == 0:
                accelerator.wait_for_everyone()
                if is_main:
                    student_path = os.path.join(args.output_path, f"step-{global_step}.safetensors")
                    critic_path  = os.path.join(args.output_path, f"step-{global_step}_critic.safetensors")
                    state_path   = os.path.join(args.output_path, f"step-{global_step}_state.pt")
                    save_lora_state_dict(student_pipe.dit, student_path, remove_prefix="dit.")
                    save_lora_state_dict(critic_pipe.dit,  critic_path,  remove_prefix="dit.")
                    cls_path     = os.path.join(args.output_path, f"step-{global_step}_cls.pt")
                    torch.save(cls_branch.state_dict(), cls_path)
                    # EMA LoRA — this is typically what you want for inference
                    if hasattr(main, "_ema_state"):
                        ema_path = os.path.join(args.output_path, f"step-{global_step}_ema.safetensors")
                        ema_sd = {}
                        for n, v in main._ema_state.items():
                            if "lora_A" in n or "lora_B" in n:
                                ema_sd[n] = v.detach().cpu().contiguous()
                        save_file(ema_sd, ema_path)
                    else:
                        ema_path = "(not yet started)"
                    # Optimizer moments + step counter + SVI buffer for seamless resume.
                    torch.save({
                        "student_optimizer": student_optimizer.state_dict(),
                        "critic_optimizer":  critic_optimizer.state_dict(),
                        "global_step": global_step,
                        # SVI error recycle state (per-rank buffer; we save rank-0's only)
                        "error_buffer": list(error_buffer.buf),     # list of CPU bf16 [C,h,w]
                        "error_warmup_done_step": error_warmup_done_step,
                    }, state_path)
                    print(f"[save] {student_path} + ema:{ema_path} + {cls_path} + {state_path} "
                          f"(buf={len(error_buffer)})")

    if is_main:
        log_file.write(f"=== run end {time.strftime('%Y-%m-%d %H:%M:%S')} | "
                       f"total_steps={args.global_step_offset + local_step} ===\n")
        log_file.close()


if __name__ == "__main__":
    main()
