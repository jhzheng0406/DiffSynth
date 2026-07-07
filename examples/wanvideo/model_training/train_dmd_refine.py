"""
DMD2 + SELF-FORCING ROLLOUT + GAN + TEACHER-REFINEMENT DISTILLATION.

Builds on train_dmd_oneforcing.py (DMD + One-Forcing cls_branch GAN) and
adds a TRAINING-TIME-ONLY teacher refinement signal that targets 1-step
clarity (裙边 / hands / fine detail) WITHOUT changing inference.

  Teacher trajectory-anchor loss (the new bit — Progressive Distillation style):
    1. draw init_noise; student 1-step rollout from it → x_pred (WITH grad)
    2. teacher runs K-step deterministic ODE from the SAME init_noise → x_teacher
       (NO grad). Same start + same pose/ref ⇒ structurally aligned with x_pred,
       but the K-step integration resolves the high-frequency detail the single
       Euler step skips (the ~20% sharpness gap measured: teacher 50-step > student).
    3. anchor_loss = L1(x_pred, x_teacher)         (or high-freq residual only)
    4. gen_loss += anchor_weight * anchor_loss     ← grad flows to student

  Why this beats the earlier renoise-refine: renoising the student's already-
  committed (blurry) x0 to low t and denoising 1-2 steps can't recover detail
  it never produced — the teacher trusts the blurry input at low t and only
  regenerates (changing structure) at high t. Anchoring on the teacher's OWN
  full trajectory from shared noise gives a sharp, aligned target instead.

  Inference is UNCHANGED and stays 1-NFE: the detail is distilled into the
  student LoRA weights; the teacher trajectory is a training-time target only.

  Knobs that matter most:
    --anchor_weight     auxiliary weight (keep small; DMD stays the main loss).
    --anchor_steps      teacher ODE steps (more = sharper target, K× forwards).
    --anchor_highfreq   match only x - blur(x) so DMD owns structure, anchor
                        transfers detail (avoids L1 fighting DMD over low-freq).
    --anchor_cfg        CFG on the teacher target (sharper, 2× forwards).

Everything else (DMD, cls_branch GAN) is identical to train_dmd_oneforcing.py.

Run via lora/wan1.3b_dmd_refine_v1.sh
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
    initial_noise=None,
):
    """Roll the student through the full few-step denoising schedule, starting
    from PURE NOISE. Random exit step → only one forward keeps grad; everything
    earlier runs under no_grad with re-noising between steps. The exit step's
    x0 estimate is returned as x_pred (the model's "final" output at that step).

    Returns: (x_pred, exit_t)  where exit_t is the timestep at the exit step.

    initial_noise: if given, used as the starting latent at denoising_step_list[0]
        instead of a fresh randn. Pass the SAME tensor to teacher_trajectory() so
        the teacher's multi-step output is structurally aligned with x_pred (both
        integrate from the identical starting point) — required for the
        trajectory-anchor L1 to transfer detail without fighting over structure.

    Mirrors pipeline/bidirectional_training.py:inference_with_trajectory in the
    upstream Causal-Forcing repo, simplified for our chunk-based setting.
    """
    N = len(denoising_step_list)
    exit_idx = random.randrange(N)        # uniform over 0..N-1

    # Start from pure noise — KEY DIFFERENCE from current train_dmd.py which
    # noises GT. The whole rollout is data-free for the target latent.
    noisy = initial_noise if initial_noise is not None else \
        torch.randn(target_shape, dtype=dtype, device=device)

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
# Teacher trajectory anchor — multi-step teacher from the SAME initial noise
# ===========================================================================
def teacher_trajectory(
    teacher_dit, initial_noise, teacher_step_list,
    prompt_embed, control_latents, ref_teacher, clip_teacher,
    device, dtype, neg_prompt_embed=None, cfg_scale=0.0,
):
    """Run the teacher's MULTI-STEP deterministic (flow-matching Euler) ODE from
    `initial_noise` and return its clean x0. Used as a regression target for the
    1-step student that started from the SAME initial_noise.

    Why this beats renoise-refine: the 1-step student commits structure in one
    jump; the K-step teacher integrates the SAME ODE finely. From the same start,
    the two endpoints share structure but the teacher resolves the high-frequency
    detail the single Euler step skips. L1(student, teacher) transfers exactly
    that detail — the ~20% sharpness gap we measured (teacher 50-step > student).

    This is Progressive/Trajectory Distillation (Salimans & Ho 2022) adapted as
    an auxiliary signal on top of DMD.

    Deterministic Euler step for flow matching: with v = noise - x0 and
    x_t = (1-σ)x0 + σ·noise, stepping σ→σ' gives x' = x_t + (σ'-σ)·v.

    teacher_step_list: DESCENDING timesteps; [0] MUST equal the student's single
        step so both interpret initial_noise as the latent at the same σ.
    cfg_scale > 0: CFG-enhance the teacher (sharper, but 2× forwards/step).
    """
    with torch.no_grad():
        x = initial_noise
        K = len(teacher_step_list)
        for idx, t in enumerate(teacher_step_list):
            sigma = float(t) / NUM_TRAIN_TIMESTEPS
            v = dit_forward(teacher_dit, x, t, prompt_embed,
                            control_latents, ref_teacher, clip_teacher, device, dtype)
            if cfg_scale and cfg_scale != 0 and neg_prompt_embed is not None:
                v_u = dit_forward(teacher_dit, x, t, neg_prompt_embed,
                                  control_latents, ref_teacher, clip_teacher, device, dtype)
                v = v_u + cfg_scale * (v - v_u)
            x0 = velocity_to_x0(v, x, sigma)
            if idx == K - 1:
                return x0
            sigma_next = float(teacher_step_list[idx + 1]) / NUM_TRAIN_TIMESTEPS
            # deterministic Euler step toward the next (lower) sigma
            x = x + (sigma_next - sigma) * v
        return x0


def highfreq_residual(x, k=3):
    """High-frequency component = x - blur(x), via a k×k average pool (depthwise,
    reflect pad) over the spatial dims. x: [B, C, T, H, W]. Lets the anchor L1
    transfer DETAIL only, leaving low-freq structure to DMD."""
    B, C, T, H, W = x.shape
    xr = x.reshape(B * C * T, 1, H, W)
    pad = k // 2
    xr = F.pad(xr, (pad, pad, pad, pad), mode="reflect")
    blur = F.avg_pool2d(xr, kernel_size=k, stride=1)
    blur = blur.reshape(B, C, T, H, W)
    return x - blur


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
    # ─── Teacher trajectory anchor (Progressive-Distillation style) ───
    p.add_argument("--anchor_weight", type=float, default=0.0,
                   help="Weight of the teacher trajectory-anchor L1. 0 disables "
                        "(file then == train_dmd_oneforcing.py). The anchor runs "
                        "the teacher K-step from the student's OWN initial noise "
                        "and regresses x_pred toward it (transfers the multi-step "
                        "detail the 1-step student misses).")
    p.add_argument("--anchor_steps", type=int, default=4,
                   help="Number of teacher ODE steps for the anchor target. More "
                        "= sharper target but K× teacher forwards/step.")
    p.add_argument("--anchor_start_step", type=int, default=200,
                   help="Don't apply the anchor until this global step (early "
                        "x_pred too rough; let DMD establish structure first).")
    p.add_argument("--anchor_highfreq", action="store_true", default=False,
                   help="Match only the HIGH-FREQUENCY residual (x - blur(x)) so "
                        "the anchor transfers detail while DMD owns structure.")
    p.add_argument("--anchor_hf_kernel", type=int, default=3,
                   help="Blur kernel size for the high-freq residual (odd).")
    p.add_argument("--anchor_cfg", type=float, default=0.0,
                   help="CFG scale for the teacher anchor trajectory. >0 = sharper "
                        "target, 2× teacher forwards. 0 = unguided (cheaper).")
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

        # Optimizer states
        state_path = args.resume_student_from.replace(".safetensors", "_state.pt")
        if os.path.isfile(state_path):
            ckpt = torch.load(state_path, map_location=device)
            student_optimizer.load_state_dict(ckpt["student_optimizer"])
            try:
                critic_optimizer.load_state_dict(ckpt["critic_optimizer"])
            except (ValueError, KeyError) as e:
                # Old checkpoint(without cls_branch in critic_optimizer)or
                # param-group size mismatch → skip critic optimizer state.
                if is_main:
                    print(f"[resume] critic_optimizer state mismatch ({e}) — cold start critic moments.")
            if is_main:
                print(f"[resume] student optimizer state from {state_path} "
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

    # Build the teacher anchor schedule: K descending steps whose FIRST entry
    # equals the student's single denoising step, so both interpret the shared
    # initial noise as the latent at the same sigma.
    teacher_step_list = None
    if args.anchor_weight > 0:
        student_pipe.scheduler.set_timesteps(
            num_inference_steps=args.anchor_steps, shift=args.flow_shift,
        )
        teacher_step_list = [float(t) for t in student_pipe.scheduler.timesteps]
        teacher_step_list[0] = denoising_step_list[0]   # align starting point
        if is_main:
            print(f"[setup] teacher trajectory anchor ENABLED: weight={args.anchor_weight}, "
                  f"steps={args.anchor_steps}, start_step={args.anchor_start_step}, "
                  f"highfreq={args.anchor_highfreq} (k={args.anchor_hf_kernel}), "
                  f"cfg={args.anchor_cfg}")
            print(f"[setup] teacher_step_list: {[round(t,1) for t in teacher_step_list]}")

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

            # ─── ROLLOUT student through the full schedule (random exit) ───
            # Replaces train_dmd.py's GT-noising + single forward. The student
            # sees its OWN intermediate states; only the exit step has grad.
            # x_pred is the exit-step x0 estimate (used for both critic update
            # and the DMD generator update below).
            # Shared initial noise: reused by the teacher anchor below so the
            # teacher's multi-step output is structurally aligned with x_pred.
            init_noise = torch.randn(tuple(target_latents.shape), dtype=dtype, device=device)
            x_pred, t_gen = rollout_student(
                student_pipe.dit, denoising_step_list, tuple(target_latents.shape),
                prompt_embed, control_latents, ref_student, clip_student,
                device, dtype,
                use_gradient_checkpointing=args.use_gradient_checkpointing,
                initial_noise=init_noise,
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

            # ─── Teacher trajectory-anchor loss (NEW) ───
            # Run the teacher K-step from the SAME init_noise → structurally
            # aligned, sharper target. Regress x_pred toward it (grad only
            # through x_pred). Optionally match high-freq residual only, so the
            # anchor transfers detail and DMD keeps owning low-freq structure.
            # Skipped before anchor_start_step and when anchor_weight == 0.
            anchor_loss = x_pred.new_zeros(())
            if args.anchor_weight > 0 and global_step >= args.anchor_start_step:
                x_teacher = teacher_trajectory(
                    teacher_pipe.dit, init_noise, teacher_step_list,
                    prompt_embed, control_latents, ref_teacher, clip_teacher,
                    device, dtype,
                    neg_prompt_embed=neg_prompt_embed, cfg_scale=args.anchor_cfg,
                ).detach()
                if args.anchor_highfreq:
                    anchor_loss = F.l1_loss(
                        highfreq_residual(x_pred, args.anchor_hf_kernel),
                        highfreq_residual(x_teacher, args.anchor_hf_kernel),
                    )
                else:
                    anchor_loss = F.l1_loss(x_pred, x_teacher)

            gen_loss = dmd_g_loss + g_loss + args.anchor_weight * anchor_loss
            student_optimizer.zero_grad()
            gen_loss.backward()
            all_reduce_grads(student_params, num_proc)   # sync across ranks
            student_optimizer.step()

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
                    an=f"{float(anchor_loss):.4f}", t=t_gen,
                )
                log_file.write(
                    f"{time.strftime('%H:%M:%S')} epoch={epoch} step={global_step} "
                    f"t_gen={t_gen} gen_loss={gen_loss.item():.6f} "
                    f"dmd_g={dmd_g_loss.item():.6f} gan_g={g_loss.item():.6f} "
                    f"critic_loss={critic_loss.item():.6f} gan_d={d_loss.item():.6f} "
                    f"anchor={float(anchor_loss):.6f}\n"
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
                    # Optimizer moments + step counter for seamless resume.
                    torch.save({
                        "student_optimizer": student_optimizer.state_dict(),
                        "critic_optimizer":  critic_optimizer.state_dict(),
                        "global_step": global_step,
                    }, state_path)
                    print(f"[save] {student_path} + ema:{ema_path} + {cls_path} + {state_path}")

    if is_main:
        log_file.write(f"=== run end {time.strftime('%Y-%m-%d %H:%M:%S')} | "
                       f"total_steps={args.global_step_offset + local_step} ===\n")
        log_file.close()


if __name__ == "__main__":
    main()
