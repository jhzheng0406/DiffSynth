"""
DMD2 + SELF-FORCING ROLLOUT + FD-LOSS  (GAN-free One-Forcing variant).

Replaces One-Forcing's GAN branch with a Fréchet-distance loss in DINOv2
feature space (paper: "Representation Fréchet Loss for Visual Generation",
Jiawei-Yang/FD-loss).

Pipeline per training step:
    1. Student rollout              → x_pred (with grad)
    2. Critic denoising MSE × N     (existing DMD critic)
    3. Generator update × 1:
        a. DMD distribution-matching loss(real_score − fake_score grad)
        b. VAE.decode(x_pred) → pixels                (with grad through x_pred)
        c. DINOv2 per-frame features → new_feats
        d. all_feats = queue + new_feats                (queue is detached)
        e. fd_loss = compute_fd_loss(mu_ref, sigma_ref, all_feats)
        f. gen_loss = dmd_loss + fd_weight * fd_loss
        g. backward, step student, queue.enqueue(new_feats.detach())

Why FD instead of GAN: no discriminator → no adversarial dynamics,
deterministic gradient, much more stable training.

Reference stats (mu_ref, sigma_ref) must be precomputed once on real data:
    python examples/wanvideo/model_training/precompute_fd_stats.py
    → ./data/cartoon_15s/fd_stats_dinov2.npz

Run via lora/Wan2.1-Fun-V1.1-1.3B-Control_cartoon_sink_dmd_fd.sh
"""
import argparse, os, random, sys, time
import accelerate
import decord
import numpy as np
import pandas as pd
import torch
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
from fd_loss import (
    FeatureQueue, VideoFeatureExtractor,
    compute_fd_loss, precompute_sigma_ref_sqrt, load_fd_stats,
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
):
    """Roll the student through the full few-step denoising schedule, starting
    from PURE NOISE. Random exit step → only one forward keeps grad; everything
    earlier runs under no_grad with re-noising between steps. The exit step's
    x0 estimate is returned as x_pred (the model's "final" output at that step).

    Returns: (x_pred, exit_t)  where exit_t is the timestep at the exit step.

    Mirrors pipeline/bidirectional_training.py:inference_with_trajectory in the
    upstream Causal-Forcing repo, simplified for our chunk-based setting.
    """
    N = len(denoising_step_list)
    exit_idx = random.randrange(N)        # uniform over 0..N-1

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
    p.add_argument("--fd_stats_path", required=True,
                   help="Precomputed reference stats .npz from precompute_fd_stats.py.")
    p.add_argument("--fd_weight", type=float, default=0.1,
                   help="Weight on FD loss added to dmd_loss. Try 0.05-0.3 range.")
    p.add_argument("--fd_feature_model", default="dinov2_vitb14",
                   help="Frozen backbone for per-frame features.")
    p.add_argument("--fd_queue_size", type=int, default=5000,
                   help="Ring-buffer size for detached features (larger → "
                        "more stable statistics, more memory).")
    p.add_argument("--fd_num_frames", type=int, default=8,
                   help="How many frames per chunk to sample for FD (full=49 "
                        "is heaviest; 4-8 is a fine subsample).")
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

    # ---------------- Optimizers ----------------
    student_params = trainable_params(student_pipe.dit)
    critic_params  = trainable_params(critic_pipe.dit)
    if is_main:
        print(f"[setup] student trainable params: {sum(p.numel() for p in student_params)/1e6:.1f}M")
        print(f"[setup] critic  trainable params: {sum(p.numel() for p in critic_params )/1e6:.1f}M")

    student_optimizer = torch.optim.AdamW(student_params, lr=args.learning_rate_student, weight_decay=1e-2)
    critic_optimizer  = torch.optim.AdamW(critic_params,  lr=args.learning_rate_critic,  weight_decay=1e-2)

    # ---------------- FD-loss setup ----------------
    if is_main: print(f"[setup] loading FD feature extractor: {args.fd_feature_model}")
    fd_extractor = VideoFeatureExtractor(args.fd_feature_model, dtype=dtype).to(device)
    if is_main: print(f"[setup] loading FD reference stats: {args.fd_stats_path}")
    mu_ref, sigma_ref = load_fd_stats(args.fd_stats_path)
    mu_ref    = mu_ref.to(device)
    sigma_ref = sigma_ref.to(device)
    sigma_ref_sqrt = precompute_sigma_ref_sqrt(sigma_ref).to(device)
    fd_queue = FeatureQueue(args.fd_queue_size, fd_extractor.feat_dim).to(device)
    if is_main:
        print(f"[setup] FD queue: size={args.fd_queue_size}, "
              f"feat_dim={fd_extractor.feat_dim}, "
              f"frames/chunk={args.fd_num_frames}, weight={args.fd_weight}")

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
            x_pred, t_gen = rollout_student(
                student_pipe.dit, denoising_step_list, tuple(target_latents.shape),
                prompt_embed, control_latents, ref_student, clip_student,
                device, dtype,
                use_gradient_checkpointing=args.use_gradient_checkpointing,
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
                all_reduce_grads(critic_params, num_proc)   # sync across ranks
                critic_optimizer.step()

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

            # ─── FD-loss: VAE-decode x_pred → DINOv2 features → Frechet to ref ───
            # Subsample fd_num_frames frames from the chunk's T_lat=13 latent dim,
            # then VAE decode (memory-bounded). With grad through x_pred.
            T_lat = x_pred.shape[2]
            if args.fd_num_frames < T_lat:
                idx = torch.randperm(T_lat, device=device)[: args.fd_num_frames]
                x_pred_sub = x_pred.index_select(2, idx)
            else:
                x_pred_sub = x_pred
            # Wan VAE decode wants [B, C, T_lat, H_lat, W_lat] → returns [B, 3, T, H, W] in [-1,1]
            with torch.autograd.graph.save_on_cpu(pin_memory=False):
                decoded = student_pipe.vae.decode(x_pred_sub.to(dtype),
                                                  device=device).to(dtype)
                pixels = ((decoded.clamp(-1, 1) + 1) / 2)              # [B, 3, T, H, W]
                pixels = pixels.permute(0, 2, 1, 3, 4)                  # [B, T, 3, H, W]
                new_feats = fd_extractor(pixels)                        # [B*T, D]
            all_feats = torch.cat([fd_queue.get_valid(), new_feats], dim=0)
            fd_g_loss = compute_fd_loss(mu_ref, sigma_ref, all_feats,
                                        sigma_ref_sqrt=sigma_ref_sqrt)

            gen_loss = dmd_g_loss + args.fd_weight * fd_g_loss
            student_optimizer.zero_grad()
            gen_loss.backward()
            all_reduce_grads(student_params, num_proc)   # sync across ranks
            student_optimizer.step()

            # Enqueue current batch's features (detached) for next step
            fd_queue.enqueue(new_feats.detach())

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
                pbar.set_postfix(
                    g=f"{gen_loss.item():.4f}", c=f"{critic_loss.item():.4f}",
                    fd=f"{fd_g_loss.item():.2f}", q=int(fd_queue.count.item()), t=t_gen,
                )
                log_file.write(
                    f"{time.strftime('%H:%M:%S')} epoch={epoch} step={global_step} "
                    f"t_gen={t_gen} gen_loss={gen_loss.item():.6f} "
                    f"dmd_g={dmd_g_loss.item():.6f} fd={fd_g_loss.item():.6f} "
                    f"q={int(fd_queue.count.item())} "
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
