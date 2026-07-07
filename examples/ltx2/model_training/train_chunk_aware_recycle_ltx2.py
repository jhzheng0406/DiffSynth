"""
LTX-2 port of the Wan chunk-aware teacher LoRA training + SVI-style latent
error recycle (examples/wanvideo/model_training/train_chunk_aware_recycle.py).

Purpose: cross-architecture generalization evidence for the recycle paper
story — same chunked long-video setting (sink + recent reference, 1-frame
overlap chaining), different bidirectional DiT + different VAE.

Conditioning mapping (Wan FunReference → LTX-2 input_images interface):
    recent  → input_images index 0, strength `--input_images_strength`.
              With the 1-frame-overlap chunk convention, the recent frame IS
              frame 0 of the current chunk, so index-0 replacement
              conditioning (model_fn_ltx2 swaps latent frame 0 in and sets
              its timestep to 0) is the natural analogue. At stitch time
              chunk k>0 drops its frame 0, same as the Wan chain script.
    sink    → input_images index `--sink_index` (default -49 ≈ one chunk in
              the past). Non-zero index makes InputImagesEmbedder append the
              latent as a t=0 reference token with time position
              (coords + index) / frame_rate — the analogue of Wan's
              reference token at ref T=0. Negative time = "past frame";
              verify RoPE handles this with the chaining sanity script
              (examples/ltx2/model_inference/LTX-2-Chunk-Chain-Sanity.py)
              BEFORE training.
    pose    → optional in_context_videos (IC-LoRA Pose-Control path),
              enabled with --use_pose_control. Requires the IC-LoRA weights
              fused via --preset_lora_path; whether pose-control handles
              cartoon pose still needs its own sanity check.

Recycle mechanism is identical to the Wan version:
  1. Units run; InputImagesEmbedder produces input_latents_video (recent at
     latent frame 0) + ref_frames_latents (sink token).
  2. install_recycle_hook_ltx2 wraps InputImagesEmbedder.process and adds
     alpha * buffer.sample() to input_latents_video[:, :, 0:1] ONLY — the
     sink reference token stays clean (global anchor must not drift).
  3. FlowMatchSFTVideoRecycleLoss (video-only FlowMatch, loss masked by
     denoise_mask_video so the replaced frame-0 region is excluded) captures
     pred_x0 at the last latent frame and pushes the residual
     pred_x0_last - gt_last into the per-rank FIFO.
  4. Same push gates as Wan: ref-was-clean flag, collect_start_step,
     push_prob.

Caveats (inherited + LTX-2 specific):
  - LATENT SPACE MISMATCH, worse than Wan: the residual is collected from
    the video VAE's last latent frame (temporal compression 8 — one latent
    covers 8 pixel frames), but injected into the recent latent which is a
    single-frame encode (latent frame 0 covers exactly 1 frame). On Wan the
    same mismatch existed with compression 4. Treated as a proxy; direction
    should be approximately right.
  - Audio: we train video-only (no input_audio). audio_latents is pure
    noise through model_fn and contributes no loss — same as the
    LTX-2-T2AV-noaudio recipe. Use --find_unused_parameters.
  - Loss masking: FlowMatchSFTAudioVideoLoss computes loss everywhere
    including the conditioned frame-0 tokens (where the model sees clean
    latents at t=0). We weight by denoise_mask_video instead, the LTX-2
    analogue of Wan's first_frame_latents exclusion.
"""
import argparse, os, random, sys
import accelerate
import decord
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from diffsynth.diffusion import ModelLogger, launch_training_task
from diffsynth.pipelines.ltx2_audio_video import (
    LTX2AudioVideoUnit_InputImagesEmbedder,
    LTX2AudioVideoUnit_NoiseInitializer,
    LTX2AudioVideoUnit_InContextVideoEmbedder,
)

from train import LTX2TrainingModule, ltx2_parser


# ---------------------------------------------------------------------------
# Image-level augmentation on the recent ref (copy of the Wan version —
# kept local so the ltx2 example tree has no import edge into wanvideo/)
# ---------------------------------------------------------------------------
def augment_recent(pil_img, strength=0.5):
    """Color jitter + gaussian pixel noise to simulate chain-inference drift.
    `strength` ∈ [0, ~2] scales every perturbation's magnitude.
    """
    if strength <= 0:
        return pil_img
    img = pil_img
    if random.random() < 0.9:
        img = ImageEnhance.Brightness(img).enhance(1.0 + random.uniform(-0.20, 0.20) * strength)
    if random.random() < 0.9:
        img = ImageEnhance.Contrast(img).enhance(1.0 + random.uniform(-0.20, 0.20) * strength)
    if random.random() < 0.9:
        img = ImageEnhance.Color(img).enhance(1.0 + random.uniform(-0.30, 0.30) * strength)
    if random.random() < 0.8:
        arr = np.asarray(img, dtype=np.float32)
        sigma = random.uniform(2.0, 8.0) * strength
        arr = arr + np.random.randn(*arr.shape).astype(np.float32) * sigma
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
    return img


# ---------------------------------------------------------------------------
# Dataset — same chunk enumeration as the Wan ChunkAwareDataset.
# chunk_frames must satisfy LTX-2's time_division: ≡ 1 (mod 8). 49 works.
# ---------------------------------------------------------------------------
class LTX2ChunkAwareDataset(torch.utils.data.Dataset):
    load_from_cache = False

    def __init__(self, csv_path, height, width, chunk_frames=49,
                 recent_aug_strength=0.5, dataset_repeat=1):
        assert chunk_frames % 8 == 1, f"LTX-2 needs chunk_frames ≡ 1 mod 8, got {chunk_frames}"
        src_df = pd.read_csv(csv_path)
        self.height = height
        self.width = width
        self.chunk_frames = chunk_frames
        self.recent_aug_strength = recent_aug_strength
        self.dataset_repeat = dataset_repeat

        self.samples = []
        stride = chunk_frames - 1
        skipped = 0
        for _, row in src_df.iterrows():
            n_frames = int(row.get("num_frames", 0))
            if n_frames <= 0:
                n_frames = int(decord.VideoReader(row["video"]).count_frames())
            n_chunks = max(0, (n_frames - chunk_frames) // stride + 1)
            if n_chunks == 0:
                skipped += 1
                continue
            # LTX-2's prompt convention is "[VISUAL]:... [SOUNDS]:..." — wrap
            # plain prompts (e.g. reused Wan metadata) so training matches the
            # inference-side format used by the sanity/chain scripts.
            prompt = str(row["prompt"])
            if "[VISUAL]" not in prompt:
                prompt = f"[VISUAL]:{prompt}"
            for k in range(n_chunks):
                self.samples.append({
                    "source_video": row["video"],
                    "source_pose":  row.get("control_video", None),
                    "chunk_start":  k * stride,
                    "prompt":       prompt,
                })
        print(f"[LTX2ChunkAwareDataset] {len(src_df)} source videos → {len(self.samples)} chunk samples"
              f"  (avg {len(self.samples) / max(1, len(src_df)):.1f}/video,  skipped {skipped} too-short)")

    def __len__(self):
        return len(self.samples) * self.dataset_repeat

    def _to_pil(self, frame_np):
        # Center-crop to the target aspect, THEN resize. The old direct
        # resize squished 960x960 sources into 480x832 (73% vertical
        # stretch), teaching the model deformed anatomy — v5's first-chunk
        # deformation traced back to this. Applies identically to video,
        # recent, sink and pose frames (all flow through here), so the
        # pose↔pixel correspondence is preserved.
        img = Image.fromarray(frame_np).convert("RGB")
        tw, th = self.width, self.height
        w, h = img.size
        if w * th > h * tw:        # too wide → crop width
            new_w = h * tw // th
            x0 = (w - new_w) // 2
            img = img.crop((x0, 0, x0 + new_w, h))
        elif w * th < h * tw:      # too tall → crop height
            new_h = w * th // tw
            y0 = (h - new_h) // 2
            img = img.crop((0, y0, w, y0 + new_h))
        return img.resize((tw, th))

    def __getitem__(self, idx):
        sample = self.samples[idx % len(self.samples)]
        s = sample["chunk_start"]

        vr_vid = decord.VideoReader(sample["source_video"])
        chunk_idx = np.arange(s, s + self.chunk_frames)
        target_np = vr_vid.get_batch(chunk_idx).asnumpy()

        sink_pil = self._to_pil(vr_vid[0].asnumpy())

        # recent = frame at target[0]'s position (1-frame-overlap convention,
        # same as Wan: equals previous chunk's last frame at chain time).
        # The clean (un-augmented) recent is also returned for consumers that
        # route clean/aug separately (DMD asymmetric mode); Stage A ignores it.
        if s == 0:
            recent_clean_pil = sink_pil
            recent_pil = sink_pil
        else:
            recent_clean_pil = self._to_pil(vr_vid[s].asnumpy())
            recent_pil = augment_recent(recent_clean_pil,
                                        strength=self.recent_aug_strength)

        out = {
            "video":        [self._to_pil(f) for f in target_np],
            "recent_image": recent_pil,
            "recent_image_clean": recent_clean_pil,
            "sink_image":   sink_pil,
            "prompt":       sample["prompt"],
        }
        if sample["source_pose"]:
            vr_pose = decord.VideoReader(sample["source_pose"])
            pose_np = vr_pose.get_batch(chunk_idx).asnumpy()
            out["control_video"] = [self._to_pil(f) for f in pose_np]
        return out


# ---------------------------------------------------------------------------
# Error buffer + alpha schedule (identical to Wan version)
# ---------------------------------------------------------------------------
class ErrorBuffer:
    """Per-rank FIFO of last-latent-frame residuals, shape [C, h, w]."""
    def __init__(self, max_size: int = 500):
        self.max_size = max_size
        self.buf = []

    def push(self, err: torch.Tensor):
        self.buf.append(err.detach().to(torch.bfloat16).cpu().contiguous())
        if len(self.buf) > self.max_size:
            self.buf.pop(0)

    def sample(self, device=None, dtype=None) -> torch.Tensor:
        assert len(self.buf) > 0, "ErrorBuffer empty"
        out = self.buf[torch.randint(0, len(self.buf), (1,)).item()]
        if device is not None: out = out.to(device)
        if dtype is not None:  out = out.to(dtype)
        return out

    def __len__(self):
        return len(self.buf)


class AlphaSchedule:
    """0 during warmup, linear ramp 0→max over `ramp` steps, then constant."""
    def __init__(self, max_alpha: float, warmup: int, ramp: int):
        self.max_alpha = max_alpha
        self.warmup = warmup
        self.ramp = max(1, ramp)

    def __call__(self, step: int) -> float:
        if step < self.warmup:
            return 0.0
        return self.max_alpha * min(1.0, (step - self.warmup) / self.ramp)


# ---------------------------------------------------------------------------
# Time-shift hook — the "positive-shift" sink mapping (replaces sink_index=-49)
#
# Mirrors Wan-Fun's prepend-and-shift: instead of placing the sink token at a
# NEGATIVE time (out-of-distribution for RoPE — the zero-shot sanity measured
# motion jitter from it), shift the WHOLE chunk's time coords by
# +time_shift_frames (one chunk, 48f → chunk lives at t∈[2s,4s]) and put the
# sink at small positive index (~t=0). All coordinates stay positive and
# inside the trained range; the only novelty is "video doesn't start at 0".
#
# The pose-control in_context tokens are shifted by the SAME offset (official
# IC design aligns them with the video's time axis). Audio positions are NOT
# shifted: the audio tower idles on pure noise (no signal → misalignment is
# harmless), and its position format differs.
# ---------------------------------------------------------------------------
def install_time_shift_hook_ltx2(pipe, shift_frames: int):
    n_patched = 0
    for unit in pipe.units:
        if isinstance(unit, LTX2AudioVideoUnit_NoiseInitializer):
            orig = unit.process
            def wrapped_noise(pipe_arg, _orig=orig, **kwargs):
                out = _orig(pipe_arg, **kwargs)
                fr = kwargs.get("frame_rate", 24.0) or 24.0
                out["video_positions"] = out["video_positions"].clone()
                out["video_positions"][:, 0, ...] += shift_frames / fr
                return out
            unit.process = wrapped_noise
            n_patched += 1
        if isinstance(unit, LTX2AudioVideoUnit_InContextVideoEmbedder):
            orig_ic = unit.process
            def wrapped_ic(pipe_arg, _orig=orig_ic, **kwargs):
                out = _orig(pipe_arg, **kwargs)
                if out.get("in_context_video_positions") is not None:
                    fr = kwargs.get("frame_rate", 24.0) or 24.0
                    out["in_context_video_positions"] = out["in_context_video_positions"].clone()
                    out["in_context_video_positions"][:, 0, ...] += shift_frames / fr
                return out
            unit.process = wrapped_ic
            n_patched += 1
    return n_patched


# ---------------------------------------------------------------------------
# Hooks on InputImagesEmbedder — corrupt ONLY the recent slice
# (input_latents_video latent frame 0). Sink ref token stays clean.
# ---------------------------------------------------------------------------
def install_ead_hook_ltx2(pipe, corrupt_ratio: float, clean_prob: float):
    """Helios-style EAD on the recent latent:
        x = sigma * randn + (1 - sigma) * x,  sigma ~ U[0, corrupt_ratio]
    """
    n_patched = 0
    for unit in pipe.units:
        if isinstance(unit, LTX2AudioVideoUnit_InputImagesEmbedder):
            orig_process = unit.process

            def wrapped(pipe_arg, _orig=orig_process, **kwargs):
                out = _orig(pipe_arg, **kwargs)
                lat = out.get("input_latents_video") if isinstance(out, dict) else None
                if lat is not None and random.random() >= clean_prob:
                    recent = lat[:, :, 0:1]
                    sigma_shape = (recent.shape[0],) + (1,) * (recent.ndim - 1)
                    sigma = torch.rand(*sigma_shape, device=lat.device, dtype=lat.dtype) * corrupt_ratio
                    lat_new = lat.clone()
                    lat_new[:, :, 0:1] = sigma * torch.randn_like(recent) + (1 - sigma) * recent
                    out["input_latents_video"] = lat_new
                    pipe_arg._recycle_ref_clean = False
                return out

            unit.process = wrapped
            n_patched += 1
    return n_patched


def install_recycle_hook_ltx2(pipe, buffer: ErrorBuffer, alpha_fn, clean_prob: float):
    """Adds alpha * buffer.sample() to the recent latent slice. No-ops while
    the buffer is empty (warmup) or with probability clean_prob."""
    n_patched = 0
    for unit in pipe.units:
        if isinstance(unit, LTX2AudioVideoUnit_InputImagesEmbedder):
            orig_process = unit.process

            def wrapped(pipe_arg, _orig=orig_process, **kwargs):
                out = _orig(pipe_arg, **kwargs)
                lat = out.get("input_latents_video") if isinstance(out, dict) else None
                if lat is None or len(buffer) == 0:
                    return out
                if random.random() < clean_prob:
                    return out
                alpha = alpha_fn(getattr(pipe_arg, "_recycle_step", 0))
                if alpha <= 0:
                    return out
                err = buffer.sample(device=lat.device, dtype=lat.dtype)   # [C, h, w]
                lat_new = lat.clone()
                lat_new[:, :, 0:1] = lat[:, :, 0:1] + alpha * err.view(1, err.shape[0], 1, *err.shape[1:])
                out["input_latents_video"] = lat_new
                pipe_arg._recycle_ref_clean = False
                return out

            unit.process = wrapped
            n_patched += 1
    return n_patched


# ---------------------------------------------------------------------------
# Loss — video-only FlowMatch, denoise-mask-weighted, + gated buffer push
# ---------------------------------------------------------------------------
def FlowMatchSFTVideoRecycleLoss(pipe, buffer: ErrorBuffer, push_prob: float,
                                 collect_start_step: int, **inputs):
    """Video branch of FlowMatchSFTAudioVideoLoss with two changes:
      1. Loss is weighted by denoise_mask_video — the index-0 conditioned
         region (mask = 1 - strength) is down-weighted/excluded, the LTX-2
         analogue of Wan's first-frame exclusion.
      2. Pushes pred_x0_last - gt_last residuals to the recycle buffer under
         the same gates as the Wan version (clean-ref flag, collect_start,
         push_prob).
    pred_x0 under flow matching: target = noise - x0  →  x0 = noise - v_pred.
    """
    max_tb = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_tb = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    timestep_id = torch.randint(min_tb, max_tb, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)

    input_latents = inputs["input_latents"]
    noise = torch.randn_like(input_latents)
    inputs["video_latents"] = pipe.scheduler.add_noise(input_latents, noise, timestep)
    training_target = pipe.scheduler.training_target(input_latents, noise, timestep)

    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred, _ = pipe.model_fn(**models, **inputs, timestep=timestep)

    se = (noise_pred.float() - training_target.float()) ** 2
    mask = inputs.get("denoise_mask_video")
    if mask is not None:
        w = mask.float()                                   # [B, 1, f, h, w]
        loss = (se * w).sum() / (w.sum() * se.shape[1]).clamp(min=1.0)
    else:
        loss = se.mean()
    loss = loss * pipe.scheduler.training_weight(timestep)

    ref_was_clean = getattr(pipe, "_recycle_ref_clean", True)
    cur_step = getattr(pipe, "_recycle_step", 0)
    can_push = (
        buffer is not None
        and ref_was_clean
        and cur_step >= collect_start_step
        and random.random() < push_prob
    )
    if can_push:
        with torch.no_grad():
            pred_x0 = noise.float() - noise_pred.float()
            err = pred_x0[:, :, -1] - input_latents[:, :, -1].float()   # [B, C, h, w]
            for b in range(err.shape[0]):
                buffer.push(err[b])

    return loss


# ---------------------------------------------------------------------------
# Training module
# ---------------------------------------------------------------------------
class LTX2ChunkRecycleTrainingModule(LTX2TrainingModule):
    def __init__(self, *args, buffer=None, push_prob=0.5, collect_start_step=0,
                 step_offset=0, sink_index=-49, no_sink=False,
                 input_images_strength=1.0,
                 use_pose_control=False, pose_downsample_factor=2,
                 frame_rate=24, **kwargs):
        super().__init__(*args, **kwargs)
        self.buffer = buffer
        self.push_prob = push_prob
        self.collect_start_step = collect_start_step
        self.sink_index = sink_index
        self.no_sink = no_sink
        self.input_images_strength = input_images_strength
        self.use_pose_control = use_pose_control
        self.pose_downsample_factor = pose_downsample_factor
        self.frame_rate = frame_rate
        self.pipe._recycle_step = step_offset

        recycle_loss = lambda pipe, ish, ipi, ina: FlowMatchSFTVideoRecycleLoss(
            pipe, self.buffer, self.push_prob, self.collect_start_step,
            **ish, **ipi,
        )
        self.task_to_loss["sft"]       = recycle_loss
        self.task_to_loss["sft:train"] = recycle_loss

    def get_pipeline_inputs(self, data):
        inputs_shared, inputs_posi, inputs_nega = super().get_pipeline_inputs(data)
        inputs_shared["frame_rate"] = self.frame_rate
        # sink + recent conditioning. sink_index must not be 0 (that's the
        # recent's replacement slot) — enforced at parse time.
        # --no_sink: recent-only chaining (LTX-2's official i2v usage; the
        # zero-shot sanity validated this arm — sanity showed sink_index=-8
        # is harmful, -49 pending a same-image rerun).
        if self.no_sink:
            inputs_shared["input_images"] = [data["recent_image"]]
            inputs_shared["input_images_indexes"] = [0]
        else:
            inputs_shared["input_images"] = [data["recent_image"], data["sink_image"]]
            inputs_shared["input_images_indexes"] = [0, self.sink_index]
        inputs_shared["input_images_strength"] = self.input_images_strength
        if self.use_pose_control:
            inputs_shared["in_context_videos"] = [data["control_video"]]
            inputs_shared["in_context_downsample_factor"] = self.pose_downsample_factor
        return inputs_shared, inputs_posi, inputs_nega

    def forward(self, data, inputs=None):
        # Reset BEFORE hooks fire; hooks flip it to False when they corrupt.
        self.pipe._recycle_ref_clean = True
        loss = super().forward(data, inputs=inputs)
        self.pipe._recycle_step += 1
        return loss


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parser():
    p = ltx2_parser()
    p.add_argument("--recent_aug_strength", type=float, default=0.5,
                   help="Image-level aug strength on recent ref (0 = off).")
    p.add_argument("--sink_index", type=int, default=-49,
                   help="input_images index for the sink (≠ 0). Negative = "
                        "past-time reference token. Validate with the "
                        "chaining sanity script before training.")
    p.add_argument("--no_sink", default=False, action="store_true",
                   help="Recent-only conditioning (no sink token) — the "
                        "official-i2v fallback arm validated by the sanity.")
    p.add_argument("--time_shift_frames", type=int, default=0,
                   help="Positive-shift sink mapping: shift the chunk's time "
                        "coords by +N frames so the sink can sit at a small "
                        "POSITIVE index (~t=0) instead of a negative one. "
                        "Use 48 (one chunk) with --sink_index 1.")
    p.add_argument("--input_images_strength", type=float, default=1.0,
                   help="Replacement strength for the recent at index 0. "
                        "1.0 = frame 0 fully clamped to the recent latent.")
    p.add_argument("--use_pose_control", default=False, action="store_true",
                   help="Pass control_video as in_context_videos (IC-LoRA "
                        "path; fuse pose-control weights via --preset_lora_path).")
    p.add_argument("--pose_downsample_factor", type=int, default=2)
    p.add_argument("--ead_corrupt_ratio", type=float, default=0.0,
                   help="Latent Gaussian EAD on the recent slice (0 = off).")
    p.add_argument("--ead_clean_prob", type=float, default=0.1)
    p.add_argument("--recycle_alpha",       type=float, default=0.3)
    p.add_argument("--recycle_clean_prob",  type=float, default=0.1)
    p.add_argument("--recycle_buffer_size", type=int,   default=500)
    p.add_argument("--recycle_warmup",      type=int,   default=200)
    p.add_argument("--recycle_ramp",        type=int,   default=500)
    p.add_argument("--recycle_push_prob",   type=float, default=0.5)
    p.add_argument("--recycle_collect_start_step", type=int, default=100)
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    assert args.sink_index != 0, "--sink_index 0 collides with the recent's replacement slot"

    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(
            find_unused_parameters=args.find_unused_parameters)],
    )
    is_main = accelerator.is_main_process

    dataset = LTX2ChunkAwareDataset(
        csv_path=args.dataset_metadata_path,
        height=args.height,
        width=args.width,
        chunk_frames=args.num_frames,
        recent_aug_strength=args.recent_aug_strength,
        dataset_repeat=args.dataset_repeat,
    )

    buffer = ErrorBuffer(max_size=args.recycle_buffer_size)
    alpha_fn = AlphaSchedule(
        max_alpha=args.recycle_alpha,
        warmup=args.recycle_warmup,
        ramp=args.recycle_ramp,
    )

    model = LTX2ChunkRecycleTrainingModule(
        buffer=buffer,
        push_prob=args.recycle_push_prob,
        collect_start_step=args.recycle_collect_start_step,
        step_offset=args.global_step_offset,
        sink_index=args.sink_index,
        no_sink=args.no_sink,
        input_images_strength=args.input_images_strength,
        use_pose_control=args.use_pose_control,
        pose_downsample_factor=args.pose_downsample_factor,
        frame_rate=args.frame_rate,
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        task=args.task,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
    )

    if args.time_shift_frames > 0:
        n = install_time_shift_hook_ltx2(model.pipe, args.time_shift_frames)
        if is_main:
            print(f"[time-shift] hooked {n} unit(s) — chunk coords +{args.time_shift_frames} frames "
                  f"(positive-shift sink mapping, sink_index={args.sink_index})")

    # EAD first, then recycle — same chaining order/semantics as the Wan
    # script: final corruption is recycle(EAD(recent_latent)).
    if args.ead_corrupt_ratio > 0:
        n = install_ead_hook_ltx2(model.pipe, args.ead_corrupt_ratio, args.ead_clean_prob)
        if is_main:
            print(f"[EAD] hooked {n} InputImagesEmbedder unit(s) — "
                  f"corrupt_ratio={args.ead_corrupt_ratio}, clean_prob={args.ead_clean_prob}")
    if args.recycle_alpha > 0:
        n = install_recycle_hook_ltx2(model.pipe, buffer, alpha_fn, args.recycle_clean_prob)
        if is_main:
            print(f"[recycle] hooked {n} InputImagesEmbedder unit(s) — "
                  f"alpha={args.recycle_alpha}, warmup={args.recycle_warmup}, "
                  f"ramp={args.recycle_ramp}, buffer={args.recycle_buffer_size}, "
                  f"clean_prob={args.recycle_clean_prob}, push_prob={args.recycle_push_prob}, "
                  f"collect_start={args.recycle_collect_start_step}, "
                  f"step_offset={args.global_step_offset}")
    if is_main:
        print(f"[recent_aug] strength={args.recent_aug_strength}  "
              f"[sink_index] {args.sink_index}  "
              f"[recent_strength] {args.input_images_strength}  "
              f"[pose_control] {args.use_pose_control}")

    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        step_offset=args.global_step_offset,
    )

    launch_training_task(accelerator, dataset, model, model_logger, args=args)
