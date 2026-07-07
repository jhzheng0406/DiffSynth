"""
Teacher LoRA training (chunk-aware) + SVI-style latent-space error recycle.

Builds on train_chunk_aware.py:
  ChunkAwareDataset unchanged — still does color/noise aug on recent PIL.
  Replaces Helios EAD (random Gaussian on reference_latents) with a
  per-rank FIFO buffer of TEACHER prediction-error residuals at the last
  latent frame, sampled and added back into reference_latents during the
  next iterations.

Mechanism per training step:
  1. Pipeline units run (text/VAE/recent encoding) — reference_latents ready.
  2. install_recycle_hook has wrapped FunReference.process so reference_latents
     is corrupted as:
         alpha = alpha_fn(global_step)            # 0 during warmup, ramps up
         err   = buffer.sample()                   # [C, h, w]
         reference_latents = reference_latents + alpha * err
     With probability clean_prob we skip corruption (keeps a clean signal).
  3. Standard FlowMatch loss runs through pipe.model_fn — we wrap it with
     FlowMatchSFTRecycleLoss which is a drop-in copy of FlowMatchSFTLoss
     EXCEPT it captures v_pred and stores pred_x0 last latent frame.
  4. err_t = pred_x0[:, :, -1] - input_latents[:, :, -1]   (detached)
     buffer.push(err_t[0])                               (rank-local)

Notes:
  - 'Teacher error' here is the model's instantaneous prediction error at
    the last latent frame, NOT autoregressive rollout drift. As the LoRA
    converges, these errors shrink. So this acts as a learned/shaped
    version of EAD rather than a true rollout-recycle. A future variant
    could periodically rollout the teacher to harvest real drift samples.
  - Buffer is per-rank (no cross-rank sync). With 8 ranks that's 8 ×
    --recycle_buffer_size effective diversity.
  - α schedule: 0 during buffer warmup (--recycle_warmup steps), then
    linear ramp to --recycle_alpha over --recycle_ramp steps.

  - PUSH SAFETY: residual is pushed ONLY when this step's reference was
    seen CLEAN by the model. Otherwise pred_x0 includes bias from the
    injected corruption, and pushing it would create a feedback loop
    where injected noise gets re-amplified. The hooks flag
    `pipe._recycle_ref_clean = False` whenever they corrupt; the loss
    refuses to push unless the flag is still True.

Caveats (inherited from SVI / DMD-recycle design):
  - LATENT SPACE MISMATCH: push collects from input_latents[:, :, -1]
    which is one slice of the 13-frame video VAE encoding (its temporal
    convs see neighbouring frames). The corruption is added to the recent
    slice of reference_latents, which is a SINGLE-frame image VAE encode
    (no temporal context). The two latents share spatial dims but their
    distributions are not identical. First-pass treats this as a proxy
    — direction should be approximately right.
"""
import argparse, os, random, sys
import accelerate
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from diffsynth.diffusion import ModelLogger, launch_training_task
from diffsynth.pipelines.wan_video import WanVideoUnit_FunReference

from train import WanTrainingModule
from train_chunk_aware import ChunkAwareDataset, wan_parser as chunk_aware_parser


# ===========================================================================
# Error buffer — FIFO of single-latent-frame residuals
# ===========================================================================
class ErrorBuffer:
    """Per-rank FIFO of last-latent-frame residuals, shape [C, h, w]."""
    def __init__(self, max_size: int = 500):
        self.max_size = max_size
        self.buf = []   # CPU bf16 tensors

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


# ===========================================================================
# Alpha schedule
# ===========================================================================
class AlphaSchedule:
    """0 during warmup, linear ramp 0→max over `ramp` steps, then constant."""
    def __init__(self, max_alpha: float, warmup: int, ramp: int):
        self.max_alpha = max_alpha
        self.warmup = warmup
        self.ramp = max(1, ramp)

    def __call__(self, step: int) -> float:
        if step < self.warmup:
            return 0.0
        progress = min(1.0, (step - self.warmup) / self.ramp)
        return self.max_alpha * progress


# ===========================================================================
# EAD hook (local copy, corrected for FunReference's **kwargs signature)
# ===========================================================================
def install_ead_hook(pipe, corrupt_ratio: float, clean_prob: float):
    """Helios-style EAD: corrupt reference_latents with Gaussian noise,
        x_corrupted = sigma * randn + (1 - sigma) * x_clean,  sigma ~ U[0, corrupt_ratio]
    With prob clean_prob, skip corruption.

    This is a corrected fork of `train.install_ead_hook` — that version's
    wrapper hard-codes (reference_image, height, width) and breaks under the
    current FunReference that also takes `sink_reference_image`.
    """
    n_patched = 0
    for unit in pipe.units:
        if isinstance(unit, WanVideoUnit_FunReference):
            orig_process = unit.process

            def wrapped(pipe_arg, _orig=orig_process, **kwargs):
                out = _orig(pipe_arg, **kwargs)
                ref = out.get("reference_latents") if isinstance(out, dict) else None
                if ref is not None and random.random() >= clean_prob:
                    sigma_shape = (ref.shape[0],) + (1,) * (ref.ndim - 1)
                    sigma = torch.rand(*sigma_shape, device=ref.device, dtype=ref.dtype) * corrupt_ratio
                    out["reference_latents"] = sigma * torch.randn_like(ref) + (1 - sigma) * ref
                    # Flag ref as corrupted so the loss won't push contaminated residuals.
                    pipe_arg._recycle_ref_clean = False
                return out

            unit.process = wrapped
            n_patched += 1
    return n_patched


# ===========================================================================
# Recycle hook — analogous to EAD but samples residuals from a buffer
# ===========================================================================
def install_recycle_hook(pipe, buffer: ErrorBuffer, alpha_fn, clean_prob: float):
    """Monkey-patch FunReference.process so reference_latents gets corrupted
    with `alpha * sampled_error` from the buffer. No-ops when buffer is empty
    (warmup) or random() < clean_prob.
    """
    n_patched = 0
    for unit in pipe.units:
        if isinstance(unit, WanVideoUnit_FunReference):
            orig_process = unit.process

            def wrapped(pipe_arg, _orig=orig_process, **kwargs):
                # Use **kwargs so we don't need to know FunReference.process's
                # exact arg list (which has changed across DiffSynth versions
                # — now includes sink_reference_image).
                out = _orig(pipe_arg, **kwargs)
                ref = out.get("reference_latents") if isinstance(out, dict) else None
                if ref is None or len(buffer) == 0:
                    return out
                if random.random() < clean_prob:
                    return out

                alpha = alpha_fn(pipe_arg._recycle_step) if hasattr(pipe_arg, "_recycle_step") else 0.0
                if alpha <= 0:
                    return out

                # ref shape: [B, C, T_ref, H, W]. T_ref is 2 here (sink @ T=0,
                # recent @ T=1) per WanVideoUnit_FunReference.process. We only
                # corrupt the RECENT frame (T=1) — sink is the clean global
                # anchor and shouldn't drift.
                err = buffer.sample(device=ref.device, dtype=ref.dtype)   # [C, h, w]
                err = err.view(1, err.shape[0], 1, err.shape[1], err.shape[2])
                if ref.shape[2] >= 2:
                    # Corrupt only the recent slice (last T).
                    ref_new = ref.clone()
                    ref_new[:, :, -1:] = ref[:, :, -1:] + alpha * err
                    out["reference_latents"] = ref_new
                else:
                    out["reference_latents"] = ref + alpha * err
                # Flag ref as corrupted — prevents polluted buffer push downstream.
                pipe_arg._recycle_ref_clean = False
                return out

            unit.process = wrapped
            n_patched += 1
    return n_patched


# ===========================================================================
# Custom loss — FlowMatchSFT + push residual to buffer
# ===========================================================================
def FlowMatchSFTRecycleLoss(pipe, buffer: ErrorBuffer, push_prob: float,
                            collect_start_step: int, **inputs):
    """Mirror of FlowMatchSFTLoss, but ALSO captures pred_x0_last and pushes
    `pred_x0_last - gt_last` to the buffer with probability push_prob.

    pred_x0 derivation under flow matching: training_target = noise - x_0,
    and the model predicts that, so x_0_pred = noise - v_pred (no sigma needed).

    Push is GATED on three conditions to keep the buffer clean:
      1. The hooks didn't corrupt this step's reference (pipe._recycle_ref_clean)
         — otherwise pred carries bias from the injected noise (feedback loop).
      2. We're past collect_start_step — LoRA's cold-start predictions are
         too noisy to be useful as buffer entries.
      3. random.random() < push_prob — keeps buffer diverse / slow turnover.
    """
    max_tb = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_tb = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    timestep_id = torch.randint(min_tb, max_tb, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)

    input_latents = inputs["input_latents"]
    noise = torch.randn_like(input_latents)
    inputs["latents"] = pipe.scheduler.add_noise(input_latents, noise, timestep)
    training_target = pipe.scheduler.training_target(input_latents, noise, timestep)

    if "first_frame_latents" in inputs:
        inputs["latents"][:, :, 0:1] = inputs["first_frame_latents"]

    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep)

    if "first_frame_latents" in inputs:
        noise_pred_loss = noise_pred[:, :, 1:]
        training_target = training_target[:, :, 1:]
    else:
        noise_pred_loss = noise_pred

    loss = torch.nn.functional.mse_loss(noise_pred_loss.float(), training_target.float())
    loss = loss * pipe.scheduler.training_weight(timestep)

    # ----- Buffer push (gated) -----
    ref_was_clean = getattr(pipe, "_recycle_ref_clean", True)
    cur_step = getattr(pipe, "_recycle_step", 0)
    can_push = (
        buffer is not None
        and ref_was_clean                                 # bug-1 guard
        and cur_step >= collect_start_step                # bug-3 guard
        and random.random() < push_prob
    )
    if can_push:
        with torch.no_grad():
            pred_x0 = noise.float() - noise_pred.float()
            # err = predicted_last - gt_last  (sign matches SVI: positive =
            # model's drift direction at inference).
            err = pred_x0[:, :, -1] - input_latents[:, :, -1].float()   # [B, C, h, w]
            # Push every batch element (DMD-recycle convention).
            for b in range(err.shape[0]):
                buffer.push(err[b])

    return loss


# ===========================================================================
# Training module — overrides task_to_loss to use the recycle variant
# ===========================================================================
class WanRecycleTrainingModule(WanTrainingModule):
    def __init__(self, *args, buffer=None, push_prob=0.5,
                 collect_start_step=0, step_offset=0, plp=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.buffer = buffer
        self.push_prob = push_prob
        self.collect_start_step = collect_start_step
        self.plp = plp
        # `_recycle_step` is read by hooks (alpha schedule) and by the loss
        # (collect_start gate). Initialised from --global_step_offset so the
        # alpha schedule resumes correctly across restarts.
        self.pipe._recycle_step = step_offset

        # Replace the SFT loss entry with our recycle variant.
        recycle_loss = lambda pipe, ish, ipi, ina: FlowMatchSFTRecycleLoss(
            pipe, self.buffer, self.push_prob, self.collect_start_step,
            **ish, **ipi,
        )
        self.task_to_loss["sft"]       = recycle_loss
        self.task_to_loss["sft:train"] = recycle_loss

    # PLP recent-latent encoding now lives in the base WanTrainingModule.
    # get_pipeline_inputs (gated on the dataset's recent_window), so it applies
    # to every chunk-aware path uniformly — no override needed here.

    def forward(self, data, inputs=None):
        # Reset BEFORE hooks fire. Any hook that corrupts ref will flip this
        # to False, and the loss reads it to decide whether to push.
        self.pipe._recycle_ref_clean = True
        loss = super().forward(data, inputs=inputs)
        self.pipe._recycle_step += 1
        return loss


# ===========================================================================
# CLI
# ===========================================================================
def parser():
    p = chunk_aware_parser()
    p.add_argument("--recycle_alpha",       type=float, default=0.3,
                   help="Max α applied to buffer-sampled error when corrupting "
                        "reference_latents. 0 disables recycle (falls back to "
                        "behavior of train_chunk_aware.py).")
    p.add_argument("--recycle_clean_prob",  type=float, default=0.1,
                   help="Probability of skipping corruption on a given step "
                        "(keeps some clean reference signal).")
    p.add_argument("--recycle_buffer_size", type=int,   default=500,
                   help="Per-rank FIFO buffer size of residual tensors.")
    p.add_argument("--recycle_warmup",      type=int,   default=200,
                   help="Steps before α leaves 0 (buffer fills first).")
    p.add_argument("--recycle_ramp",        type=int,   default=500,
                   help="Steps over which α linearly ramps 0 → recycle_alpha.")
    p.add_argument("--recycle_push_prob",   type=float, default=0.5,
                   help="Probability of pushing residual to buffer each step. "
                        "Lower = slower buffer turnover, more diversity.")
    p.add_argument("--recycle_collect_start_step", type=int, default=0,
                   help="Don't push to buffer until step >= this. Skips the "
                        "cold-start regime where LoRA predictions are too "
                        "noisy to be useful as buffer entries.")
    # NOTE: --plp is inherited from chunk_aware_parser() (train_chunk_aware.py);
    # do NOT re-add it here or argparse raises "conflicting option string".
    return p


if __name__ == "__main__":
    args = parser().parse_args()

    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(
            find_unused_parameters=args.find_unused_parameters)],
    )
    is_main = accelerator.is_main_process

    dataset = ChunkAwareDataset(
        csv_path=args.dataset_metadata_path,
        height=args.height,
        width=args.width,
        chunk_frames=args.num_frames,
        recent_aug_strength=args.recent_aug_strength,
        dataset_repeat=args.dataset_repeat,
        plp=args.plp,
    )

    # ----- Buffer + α schedule (shared across model + hook) -----
    buffer = ErrorBuffer(max_size=args.recycle_buffer_size)
    alpha_fn = AlphaSchedule(
        max_alpha=args.recycle_alpha,
        warmup=args.recycle_warmup,
        ramp=args.recycle_ramp,
    )

    model = WanRecycleTrainingModule(
        buffer=buffer,
        push_prob=args.recycle_push_prob,
        collect_start_step=args.recycle_collect_start_step,
        step_offset=args.global_step_offset,
        plp=args.plp,
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        audio_processor_path=args.audio_processor_path,
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
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
    )

    # Install EAD first (if enabled), then recycle. Both hooks wrap
    # FunReference.process; installing in this order chains them so the final
    # corruption is recycle(EAD(original_ref)) — Gaussian noise first, then
    # the residual is added on top. This lets the two signals compose:
    # EAD's broad noise prevents collapse, recycle's residual targets real drift.
    if args.ead_corrupt_ratio > 0:
        n = install_ead_hook(model.pipe, args.ead_corrupt_ratio, args.ead_clean_prob)
        if is_main:
            print(f"[EAD] hooked {n} FunReference unit(s) — "
                  f"corrupt_ratio={args.ead_corrupt_ratio}, "
                  f"clean_prob={args.ead_clean_prob}")
    if args.recycle_alpha > 0:
        n = install_recycle_hook(model.pipe, buffer, alpha_fn, args.recycle_clean_prob)
        if is_main:
            print(f"[recycle] hooked {n} FunReference unit(s) — "
                  f"alpha={args.recycle_alpha}, warmup={args.recycle_warmup}, "
                  f"ramp={args.recycle_ramp}, buffer={args.recycle_buffer_size}, "
                  f"clean_prob={args.recycle_clean_prob}, push_prob={args.recycle_push_prob}, "
                  f"collect_start={args.recycle_collect_start_step}, "
                  f"step_offset={args.global_step_offset}")

    if is_main:
        print(f"[recent_aug] strength={args.recent_aug_strength}")

    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        step_offset=args.global_step_offset,
    )

    launch_training_task(accelerator, dataset, model, model_logger, args=args)
