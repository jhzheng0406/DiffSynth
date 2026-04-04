"""
Simple anti-drift training for WanModel (DiffSynth-Studio).

Trains the model to continue video generation given a reference frame (the
last frame of the previous chunk).  No architecture changes are needed —
this uses the existing `ref_conv` / `reference_latents` conditioning path
that is already present in Fun-Control V1.1 (and any model built with
`has_ref_conv=True`).

Training setup
--------------
From a latent clip [B, C, T, H, W]:
  - Split at frame ``split_t = antidrift_history_frames``.
  - reference_latents = clean latent at frame ``split_t - 1``.
  - Noisy latents cover the full sequence [0, T).
  - Supervision covers only the target portion [split_t, T) by default.
  - With probability ``antidrift_drop_ref_ratio``, the reference is dropped
    (→ None), simulating first-chunk generation with no prior context.
  - Optional Gaussian noise added to the reference latent, teaching the model
    to be robust to imperfect prior-chunk outputs (EAD-style).

Typical usage (Fun-Control model):
    python train_antidrift.py \\
        --model_id_with_origin_paths "PAI/Wan2.1-Fun-V1.1-1.3B-Control:diffusion_pytorch_model*.safetensors,PAI/Wan2.1-Fun-V1.1-1.3B-Control:models_t5_umt5-xxl-enc-bf16.pth,PAI/Wan2.1-Fun-V1.1-1.3B-Control:Wan2.1_VAE.pth,PAI/Wan2.1-Fun-V1.1-1.3B-Control:models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \\
        --dataset_base_path data/humanvid-subset1 \\
        --dataset_metadata_path data/humanvid-subset1/metadata.csv \\
        --data_file_keys video,text,control_video \\
        --extra_inputs control_video \\
        --trainable_models dit \\
        --output_path models/antidrift-control \\
        --num_frames 65 --height 480 --width 832 \\
        --gradient_accumulation_steps 4 \\
        --learning_rate 2e-5 --num_epochs 100 --save_steps 200 \\
        --antidrift_history_frames 4 \\
        --antidrift_drop_ref_ratio 0.10 \\
        --antidrift_ref_noise_ratio 0.02
"""

import os
import sys

import accelerate
import torch

sys.path.insert(0, os.path.dirname(__file__))

from train import WanTrainingModule, wan_parser as _base_wan_parser

from diffsynth.core import UnifiedDataset
from diffsynth.core.data.operators import ImageCropAndResize, LoadAudio, LoadVideo, ToAbsolutePath
from diffsynth.diffusion import *


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def AntidriftFlowMatchSFTLoss(
    pipe,
    antidrift_history_frames=13,
    antidrift_drop_ref_ratio=0.10,
    antidrift_ref_noise_ratio=0.02,
    antidrift_supervise_history=False,
    **inputs,
):
    """
    Flow-matching SFT loss with simple anti-drift reference conditioning.

    The last frame of the history portion is passed as ``reference_latents``
    so the model learns to condition its output on the appearance of the
    previous chunk's final frame.

    Parameters
    ----------
    antidrift_history_frames : int
        Number of latent frames reserved as "history".  The reference is
        taken from ``input_latents[:, :, split_t-1]``.  Must be ≥ 1.
    antidrift_drop_ref_ratio : float
        Probability of dropping the reference entirely (→ None) per training
        step, which teaches the model to also work in the no-prior-context
        (first-chunk) case.
    antidrift_ref_noise_ratio : float
        Std of Gaussian noise added to the reference latent before passing it
        to the model.  Simulates imperfect prior-chunk reconstruction (EAD).
        Set to 0 to disable.
    antidrift_supervise_history : bool
        If True, supervise all T latent frames.  If False (default), only
        supervise frames ≥ ``antidrift_history_frames``.
    """
    input_latents = inputs["input_latents"]  # [B, C, T, H, W]

    split_t = antidrift_history_frames
    if input_latents.shape[2] <= split_t:
        raise ValueError(
            f"antidrift_history_frames={split_t} but latent sequence has only "
            f"{input_latents.shape[2]} frames.  Reduce --antidrift_history_frames "
            "or increase --num_frames."
        )

    # ── Build reference latent ────────────────────────────────────────────
    # Take the last frame of the history portion as a clean reference.
    # ref_conv in the model expects [B, C, H, W] or [B, C, 1, H, W].
    ref_latent = input_latents[:, :, split_t - 1 : split_t].clone()  # [B, C, 1, H, W]

    # Optional: add light Gaussian noise for EAD-style robustness training
    if antidrift_ref_noise_ratio > 0:
        ref_latent = ref_latent + antidrift_ref_noise_ratio * torch.randn_like(ref_latent)

    # Optional: randomly drop the reference (simulate first-chunk scenario)
    if antidrift_drop_ref_ratio > 0 and torch.rand(()).item() < antidrift_drop_ref_ratio:
        ref_latent = None

    # ── Flow-matching noise ───────────────────────────────────────────────
    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)

    noise = torch.randn_like(input_latents)
    latents = pipe.scheduler.add_noise(input_latents, noise, timestep)
    training_target = pipe.scheduler.training_target(input_latents, noise, timestep)

    # ── Assemble model inputs ─────────────────────────────────────────────
    model_inputs = dict(inputs)
    model_inputs["latents"] = latents
    # Inject our reference into the model input dict; model_fn_wan_video will
    # pick it up and pass it through dit.ref_conv automatically.
    if ref_latent is not None:
        model_inputs["reference_latents"] = ref_latent.to(dtype=pipe.torch_dtype, device=pipe.device)
    else:
        # Remove any reference_latents that might have been pre-encoded by the
        # pipeline's ReferenceImageEncoder unit.
        model_inputs.pop("reference_latents", None)

    if "first_frame_latents" in model_inputs:
        model_inputs["latents"][:, :, 0:1] = model_inputs["first_frame_latents"]

    # ── Forward pass ─────────────────────────────────────────────────────
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred = pipe.model_fn(**models, **model_inputs, timestep=timestep)

    # ── Supervision window ────────────────────────────────────────────────
    supervise_from = 0 if antidrift_supervise_history else split_t
    if "first_frame_latents" in model_inputs:
        supervise_from = max(supervise_from, 1)

    noise_pred = noise_pred[:, :, supervise_from:]
    training_target = training_target[:, :, supervise_from:]

    loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
    loss = loss * pipe.scheduler.training_weight(timestep)
    return loss


class AntidriftWanTrainingModule(WanTrainingModule):
    def __init__(
        self,
        antidrift_history_frames,
        antidrift_drop_ref_ratio,
        antidrift_ref_noise_ratio,
        antidrift_supervise_history,
        **kwargs,
    ):
        super().__init__(**kwargs)

        loss_kwargs = dict(
            antidrift_history_frames=antidrift_history_frames,
            antidrift_drop_ref_ratio=antidrift_drop_ref_ratio,
            antidrift_ref_noise_ratio=antidrift_ref_noise_ratio,
            antidrift_supervise_history=antidrift_supervise_history,
        )
        self.task_to_loss["sft"] = (
            lambda pipe, inputs_shared, inputs_posi, inputs_nega: AntidriftFlowMatchSFTLoss(
                pipe, **loss_kwargs, **inputs_shared, **inputs_posi
            )
        )
        self.task_to_loss["sft:train"] = self.task_to_loss["sft"]


def wan_parser():
    parser = _base_wan_parser()
    parser.add_argument(
        "--antidrift_history_frames",
        type=int,
        default=1,
        help=(
            "Number of latent frames to treat as 'history'. "
            "The reference latent is taken from frame index (history_frames - 1). "
            "Default=1: use the very first latent frame as the reference and supervise "
            "frames 1..T-1, which exactly mirrors inference where the last frame of the "
            "previous chunk is passed as reference_latents via ref_conv."
        ),
    )
    parser.add_argument(
        "--antidrift_drop_ref_ratio",
        type=float,
        default=0.10,
        help="Probability of dropping the reference latent per step (simulates first-chunk).",
    )
    parser.add_argument(
        "--antidrift_ref_noise_ratio",
        type=float,
        default=0.02,
        help=(
            "Std of Gaussian noise added to the reference latent. "
            "Improves robustness to imperfect prior-chunk outputs (EAD). "
            "Set to 0 to disable."
        ),
    )
    parser.add_argument(
        "--antidrift_supervise_history",
        default=False,
        action="store_true",
        help="Supervise all latent frames; by default only frames >= history_frames are supervised.",
    )
    return parser


if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()

    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[
            accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)
        ],
    )

    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=UnifiedDataset.default_video_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
            num_frames=args.num_frames,
            time_division_factor=4,
            time_division_remainder=1,
        ),
        special_operator_map={
            "animate_face_video": (
                ToAbsolutePath(args.dataset_base_path)
                >> LoadVideo(
                    args.num_frames,
                    4,
                    1,
                    frame_processor=ImageCropAndResize(512, 512, None, 16, 16),
                )
            ),
            "input_audio": ToAbsolutePath(args.dataset_base_path) >> LoadAudio(sr=16000),
        },
    )

    model = AntidriftWanTrainingModule(
        antidrift_history_frames=args.antidrift_history_frames,
        antidrift_drop_ref_ratio=args.antidrift_drop_ref_ratio,
        antidrift_ref_noise_ratio=args.antidrift_ref_noise_ratio,
        antidrift_supervise_history=args.antidrift_supervise_history,
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

    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
    )
    launcher_map = {
        "sft": launch_training_task,
        "sft:train": launch_training_task,
    }
    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)
