"""
Helios training for WanModel (DiffSynth-Studio).

Extends the standard WanTrainingModule to:
  1. Apply `patch_wan_model_for_helios(pipe.dit)` after model loading.
  2. Build Helios long/mid/short history tiers from the latent video clip.
  3. Inject the history tiers into every Wan self-attention block through the
     Helios side channel during SFT training.
  4. Supervise only the post-history part of the clip by default, which more
     closely matches long-video chunk generation.

Typical usage (SFT fine-tuning, Fun-Control model):
    python train_helios.py \
        --model_id_with_origin_paths "PAI/Wan2.1-Fun-V1.1-1.3B-Control:diffusion_pytorch_model*.safetensors,PAI/Wan2.1-Fun-V1.1-1.3B-Control:models_t5_umt5-xxl-enc-bf16.pth,PAI/Wan2.1-Fun-V1.1-1.3B-Control:Wan2.1_VAE.pth,PAI/Wan2.1-Fun-V1.1-1.3B-Control:models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
        --dataset_base_path data/humanvid-subset1 \
        --dataset_metadata_path data/humanvid-subset1/metadata.csv \
        --data_file_keys video,text,control_video \
        --extra_inputs control_video \
        --trainable_models dit \
        --output_path models/helios-control \
        --num_frames 49 --height 480 --width 832 \
        --batch_size 1 --gradient_accumulation_steps 4 \
        --learning_rate 1e-5 --num_steps 5000 \
        --helios_history_sizes 4,2,1 \
        --helios_init_scale_logit -3.0
"""

import argparse
import os
import sys

import accelerate
import torch

# Allow importing WanTrainingModule from the sibling train.py
sys.path.insert(0, os.path.dirname(__file__))

from train import WanTrainingModule, wan_parser as _base_wan_parser

from diffsynth.core import UnifiedDataset
from diffsynth.core.data.operators import ImageCropAndResize, LoadAudio, LoadVideo, ToAbsolutePath
from diffsynth.diffusion import *
from diffsynth.models.wan_video_helios_attention import (
    add_saturation_to_history_latents,
    corrupt_history_latents,
    patch_wan_model_for_helios,
    split_into_helios_history_and_target,
)


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def HeliosFlowMatchSFTLoss(
    pipe,
    helios_history_sizes,
    helios_corrupt_mode="random",
    helios_noise_ratio_short=1.0 / 3.0,
    helios_noise_ratio_mid=1.0 / 3.0,
    helios_noise_ratio_long=1.0 / 3.0,
    helios_keep_x0=False,
    helios_apply_saturation=True,
    helios_saturation_min=0.3,
    helios_saturation_max=1.7,
    helios_clean_history_prob=0.1,
    helios_drop_t2v_ratio=0.10,
    helios_drop_i2v_ratio=0.05,
    helios_supervise_history=False,
    helios_use_first_frame_anchor=True,
    **inputs,
):
    """
    Flow-matching SFT with Helios history injection.

    We keep the original Wan conditioning tensors unchanged, derive Helios
    history from the first latent frames, and supervise only the target chunk.
    With first-frame anchor enabled, the very first latent serves as a
    permanent visual anchor (matching Helios upstream); the target window
    starts at position 1 + sum(history_sizes).
    """
    input_latents = inputs["input_latents"]
    total_history = sum(helios_history_sizes) + (1 if helios_use_first_frame_anchor else 0)
    if input_latents.shape[2] <= total_history:
        raise ValueError(
            "Helios training needs more latent frames than history (incl. anchor). "
            f"Got latent length={input_latents.shape[2]}, total history (incl. anchor)={total_history}. "
            "Reduce --helios_history_sizes or increase --num_frames."
        )

    target_lat, lat_long, lat_mid, lat_short, fids_long, fids_mid, fids_short, _ = split_into_helios_history_and_target(
        input_latents,
        history_sizes=helios_history_sizes,
        latent_window_size=input_latents.shape[2] - total_history,
        is_random_drop=True,
        drop_t2v_ratio=helios_drop_t2v_ratio,
        drop_i2v_ratio=helios_drop_i2v_ratio,
        use_first_frame_anchor=helios_use_first_frame_anchor,
    )

    if helios_corrupt_mode != "none":
        lat_short, lat_mid, lat_long = corrupt_history_latents(
            lat_short,
            lat_mid,
            lat_long,
            noise_ratio_short=helios_noise_ratio_short,
            noise_ratio_mid=helios_noise_ratio_mid,
            noise_ratio_long=helios_noise_ratio_long,
            corrupt_mode=helios_corrupt_mode,
            is_keep_x0=helios_keep_x0,
        )

    if helios_apply_saturation:
        lat_short, lat_mid, lat_long = add_saturation_to_history_latents(
            lat_short,
            lat_mid,
            lat_long,
            sat_min=helios_saturation_min,
            sat_max=helios_saturation_max,
            clean_prob=helios_clean_history_prob,
        )

    pipe.dit.set_helios_history(lat_long, lat_mid, lat_short, fids_long, fids_mid, fids_short)
    try:
        max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
        min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

        timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
        timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)

        # Only add noise to / supervise the TARGET frames (not the history).
        # This forces the model to use Helios history tokens for temporal context
        # instead of shortcutting through the history frames in the current sequence.
        # This matches inference, where only the new chunk frames are in the current
        # token sequence and cross-chunk history arrives exclusively via Helios tokens.
        noise = torch.randn_like(target_lat)
        latents = pipe.scheduler.add_noise(target_lat, noise, timestep)
        training_target = pipe.scheduler.training_target(target_lat, noise, timestep)

        model_inputs = dict(inputs)
        model_inputs["latents"] = latents
        # Slice any temporal tensors in model_inputs to target frames only.
        for key in ("y", "image_latents", "control_latents"):
            if key in model_inputs and model_inputs[key] is not None:
                t = model_inputs[key]
                if t.dim() == 5 and t.shape[2] == input_latents.shape[2]:
                    model_inputs[key] = t[:, :, total_history:]
        # first_frame_latents conditioning does not apply when we pass only target frames.
        model_inputs.pop("first_frame_latents", None)

        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        noise_pred = pipe.model_fn(**models, **model_inputs, timestep=timestep)

        # All target frames are supervised (no skip needed since history frames are absent).
        supervise_from = 0
        noise_pred = noise_pred[:, :, supervise_from:]
        training_target = training_target[:, :, supervise_from:]

        loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
        loss = loss * pipe.scheduler.training_weight(timestep)
    finally:
        pipe.dit.clear_helios_history()
    return loss


class HeliosWanTrainingModule(WanTrainingModule):
    def __init__(
        self,
        helios_history_sizes,
        helios_init_scale_logit,
        helios_corrupt_mode,
        helios_noise_ratio_short,
        helios_noise_ratio_mid,
        helios_noise_ratio_long,
        helios_keep_x0,
        helios_apply_saturation,
        helios_saturation_min,
        helios_saturation_max,
        helios_clean_history_prob,
        helios_drop_t2v_ratio,
        helios_drop_i2v_ratio,
        helios_supervise_history,
        helios_use_first_frame_anchor=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.helios_history_sizes = list(helios_history_sizes)

        patch_wan_model_for_helios(
            self.pipe.dit,
            history_sizes=self.helios_history_sizes,
            is_amplify_history=True,
            freeze_history_scale=False,
            init_scale_logit=helios_init_scale_logit,
            use_first_frame_anchor=helios_use_first_frame_anchor,
        )
        print(f"[Helios] patch applied to dit. first_frame_anchor={helios_use_first_frame_anchor}")

        loss_kwargs = dict(
            helios_history_sizes=self.helios_history_sizes,
            helios_corrupt_mode=helios_corrupt_mode,
            helios_noise_ratio_short=helios_noise_ratio_short,
            helios_noise_ratio_mid=helios_noise_ratio_mid,
            helios_noise_ratio_long=helios_noise_ratio_long,
            helios_keep_x0=helios_keep_x0,
            helios_apply_saturation=helios_apply_saturation,
            helios_saturation_min=helios_saturation_min,
            helios_saturation_max=helios_saturation_max,
            helios_clean_history_prob=helios_clean_history_prob,
            helios_drop_t2v_ratio=helios_drop_t2v_ratio,
            helios_drop_i2v_ratio=helios_drop_i2v_ratio,
            helios_supervise_history=helios_supervise_history,
            helios_use_first_frame_anchor=helios_use_first_frame_anchor,
        )
        self.task_to_loss["sft"] = (
            lambda pipe, inputs_shared, inputs_posi, inputs_nega: HeliosFlowMatchSFTLoss(
                pipe, **loss_kwargs, **inputs_shared, **inputs_posi
            )
        )
        self.task_to_loss["sft:train"] = self.task_to_loss["sft"]


def wan_parser():
    parser = _base_wan_parser()
    parser.add_argument(
        "--helios_history_sizes",
        type=str,
        default="4,2,1",
        help="Comma-separated latent history tiers ordered as 'long,mid,short'.",
    )
    parser.add_argument(
        "--helios_init_scale_logit",
        type=float,
        default=-3.0,
        help="Initial value of history_key_scale logit (sigmoid(-3)≈0.05).",
    )
    parser.add_argument(
        "--helios_corrupt_mode",
        type=str,
        default="random",
        choices=["none", "noise", "downsample", "random"],
        help="Easy-Anti-Drifting corruption mode for history latents.",
    )
    parser.add_argument(
        "--helios_noise_ratio_short",
        type=float,
        default=1.0 / 3.0,
        help="History corruption strength for the short tier (upper bound for σ).",
    )
    parser.add_argument(
        "--helios_noise_ratio_mid",
        type=float,
        default=1.0 / 3.0,
        help="History corruption strength for the mid tier (upper bound for σ).",
    )
    parser.add_argument(
        "--helios_noise_ratio_long",
        type=float,
        default=1.0 / 3.0,
        help="History corruption strength for the long tier (upper bound for σ).",
    )
    parser.add_argument(
        "--helios_keep_x0",
        default=False,
        action="store_true",
        help="Keep the oldest long-tier history frame clean during corruption.",
    )
    parser.add_argument(
        "--helios_apply_saturation",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Apply saturation-drift augmentation to history latents (use --no-helios_apply_saturation to disable).",
    )
    parser.add_argument(
        "--helios_saturation_min",
        type=float,
        default=0.3,
        help="Lower bound for saturation augmentation.",
    )
    parser.add_argument(
        "--helios_saturation_max",
        type=float,
        default=1.7,
        help="Upper bound for saturation augmentation.",
    )
    parser.add_argument(
        "--helios_clean_history_prob",
        type=float,
        default=0.1,
        help="Probability of leaving a history frame unchanged during saturation augmentation.",
    )
    parser.add_argument(
        "--helios_drop_t2v_ratio",
        type=float,
        default=0.10,
        help="Probability of dropping all history, simulating T2V training.",
    )
    parser.add_argument(
        "--helios_drop_i2v_ratio",
        type=float,
        default=0.05,
        help="Probability of keeping only the oldest history frame, simulating I2V training.",
    )
    parser.add_argument(
        "--helios_supervise_history",
        default=False,
        action="store_true",
        help="Also supervise the history prefix instead of only the post-history region.",
    )
    parser.add_argument(
        "--helios_use_first_frame_anchor",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Reserve position 0 for the very first VAE latent as a permanent visual anchor "
             "(combats colour drift). Use --no-helios_use_first_frame_anchor to disable.",
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

    model = HeliosWanTrainingModule(
        helios_history_sizes=[int(x) for x in args.helios_history_sizes.split(",")],
        helios_init_scale_logit=args.helios_init_scale_logit,
        helios_corrupt_mode=args.helios_corrupt_mode,
        helios_noise_ratio_short=args.helios_noise_ratio_short,
        helios_noise_ratio_mid=args.helios_noise_ratio_mid,
        helios_noise_ratio_long=args.helios_noise_ratio_long,
        helios_keep_x0=args.helios_keep_x0,
        helios_apply_saturation=args.helios_apply_saturation,
        helios_saturation_min=args.helios_saturation_min,
        helios_saturation_max=args.helios_saturation_max,
        helios_clean_history_prob=args.helios_clean_history_prob,
        helios_drop_t2v_ratio=args.helios_drop_t2v_ratio,
        helios_drop_i2v_ratio=args.helios_drop_i2v_ratio,
        helios_supervise_history=args.helios_supervise_history,
        helios_use_first_frame_anchor=args.helios_use_first_frame_anchor,
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
