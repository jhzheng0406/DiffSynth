"""
Chunk-aware training for sink + recent ref LoRA.

At dataset __init__ we enumerate every (source_video, chunk_position) pair
from a "long video" metadata.csv (e.g. data/cartoon_15s/metadata.csv) so the
chunk slicing happens at dataset-build time — the same shape as Helios's
pre-sliced sections, but kept in-memory rather than on disk. Adding a new
source video to the CSV automatically expands the chunk list next run.

Each chunk = one training sample:
    target  = video[s : s + chunk_frames]
    control = pose[s : s + chunk_frames]
    sink    = video[0]                        (clean global anchor)
    recent  = video[s-1] (or sink if s=0)     (image-level aug applied)

Pipeline must accept sink_reference_image — see patched
WanVideoUnit_FunReference in wan_video.py and the parse_extra_inputs change
in train.py.

Usage: see lora/Wan2.1-Fun-V1.1-1.3B-Control_cartoon_long.sh
"""
import argparse, os, random, sys
import accelerate
import decord
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance

# Make sibling train.py importable regardless of cwd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from diffsynth.diffusion import ModelLogger, launch_training_task
from train import WanTrainingModule, install_ead_hook, wan_parser as base_wan_parser


# ---------------------------------------------------------------------------
# Image-level augmentation on the recent ref
# ---------------------------------------------------------------------------
def augment_recent(pil_img, strength=0.5):
    """Color jitter + gaussian pixel noise to simulate chain-inference drift.
    `strength` ∈ [0, ~2] scales every perturbation's magnitude.
        - brightness ±(0.20 * strength)
        - contrast   ±(0.20 * strength)
        - saturation ±(0.30 * strength)   (positive bias supports oversaturation)
        - gaussian noise sigma ~ U[2, 8] * strength in 0-255 units
    Each perturbation fires with ~0.8-0.9 probability so the model also sees
    some near-clean recents.
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
# Dataset: enumerates all (video, chunk_idx) pairs at init
# ---------------------------------------------------------------------------
class ChunkAwareDataset(torch.utils.data.Dataset):
    # runner.py probes dataset.load_from_cache; we don't use the cached path
    load_from_cache = False

    def __init__(
        self,
        csv_path,
        height,
        width,
        chunk_frames=49,
        recent_aug_strength=0.5,
        dataset_repeat=1,
    ):
        src_df = pd.read_csv(csv_path)
        self.height = height
        self.width = width
        self.chunk_frames = chunk_frames
        self.recent_aug_strength = recent_aug_strength
        self.dataset_repeat = dataset_repeat

        # ---- enumerate (source_video, chunk_idx) pairs ----
        # 1-frame overlap convention: chunk k spans [k*stride, k*stride+chunk_frames)
        self.samples = []
        stride = chunk_frames - 1
        skipped = 0
        for _, row in src_df.iterrows():
            n_frames = int(row.get("num_frames", 0))
            if n_frames <= 0:
                # fall back to probing the video (slower init but works without num_frames col)
                n_frames = int(decord.VideoReader(row["video"]).count_frames())
            n_chunks = max(0, (n_frames - chunk_frames) // stride + 1)
            if n_chunks == 0:
                skipped += 1
                continue
            for k in range(n_chunks):
                self.samples.append({
                    "source_video": row["video"],
                    "source_pose":  row["control_video"],
                    "chunk_start":  k * stride,
                    "prompt":       row["prompt"],
                })

        print(f"[ChunkAwareDataset] {len(src_df)} source videos → {len(self.samples)} chunk samples"
              f"  (avg {len(self.samples) / max(1, len(src_df)):.1f}/video,  skipped {skipped} too-short)")

    def __len__(self):
        return len(self.samples) * self.dataset_repeat

    def _to_pil(self, frame_np):
        img = Image.fromarray(frame_np).convert("RGB")
        return img.resize((self.width, self.height))

    def __getitem__(self, idx):
        sample = self.samples[idx % len(self.samples)]
        s = sample["chunk_start"]

        vr_vid  = decord.VideoReader(sample["source_video"])
        vr_pose = decord.VideoReader(sample["source_pose"])

        # target + pose chunk
        chunk_idx = np.arange(s, s + self.chunk_frames)
        target_np = vr_vid.get_batch(chunk_idx).asnumpy()    # [T, H, W, 3] uint8
        pose_np   = vr_pose.get_batch(chunk_idx).asnumpy()

        # sink = frame 0 (clean, no aug)
        sink_np  = vr_vid[0].asnumpy()
        sink_pil = self._to_pil(sink_np)

        # recent = frame at target[0]'s position. With 1-frame overlap chunk
        # convention (stride = chunk_frames - 1 = 48), this position equals
        # chunk_{k-1}'s last frame at chain inference time — so training and
        # inference recent are at the SAME position. For s=0 there's no prior
        # chunk, fall back to sink.
        if s == 0:
            recent_pil_clean = sink_pil
            recent_pil = sink_pil
        else:
            recent_np = vr_vid[s].asnumpy()
            recent_pil_clean = self._to_pil(recent_np)
            recent_pil = augment_recent(recent_pil_clean, strength=self.recent_aug_strength)

        target_pil = [self._to_pil(f) for f in target_np]
        pose_pil   = [self._to_pil(f) for f in pose_np]

        return {
            "video":                 target_pil,
            "control_video":         pose_pil,
            "reference_image":       [recent_pil],         # augmented (per recent_aug_strength)
            "reference_image_clean": [recent_pil_clean],   # un-augmented; DMD asymmetric uses this for teacher
            "sink_reference_image":  [sink_pil],
            "prompt":                sample["prompt"],
        }


# ---------------------------------------------------------------------------
# CLI parser: regular train.py parser + recent_aug_strength
# ---------------------------------------------------------------------------
def wan_parser():
    parser = base_wan_parser()
    parser.add_argument(
        "--recent_aug_strength", type=float, default=0.5,
        help="Image-level augmentation strength on recent ref. 0 = no aug. "
             "0.5 mild (brightness ±10%, sat ±15%, noise σ~2-4); "
             "1.0 strong (brightness ±20%, sat ±30%, noise σ~2-8); "
             ">1.0 simulates more severe drift.",
    )
    return parser


if __name__ == "__main__":
    args = wan_parser().parse_args()
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )

    dataset = ChunkAwareDataset(
        csv_path=args.dataset_metadata_path,
        height=args.height,
        width=args.width,
        chunk_frames=args.num_frames,
        recent_aug_strength=args.recent_aug_strength,
        dataset_repeat=args.dataset_repeat,
    )

    model = WanTrainingModule(
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

    # Optional latent-space EAD on TOP of image-level aug (off by default —
    # the image aug is the primary noise source here).
    if args.ead_corrupt_ratio > 0:
        n = install_ead_hook(model.pipe, args.ead_corrupt_ratio, args.ead_clean_prob)
        print(f"[EAD] hooked {n} FunReference unit(s) — corrupt_ratio={args.ead_corrupt_ratio}")

    print(f"[recent_aug] strength={args.recent_aug_strength}")

    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        step_offset=args.global_step_offset,
    )

    launch_training_task(accelerator, dataset, model, model_logger, args=args)
