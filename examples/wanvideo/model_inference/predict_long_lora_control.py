"""
Long video generation — DiffSynth baseline + LoRA only.

This script is intentionally the same as predict_long_baseline_control.py
except that it loads one LoRA into pipe.dit after the base model is loaded.
Use it to check whether the LoRA itself improves or degrades the baseline
before adding any Helios/history logic.

Usage:
    python examples/wanvideo/model_inference/predict_long_lora_control.py

Optional overrides:
    LORA_PATH=models/train/.../epoch-0.safetensors
    LORA_ALPHA=1.0
    NUM_CHUNKS=10
"""

import os
import re
from datetime import datetime

import torch
from PIL import Image

from diffsynth.utils.data import VideoData, save_video
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig


def _safe_filename_tag(text):
    text = str(text).strip()
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return text.strip("._-") or "unknown"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_DIR        = "models/Wan2.1-Fun-V1.1-1.3B-Control"
USE_MODEL_ID     = True

LORA_PATH        = os.environ.get(
    "LORA_PATH",
    "models/train/Wan2.1-Fun-V1.1-1.3B-Control_lora_humanvid_smoke/epoch-0.safetensors",
)
LORA_ALPHA       = float(os.environ.get("LORA_ALPHA", "1.0"))

HEIGHT           = 480
WIDTH            = 832
CHUNK_FRAMES     = 49
NUM_CHUNKS       = int(os.environ.get("NUM_CHUNKS", "10"))

CONTROL_VIDEO    = "asset/pose_loop.mp4"
REFERENCE_IMAGE  = None   # PIL.Image or path; used for chunk 0 only

PROMPT = (
    "动漫风格，紫色短发少女正在轻盈起舞。她头戴黑色发箍，身穿白色连衣裙，外搭黑色背心，胸前系着粉色蝴蝶结。背景是粉色的大圆圈，画面简洁柔和。"
)
NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)

SEED             = 42
CFG_SCALE        = 6.0
NUM_STEPS        = 50
FPS              = 16
SAVE_PATH        = "samples/long_lora_control"


if not LORA_PATH:
    raise ValueError("LORA_PATH is empty. Set LORA_PATH to the LoRA checkpoint you want to validate.")
if not os.path.exists(LORA_PATH):
    raise FileNotFoundError(f"LoRA checkpoint not found: {LORA_PATH}")


# ---------------------------------------------------------------------------
# Load pipeline
# ---------------------------------------------------------------------------
if USE_MODEL_ID:
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-1.3B-Control",
                        origin_file_pattern="diffusion_pytorch_model*.safetensors"),
            ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-1.3B-Control",
                        origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
            ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-1.3B-Control",
                        origin_file_pattern="Wan2.1_VAE.pth"),
            ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-1.3B-Control",
                        origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
        ],
        tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B",
                                     origin_file_pattern="google/umt5-xxl/"),
    )
else:
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(os.path.join(MODEL_DIR, "diffusion_pytorch_model.safetensors")),
            ModelConfig(os.path.join(MODEL_DIR, "models_t5_umt5-xxl-enc-bf16.pth")),
            ModelConfig(os.path.join(MODEL_DIR, "Wan2.1_VAE.pth")),
            ModelConfig(os.path.join(MODEL_DIR,
                        "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth")),
        ],
        tokenizer_config=ModelConfig(os.path.join(MODEL_DIR, "google/umt5-xxl")),
    )

pipe.load_lora(pipe.dit, LORA_PATH, alpha=LORA_ALPHA)
print(f"Loaded LoRA: {LORA_PATH}  alpha={LORA_ALPHA}")


# ---------------------------------------------------------------------------
# Load full control video once
# ---------------------------------------------------------------------------
total_frames = CHUNK_FRAMES + (NUM_CHUNKS - 1) * (CHUNK_FRAMES - 1)
full_ctrl = VideoData(CONTROL_VIDEO, height=HEIGHT, width=WIDTH)
all_ctrl_frames = [full_ctrl[i] for i in range(min(len(full_ctrl), total_frames))]
while len(all_ctrl_frames) < total_frames:
    all_ctrl_frames += all_ctrl_frames
all_ctrl_frames = all_ctrl_frames[:total_frames]

ref_image = (Image.open(REFERENCE_IMAGE).convert("RGB")
             if isinstance(REFERENCE_IMAGE, str) else REFERENCE_IMAGE)


# ---------------------------------------------------------------------------
# Chunk generation loop — LoRA only, no inter-chunk conditioning
# ---------------------------------------------------------------------------
all_frames = []

for k in range(NUM_CHUNKS):
    is_first_chunk = (k == 0)
    print(f"\n-- Chunk {k+1}/{NUM_CHUNKS} --")

    frame_start = k * (CHUNK_FRAMES - 1)
    ctrl_chunk = all_ctrl_frames[frame_start : frame_start + CHUNK_FRAMES]

    chunk_frames = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        control_video=ctrl_chunk,
        reference_image=ref_image if is_first_chunk else None,
        height=HEIGHT,
        width=WIDTH,
        num_frames=CHUNK_FRAMES,
        seed=SEED + k,
        cfg_scale=CFG_SCALE,
        num_inference_steps=NUM_STEPS,
        tiled=True,
    )

    if is_first_chunk:
        all_frames.extend(chunk_frames)
    else:
        all_frames.extend(chunk_frames[1:])


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
os.makedirs(SAVE_PATH, exist_ok=True)
lora_tag = _safe_filename_tag(os.path.splitext(os.path.basename(LORA_PATH))[0])
alpha_tag = _safe_filename_tag(f"{LORA_ALPHA:g}")
timestamp = datetime.now().strftime("%m%d_%H%M")
out_path = os.path.join(SAVE_PATH, f"lora_{lora_tag}_alpha-{alpha_tag}_{timestamp}.mp4")
save_video(all_frames, out_path, fps=FPS, quality=7)
print(f"\nSaved {len(all_frames)} frames -> {out_path}")
