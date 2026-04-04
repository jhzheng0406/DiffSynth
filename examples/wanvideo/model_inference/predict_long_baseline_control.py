"""
Long video generation — pure DiffSynth baseline (no anti-drift, no Helios).

Each chunk is generated independently with only the text prompt and control
video as conditioning.  Reference image is used only for chunk 0.

Use this as the reference baseline to quantify how much drift occurs with
standard chunk-by-chunk generation, and to compare against antidrift / Helios.

Usage:
    python predict_long_baseline_control.py
"""

import os, torch
from PIL import Image
from diffsynth.utils.data import VideoData, save_video
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_DIR        = "models/Wan2.1-Fun-V1.1-1.3B-Control"
USE_MODEL_ID     = True

HEIGHT           = 832
WIDTH            = 480
CHUNK_FRAMES     = 49
NUM_CHUNKS       = 6

CONTROL_VIDEO    = "asset/pose_loop.mp4"
REFERENCE_IMAGE  = None   # PIL.Image or path; used for chunk 0 only

PROMPT = (
    "在这个阳光明媚的户外花园里，美女身穿一袭及膝的白色无袖连衣裙，裙摆在她轻盈的舞姿中轻柔地摆动。阳光透过树叶间洒下斑驳的光影，映衬出她柔和的脸庞和清澈的眼眸，显得格外优雅。"
)
NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)

SEED             = 42
CFG_SCALE        = 6.0
NUM_STEPS        = 50
FPS              = 16
SAVE_PATH        = "samples/long_baseline_control"

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
# Chunk generation loop — no special inter-chunk conditioning
# ---------------------------------------------------------------------------
all_frames = []

for k in range(NUM_CHUNKS):
    is_first_chunk = (k == 0)
    print(f"\n── Chunk {k+1}/{NUM_CHUNKS} ──")

    frame_start = k * (CHUNK_FRAMES - 1)
    ctrl_chunk  = all_ctrl_frames[frame_start : frame_start + CHUNK_FRAMES]

    chunk_frames = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        control_video=ctrl_chunk,
        # Only use reference_image for the very first chunk
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
index = len(os.listdir(SAVE_PATH)) + 1
out_path = os.path.join(SAVE_PATH, f"{index:08d}.mp4")
save_video(all_frames, out_path, fps=FPS, quality=7)
print(f"\nSaved {len(all_frames)} frames → {out_path}")
