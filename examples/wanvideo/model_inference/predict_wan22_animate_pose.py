"""
Wan2.2-Animate-14B pose-driven generation, single 81-frame chunk.

Inputs:
  - input_image:        a PIL image of the character (any image matching prompt)
  - animate_pose_video: skeleton/pose video (DWPose / OpenPose colored skeleton)
  - animate_face_video: optional face video (we leave None — pose-only is fine)

Auto-downloads ~28GB of model weights on first run via modelscope.

Run from DiffSynth-Studio repo root:
    python examples/wanvideo/model_inference/predict_wan22_animate_pose.py
"""
import os
import torch
from datetime import datetime
from PIL import Image

from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.utils.data import VideoData, save_video


# ---------------------------------------------------------------------------
# Config — edit these
# ---------------------------------------------------------------------------
# Pose skeleton video (colored DWPose / OpenPose render) — your existing
# control_video is exactly this format and works as-is.
POSE_VIDEO       = "asset/pose_loop.mp4"

# Character reference image. Set to None to let the model pick from prompt only
# (quality will suffer; recommended to provide a real image).
REFERENCE_IMAGE  = "asset/1.png"

# Generation
PROMPT = (
    "在这个阳光明媚的户外花园里，美女身穿一袭及膝的白色无袖连衣裙，裙摆在她轻盈的舞姿中轻柔地摆动。"
    "阳光透过树叶间洒下斑驳的光影，映衬出她柔和的脸庞和清澈的眼眸，显得格外优雅。"
)
NEGATIVE_PROMPT  = ""

# Wan2.2-Animate native chunk size
NUM_FRAMES       = 81           # max per single forward pass; 5 sec @ 16fps
HEIGHT           = 720
WIDTH            = 1280
NUM_INFERENCE_STEPS = 20
CFG_SCALE        = 1            # Animate is trained for cfg=1
SEED             = 42
FPS              = 15

SAVE_PATH        = "samples/wan22_animate"


# ---------------------------------------------------------------------------
# Load pipeline (modelscope auto-downloads on first run)
# ---------------------------------------------------------------------------
print("Loading Wan2.2-Animate-14B (will auto-download to models/ on first run)...")
pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.2-Animate-14B",
                    origin_file_pattern="diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Wan-AI/Wan2.2-Animate-14B",
                    origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(model_id="Wan-AI/Wan2.2-Animate-14B",
                    origin_file_pattern="Wan2.1_VAE.pth"),
        ModelConfig(model_id="Wan-AI/Wan2.2-Animate-14B",
                    origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
    ],
    tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B",
                                 origin_file_pattern="google/umt5-xxl/"),
)


# ---------------------------------------------------------------------------
# Load inputs
# ---------------------------------------------------------------------------
ref_image = Image.open(REFERENCE_IMAGE).convert("RGB") if REFERENCE_IMAGE else None

# Pose video: load and clip to NUM_FRAMES - 4 (pipeline trims internally)
pose_clip_len = NUM_FRAMES - 4
pose_video = VideoData(POSE_VIDEO, height=HEIGHT, width=WIDTH).raw_data()[:pose_clip_len]

print(f"Pose video frames loaded: {len(pose_video)}")
print(f"Reference image: {REFERENCE_IMAGE if ref_image else '(none)'}")


# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------
print(f"\nGenerating {NUM_FRAMES}-frame video at {WIDTH}x{HEIGHT} ({NUM_INFERENCE_STEPS} steps)...")

video = pipe(
    prompt=PROMPT,
    negative_prompt=NEGATIVE_PROMPT,
    seed=SEED, tiled=True,
    input_image=ref_image,
    animate_pose_video=pose_video,
    animate_face_video=None,            # pose-only mode
    num_frames=NUM_FRAMES,
    height=HEIGHT, width=WIDTH,
    num_inference_steps=NUM_INFERENCE_STEPS,
    cfg_scale=CFG_SCALE,
)


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
os.makedirs(SAVE_PATH, exist_ok=True)
ts = datetime.now().strftime("%m%d_%H%M")
out_path = os.path.join(SAVE_PATH, f"wan22_animate_{NUM_FRAMES}f_{ts}.mp4")
save_video(video, out_path, fps=FPS, quality=5)
print(f"\nSaved → {out_path}")
