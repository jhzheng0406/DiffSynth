"""
Long video generation with anti-drift conditioning for Wan2.1-Fun-V1.1-Control.

Anti-drift strategy:
  - Chunk 0: generated with the provided reference_image / input_image.
  - Chunk k > 0: the LAST FRAME of chunk k-1 is used as:
      • reference_image  (CLIP + VAE appearance anchor)
      • input_image      (I2V first-frame conditioning, if the model supports it)
    This anchors each chunk to the content of the previous one, preventing
    long-range appearance drift.

Control video is automatically sliced per chunk with a 1-frame overlap so
chunks tile seamlessly.

Usage:
    python predict_long_antidrift_control.py
"""

import os, torch
from PIL import Image
from diffsynth.utils.data import VideoData, save_video
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig

# ---------------------------------------------------------------------------
# Config — edit these
# ---------------------------------------------------------------------------
MODEL_DIR        = "/mnt/vita/scratch/vita-students/users/jinghao/code/VideoX-Fun/models/Diffusion_Transformer/Wan2.1-Fun-V1.1-1.3B-Control"
USE_MODEL_ID     = False  # set False to load from MODEL_DIR

# Fine-tuned checkpoint to load on top of base weights (None = use base model only)
FINETUNED_CKPT   = None  # fine-tuning degraded quality; base model works better

HEIGHT           = 832
WIDTH            = 480
CHUNK_FRAMES     = 49     # frames per chunk; must satisfy (N-1) % 4 == 0
NUM_CHUNKS       = 6      # total chunks → (49-1)*6+1 = 289 frames ≈ 18 s @ 16 fps

CONTROL_VIDEO    = "asset/pose_loop.mp4"   # full-length control video
REFERENCE_IMAGE  = None                    # PIL.Image or path; used for chunk 0

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
SAVE_PATH        = "samples/long_antidrift_control"

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
    if FINETUNED_CKPT is not None:
        import safetensors.torch
        state_dict = safetensors.torch.load_file(FINETUNED_CKPT, device="cpu")
        missing, unexpected = pipe.dit.load_state_dict(state_dict, strict=False)
        print(f"[ckpt] Loaded {FINETUNED_CKPT}")
        if missing:    print(f"  missing keys : {len(missing)}")
        if unexpected: print(f"  unexpected   : {len(unexpected)}")

# ---------------------------------------------------------------------------
# Load full control video once
# ---------------------------------------------------------------------------
total_frames = CHUNK_FRAMES + (NUM_CHUNKS - 1) * (CHUNK_FRAMES - 1)
full_ctrl = VideoData(CONTROL_VIDEO, height=HEIGHT, width=WIDTH)
all_ctrl_frames = [full_ctrl[i] for i in range(min(len(full_ctrl), total_frames))]
# If control video is shorter than needed, loop it
while len(all_ctrl_frames) < total_frames:
    all_ctrl_frames += all_ctrl_frames
all_ctrl_frames = all_ctrl_frames[:total_frames]

ref_image = (Image.open(REFERENCE_IMAGE).convert("RGB")
             if isinstance(REFERENCE_IMAGE, str) else REFERENCE_IMAGE)

# ---------------------------------------------------------------------------
# Chunk generation loop
# ---------------------------------------------------------------------------
all_frames = []          # list[PIL.Image]
prev_last_frame = None   # anti-drift anchor from previous chunk

for k in range(NUM_CHUNKS):
    is_first_chunk = (k == 0)
    print(f"\n── Chunk {k+1}/{NUM_CHUNKS} ──")

    # Slice control video for this chunk
    frame_start = k * (CHUNK_FRAMES - 1)
    ctrl_chunk  = all_ctrl_frames[frame_start : frame_start + CHUNK_FRAMES]

    # Anti-drift: use last frame of previous chunk as the anchor
    anchor_ref   = ref_image       if is_first_chunk else prev_last_frame
    # Fun-Control V1.1 uses reference_image (CLIP + ref_conv) for appearance continuity.
    # Passing input_image simultaneously causes conflicting conditioning and gray-noise output.
    anchor_input = None

    chunk_frames = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        control_video=ctrl_chunk,
        reference_image=anchor_ref,
        input_image=anchor_input,
        height=HEIGHT,
        width=WIDTH,
        num_frames=CHUNK_FRAMES,
        seed=SEED + k,
        cfg_scale=CFG_SCALE,
        num_inference_steps=NUM_STEPS,
        tiled=True,
    )  # list[PIL.Image], length == CHUNK_FRAMES

    # Skip first frame on chunks k>0 (overlap with last of previous chunk)
    if is_first_chunk:
        all_frames.extend(chunk_frames)
    else:
        all_frames.extend(chunk_frames[1:])

    prev_last_frame = chunk_frames[-1]

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
os.makedirs(SAVE_PATH, exist_ok=True)
index = len(os.listdir(SAVE_PATH)) + 1
out_path = os.path.join(SAVE_PATH, f"{index:08d}.mp4")
save_video(all_frames, out_path, fps=FPS, quality=7)
print(f"\nSaved {len(all_frames)} frames → {out_path}")
