"""
Long video chain inference with the 14B sink + recent LoRA.

For every chunk k in the chain:
  reference_image       = chunk_{k-1}'s last frame (or INITIAL_REF for chunk 0)
                          ← "recent"
  sink_reference_image  = INITIAL_REF (constant across all chunks)
                          ← "sink"

Matches the dual-ref training setup in train_chunk_aware.py.

Usage:
    python Wan2.1-Fun-V1.1-14B-Control-WithLoRA-Sink.py
"""

import os, torch
from PIL import Image
from diffsynth.utils.data import VideoData, save_video
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LORA_PATH        = "models/train/Wan2.1-Fun-V1.1-14B-Control_lora_cartoon_sink/step-1647.safetensors"
LORA_ALPHA       = 1.0

HEIGHT           = 832
WIDTH            = 480
CHUNK_FRAMES     = 49
NUM_CHUNKS       = 10

CONTROL_VIDEO    = "asset/pose_loop.mp4"
INITIAL_REF      = "asset/6.png"   # chunk 0 recent + sink for ALL chunks

PROMPT = (
    "动漫风格，紫色短发少女正在轻盈起舞。她头戴黑色发箍，身穿白色连衣裙，外搭黑色背心，胸前系着粉色蝴蝶结。背景是粉色的大圆圈，画面简洁柔和。"
)
NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)

SEED             = 42
FPS              = 16
SAVE_PATH        = "samples/long_chunk_chain_V1.1_14B_cartoon_sink"

# ---------------------------------------------------------------------------
# Load pipeline (Wan2.1-Fun-V1.1-14B-Control)
# ---------------------------------------------------------------------------
pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-14B-Control", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-14B-Control", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-14B-Control", origin_file_pattern="Wan2.1_VAE.pth"),
        ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-14B-Control", origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
    ],
    tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
)

# ---------------------------------------------------------------------------
# Load LoRA on top
# ---------------------------------------------------------------------------
if not os.path.exists(LORA_PATH):
    raise FileNotFoundError(
        f"LoRA checkpoint not found: {LORA_PATH}\n"
        f"Train it first via lora/Wan2.1-Fun-V1.1-14B-Control_cartoon_sink.sh"
    )
print(f"Loading LoRA: {LORA_PATH}  (alpha={LORA_ALPHA})")
pipe.load_lora(pipe.dit, lora_config=LORA_PATH, alpha=LORA_ALPHA)

# ---------------------------------------------------------------------------
# Load full control video once, slice per chunk (1-frame overlap)
# ---------------------------------------------------------------------------
total_frames = CHUNK_FRAMES + (NUM_CHUNKS - 1) * (CHUNK_FRAMES - 1)
full_ctrl = VideoData(CONTROL_VIDEO, height=HEIGHT, width=WIDTH)
all_ctrl_frames = [full_ctrl[i] for i in range(min(len(full_ctrl), total_frames))]
while len(all_ctrl_frames) < total_frames:
    all_ctrl_frames += all_ctrl_frames
all_ctrl_frames = all_ctrl_frames[:total_frames]

sink_img = (Image.open(INITIAL_REF).convert("RGB")
            if isinstance(INITIAL_REF, str) else INITIAL_REF)

# ---------------------------------------------------------------------------
# Chunk generation loop — sink fixed; recent = previous chunk's last frame
# ---------------------------------------------------------------------------
all_frames = []
ref_for_chunk = sink_img    # chunk 0: recent = sink (same as training when s=0)

for k in range(NUM_CHUNKS):
    is_first_chunk = (k == 0)
    print(f"\n── Chunk {k+1}/{NUM_CHUNKS} ── sink=INITIAL_REF, recent={'sink (k=0)' if is_first_chunk else f'chunk{k-1}_last_frame'}")

    frame_start = k * (CHUNK_FRAMES - 1)
    ctrl_chunk  = all_ctrl_frames[frame_start : frame_start + CHUNK_FRAMES]

    chunk_frames = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        control_video=ctrl_chunk,
        reference_image=ref_for_chunk,
        sink_reference_image=sink_img,
        height=HEIGHT,
        width=WIDTH,
        num_frames=CHUNK_FRAMES,
        seed=SEED + k,
        tiled=True,
    )

    if is_first_chunk:
        all_frames.extend(chunk_frames)
    else:
        all_frames.extend(chunk_frames[1:])

    ref_for_chunk = chunk_frames[-1]

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
os.makedirs(SAVE_PATH, exist_ok=True)
lora_tag = os.path.splitext(os.path.basename(LORA_PATH))[0]
sink_tag = (
    f"_sink-{os.path.splitext(os.path.basename(INITIAL_REF))[0]}"
    if isinstance(INITIAL_REF, str) else ""
)
fname = f"V1.1-14B+LoRA-Sink-{lora_tag}{sink_tag}_a{LORA_ALPHA}_chunks{NUM_CHUNKS}x{CHUNK_FRAMES}_{HEIGHT}x{WIDTH}_seed{SEED}_fps{FPS}.mp4"
out_path = os.path.join(SAVE_PATH, fname)
save_video(all_frames, out_path, fps=FPS, quality=5)
print(f"\nSaved {len(all_frames)} frames → {out_path}")
