"""
Long video generation with Helios-style multi-tier history token compression
for Wan2.1-Fun-V1.1-Control (DiffSynth-Studio).

Differences from predict_long_antidrift_control.py:
  - After loading the pipeline, calls patch_wan_model_for_helios(pipe.dit)
    so every Wan self-attention block can prepend and attend to Helios
    multi-tier history tokens.
  - Before each chunk k > 0, VAE-encodes the output of the previous chunk and
    builds 3-tier history latents, then calls pipe.dit.set_helios_history()
    so every attention layer can attend to compressed history tokens
    alongside the current chunk tokens.
  - No reference_image / input_image conditioning for chunks k > 0 (history
    is injected at the attention level instead).

Usage:
    python predict_long_helios_control.py
"""

import os, sys, torch
import numpy as np
from PIL import Image
from diffsynth.core import load_state_dict
from diffsynth.utils.data import VideoData, save_video
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.models.wan_video_helios_attention import (
    patch_wan_model_for_helios,
    prepare_helios_history,
)

# ---------------------------------------------------------------------------
# Config — edit these
# ---------------------------------------------------------------------------
MODEL_DIR        = "/mnt/vita/scratch/vita-students/users/jinghao/code/VideoX-Fun/models/Diffusion_Transformer/Wan2.1-Fun-V1.1-1.3B-Control"
USE_MODEL_ID     = False
HELIOS_CHECKPOINT = "models/train/Wan2.1-Fun-V1.1-1.3B-Control-Helios/step-2300.safetensors"

HEIGHT           = 832
WIDTH            = 480
CHUNK_FRAMES     = 49
NUM_CHUNKS       = 6

CONTROL_VIDEO    = "asset/pose_loop.mp4"
REFERENCE_IMAGE  = None   # PIL.Image or path; used for chunk 0 only

# Helios history config (3 tiers ordered as long / mid / short)
HISTORY_SIZES    = [4, 2, 1]

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
SAVE_PATH        = "samples/long_helios_control"

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
# Apply Helios patch (AFTER loading weights)
# ---------------------------------------------------------------------------
patch_wan_model_for_helios(
    pipe.dit,
    is_amplify_history=True,
    freeze_history_scale=True,   # True at inference; False when training
)
print("Helios attention patch applied.")
if HELIOS_CHECKPOINT is not None:
    state_dict = load_state_dict(HELIOS_CHECKPOINT, torch_dtype=pipe.torch_dtype)
    missing, unexpected = pipe.dit.load_state_dict(state_dict, strict=False)
    print(f"Loaded Helios checkpoint: {HELIOS_CHECKPOINT}")
    if missing:
        print(f"  Missing keys: {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")

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
# VAE encode helper
# ---------------------------------------------------------------------------
@torch.no_grad()
def encode_chunk_to_latents(frames):
    """
    frames: list[PIL.Image], length T
    Returns: [1, C, T_lat, H_lat, W_lat] on CPU
    """
    pipe.load_models_to_device(['vae'])
    video_tensor = pipe.preprocess_video(frames, device=pipe.device)  # [1, C, T, H, W] in [-1,1]
    latents = pipe.vae.encode(video_tensor, device=pipe.device, tiled=True)
    return latents.cpu()


# ---------------------------------------------------------------------------
# Chunk generation loop
# ---------------------------------------------------------------------------
all_frames        = []           # accumulated output frames (PIL.Image)
accumulated_lats  = []           # list of [1, C, T_lat, H_lat, W_lat] CPU tensors
prev_last_frame   = None         # antidrift anchor

for k in range(NUM_CHUNKS):
    is_first_chunk = (k == 0)
    print(f"\n── Chunk {k+1}/{NUM_CHUNKS} ──")

    # Slice control video for this chunk
    frame_start = k * (CHUNK_FRAMES - 1)
    ctrl_chunk  = all_ctrl_frames[frame_start : frame_start + CHUNK_FRAMES]

    # ── Helios history injection ──────────────────────────────────────
    if not is_first_chunk and len(accumulated_lats) > 0:
        history = prepare_helios_history(
            accumulated_lats,
            history_sizes=HISTORY_SIZES,
            device=pipe.device,
            dtype=pipe.torch_dtype,
        )
        pipe.dit.set_helios_history(*history)
        lat_long, lat_mid, lat_short, fids_long, fids_mid, fids_short = history
        print(
            "  History tiers: "
            f"long={tuple(lat_long.shape)}@{fids_long.tolist()}, "
            f"mid={tuple(lat_mid.shape)}@{fids_mid.tolist()}, "
            f"short={tuple(lat_short.shape)}@{fids_short.tolist()}"
        )

    # ── Generate chunk ────────────────────────────────────────────────
    chunk_frames = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        control_video=ctrl_chunk,
        # NOTE: reference_image intentionally omitted for k>0 here.
        # Helios history tokens handle temporal context once the model is trained.
        # Zero-shot (untrained) Helios + reference_image causes artifacts.
        reference_image=ref_image if is_first_chunk else None,
        height=HEIGHT,
        width=WIDTH,
        num_frames=CHUNK_FRAMES,
        seed=SEED + k,
        cfg_scale=CFG_SCALE,
        num_inference_steps=NUM_STEPS,
        tiled=True,
    )  # list[PIL.Image]

    # ── Clear history side-channel ────────────────────────────────────
    pipe.dit.clear_helios_history()

    # ── VAE-encode chunk output for next chunk's history ─────────────
    accumulated_lats.append(encode_chunk_to_latents(chunk_frames))

    prev_last_frame = chunk_frames[-1]

    if is_first_chunk:
        all_frames.extend(chunk_frames)
    else:
        all_frames.extend(chunk_frames[1:])   # skip overlap frame

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
os.makedirs(SAVE_PATH, exist_ok=True)
index = len(os.listdir(SAVE_PATH)) + 1
out_path = os.path.join(SAVE_PATH, f"{index:08d}.mp4")
save_video(all_frames, out_path, fps=FPS, quality=7)
print(f"\nSaved {len(all_frames)} frames → {out_path}")
