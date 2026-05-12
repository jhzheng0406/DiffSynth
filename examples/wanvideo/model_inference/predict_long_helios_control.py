"""
Long video generation with Helios-style multi-tier history token compression
for Wan2.1-Fun-V1.1-Control (DiffSynth-Studio).

Differences from predict_long_antidrift_control.py:
  - After loading the pipeline, calls patch_wan_model_for_helios(pipe.dit)
    so every Wan self-attention block can prepend and attend to Helios
    multi-tier history tokens.
  - Before each chunk k > 0, reuses the previous chunk's final denoised latent
    as 3-tier Helios history, then calls pipe.dit.set_helios_history()
    so every attention layer can attend to compressed history tokens
    alongside the current chunk tokens.
  - No reference_image / input_image conditioning for chunks k > 0 (history
    is injected at the attention level instead).

Usage:
    python predict_long_helios_control.py
"""

import os, sys, torch, re
from datetime import datetime
import numpy as np
from PIL import Image
from diffsynth.core import load_state_dict
from diffsynth.utils.data import VideoData, save_video
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.models.wan_video_helios_attention import (
    patch_wan_model_for_helios,
    prepare_helios_history,
)


def _safe_filename_tag(text):
    text = str(text).strip()
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return text.strip("._-") or "unknown"


def helios_checkpoint_tag(checkpoint):
    if checkpoint is None:
        return "zeroshot_no_ckpt"

    ckpt_path = checkpoint.rstrip(os.sep)
    train_version = os.path.basename(os.path.dirname(ckpt_path)) or "trained"
    common_model_prefix = "Wan2.1-Fun-V1.1-1.3B-Control-Helios"
    if train_version.startswith(common_model_prefix):
        train_version = train_version[len(common_model_prefix):].lstrip("-_") or "trained"
    ckpt_name = os.path.basename(ckpt_path)

    m = re.search(r"step[-_]?(\d+)", ckpt_name)
    step_tag = f"step{m.group(1)}" if m else os.path.splitext(ckpt_name)[0]
    return f"{_safe_filename_tag(train_version)}_{_safe_filename_tag(step_tag)}"

# ---------------------------------------------------------------------------
# Config — edit these
# ---------------------------------------------------------------------------
MODEL_DIR        = "/mnt/vita/scratch/vita-students/users/jinghao/code/VideoX-Fun/models/Diffusion_Transformer/Wan2.1-Fun-V1.1-1.3B-Control"
USE_MODEL_ID     = False
# HELIOS_CHECKPOINT options:
#   None                                          -> zero-shot (untrained Helios)
#   "models/train/.../Helios-v6-eadtest/step-XXX.safetensors"  -> NEW ckpt (anchor + Relative RoPE + EAD fixes)
# WARNING: do NOT load v4 checkpoints with this script. v4 was trained with
# the legacy layout (no anchor, no RoPE shift, additive EAD); loading it now
# is OOD and will produce worse results, not better.
# Set HELIOS_CHECKPOINT="" (empty string) via env var to run baseline (no trained ckpt).
# Otherwise edit the literal path below.
HELIOS_CHECKPOINT = os.environ.get(
    "HELIOS_CHECKPOINT",
    "models/train/Wan2.1-Fun-V1.1-1.3B-Control-Helios-v10-eadclean/step-1000.safetensors",
) or None

# Resolution must MATCH the training script (eadtest.sh uses --height 480 --width 832).
HEIGHT           = 480
WIDTH            = 832
CHUNK_FRAMES     = 49
NUM_CHUNKS       = 6

CONTROL_VIDEO    = "asset/pose_loop.mp4"
REFERENCE_IMAGE  = None   # PIL.Image or path; used for chunk 0 only

# Helios history config (3 tiers ordered as long / mid / short)
HISTORY_SIZES    = [4, 2, 1]
HISTORY_LATENT_SOURCE = "denoised"  # "denoised" or "vae_reencode"
# Diagnostic sweep to test whether the trained model actually USES helios history
# at chunk K>0. If "normal" and "no_history" look identical, model is ignoring
# helios -> training failed to teach temporal flow.
HISTORY_ABLATION_MODES = ["no_history"]
# Other useful modes:
#   normal, no_history, zero_history, random_history
#   no_long, no_mid, no_short, no_anchor, anchor_only, recent_only

# Reference-image sourcing strategy for chunks K>0 (chunk 0 always uses
# REFERENCE_IMAGE above, which may be None).
#   "fixed_first":  use chunk 0's first generated frame for ALL later chunks.
#                   stable identity, no drift, but doesn't track motion.
#   "rolling_last": use the previous chunk's last frame as ref for the next
#                   chunk. tracks character continuously (FramePack-style)
#                   but small errors can compound.
#   "none":         no ref for chunk K>0 (legacy behavior).
REF_STRATEGY = "rolling_last"
#   long_only, mid_only, short_only
#   no_anchor, anchor_only, recent_only
# Trained Helios was optimized with the first-frame anchor and t=0 history tokens.
# In zero-shot mode these can behave like a hard reference image and hijack later
# chunks, so default them off unless a Helios checkpoint is loaded.
USE_FIRST_FRAME_ANCHOR = HELIOS_CHECKPOINT is not None
# Keep this in sync with the checkpoint's training config. The freeze-base
# Helios-module runs use --no-helios_amplify_history, matching official configs.
AMPLIFY_HISTORY        = False
ZERO_HISTORY_TIMESTEP  = HELIOS_CHECKPOINT is not None

PROMPT = (
    "在这个阳光明媚的户外花园里，美女身穿一袭及膝的白色无袖连衣裙，裙摆在她轻盈的舞姿中轻柔地摆动。阳光透过树叶间洒下斑驳的光影，映衬出她柔和的脸庞和清澈的眼眸，显得格外优雅。"
)
NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)

SEED             = 42
# Per-chunk seed strategy:
#   "varying" -> SEED + k for chunk k (each chunk samples differently)
#   "fixed"   -> SEED for ALL chunks (same initial noise, expected to reduce
#                cross-chunk identity jumps when combined with rolling_ref)
SEED_STRATEGY    = "fixed"
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
    history_sizes=HISTORY_SIZES,
    is_amplify_history=AMPLIFY_HISTORY,
    freeze_history_scale=True,   # True at inference; False when training
    use_first_frame_anchor=USE_FIRST_FRAME_ANCHOR,
    zero_history_timestep=ZERO_HISTORY_TIMESTEP,
)
print(
    "Helios attention patch applied. "
    f"anchor={USE_FIRST_FRAME_ANCHOR}, "
    f"amplify_history={AMPLIFY_HISTORY}, "
    f"zero_history_timestep={ZERO_HISTORY_TIMESTEP}, "
    f"history_latent_source={HISTORY_LATENT_SOURCE}, "
    f"history_ablation_modes={HISTORY_ABLATION_MODES}"
)
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


def apply_history_ablation(history, mode):
    lat_long, lat_mid, lat_short, fids_long, fids_mid, fids_short = history

    def _drop_anchor(lat, fids):
        if lat is None or fids is None or lat.shape[2] == 0:
            return lat, fids
        keep = fids != 0
        if not keep.any():
            return lat[:, :, :0], fids[:0]
        return lat[:, :, keep], fids[keep]

    if mode == "normal":
        return history
    if mode == "zero_history":
        return (
            torch.zeros_like(lat_long),
            torch.zeros_like(lat_mid),
            torch.zeros_like(lat_short),
            fids_long,
            fids_mid,
            fids_short,
        )
    if mode == "random_history":
        return (
            torch.randn_like(lat_long),
            torch.randn_like(lat_mid),
            torch.randn_like(lat_short),
            fids_long,
            fids_mid,
            fids_short,
        )
    if mode == "no_long":
        return None, lat_mid, lat_short, None, fids_mid, fids_short
    if mode == "no_mid":
        return lat_long, None, lat_short, fids_long, None, fids_short
    if mode == "no_short":
        return lat_long, lat_mid, None, fids_long, fids_mid, None
    if mode == "long_only":
        return lat_long, None, None, fids_long, None, None
    if mode == "mid_only":
        return None, lat_mid, None, None, fids_mid, None
    if mode == "short_only":
        return None, None, lat_short, None, None, fids_short
    if mode == "no_anchor":
        lat_short_no_anchor, fids_short_no_anchor = _drop_anchor(lat_short, fids_short)
        return lat_long, lat_mid, lat_short_no_anchor, fids_long, fids_mid, fids_short_no_anchor
    if mode == "anchor_only":
        if lat_short is None or fids_short is None:
            return None, None, None, None, None, None
        keep = fids_short == 0
        return None, None, lat_short[:, :, keep], None, None, fids_short[keep]
    if mode == "recent_only":
        lat_short_recent, fids_short_recent = _drop_anchor(lat_short, fids_short)
        return None, None, lat_short_recent, None, None, fids_short_recent
    raise ValueError(f"Unknown HISTORY_ABLATION_MODE: {mode}")


def _shape_or_none(x):
    return None if x is None else tuple(x.shape)


def _fids_or_empty(x):
    return [] if x is None else x.tolist()


# ---------------------------------------------------------------------------
# Chunk generation loop
# ---------------------------------------------------------------------------
def generate_one_ablation(ablation_mode):
    all_frames = []        # accumulated output frames (PIL.Image)
    accumulated_lats = []  # list of [1, C, T_lat, H_lat, W_lat] CPU tensors
    fixed_ref = None       # captured from chunk 0's first frame (REF_STRATEGY="fixed_first")
    rolling_ref = None     # last frame of previous chunk (REF_STRATEGY="rolling_last")

    print(f"\n========== Helios history ablation: {ablation_mode}  |  REF_STRATEGY={REF_STRATEGY} ==========")
    for k in range(NUM_CHUNKS):
        is_first_chunk = (k == 0)
        print(f"\n── Chunk {k+1}/{NUM_CHUNKS} [{ablation_mode}] ──")

        # Slice control video for this chunk
        frame_start = k * (CHUNK_FRAMES - 1)
        ctrl_chunk = all_ctrl_frames[frame_start : frame_start + CHUNK_FRAMES]

        # ── Helios history injection ──────────────────────────────────
        # TRAINED MODE: only inject for chunk K>0. Chunk 0 falls back to
        # plain Wan-Fun-Control (uses reference_image).
        pipe.dit.clear_helios_history()
        if is_first_chunk:
            print("  [chunk-0] skipping Helios injection (using reference_image instead)")
        elif ablation_mode == "no_history":
            print("  [ablation] no Helios history injected")
        else:
            history = prepare_helios_history(
                accumulated_lats,
                history_sizes=HISTORY_SIZES,
                use_first_frame_anchor=USE_FIRST_FRAME_ANCHOR,
                device=pipe.device,
                dtype=pipe.torch_dtype,
            )
            history = apply_history_ablation(history, ablation_mode)
            pipe.dit.set_helios_history(*history)
            lat_long, lat_mid, lat_short, fids_long, fids_mid, fids_short = history
            print(
                "  History tiers: "
                f"long={_shape_or_none(lat_long)}@{_fids_or_empty(fids_long)}, "
                f"mid={_shape_or_none(lat_mid)}@{_fids_or_empty(fids_mid)}, "
                f"short={_shape_or_none(lat_short)}@{_fids_or_empty(fids_short)}, "
                f"mode={ablation_mode}"
            )

        # ── Pick reference_image for this chunk ───────────────────────
        if is_first_chunk:
            this_ref = ref_image
        elif REF_STRATEGY == "fixed_first":
            this_ref = fixed_ref
        elif REF_STRATEGY == "rolling_last":
            this_ref = rolling_ref
        else:  # "none"
            this_ref = None
        if this_ref is not None and not is_first_chunk:
            print(f"  [chunk-{k+1}] using ref from {REF_STRATEGY}")

        # ── Generate chunk ────────────────────────────────────────────
        pipe_output = pipe(
            prompt=PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            control_video=ctrl_chunk,
            reference_image=this_ref,
            height=HEIGHT,
            width=WIDTH,
            num_frames=CHUNK_FRAMES,
            seed=SEED if SEED_STRATEGY == "fixed" else SEED + k,
            cfg_scale=CFG_SCALE,
            num_inference_steps=NUM_STEPS,
            tiled=True,
            return_latents=HISTORY_LATENT_SOURCE == "denoised",
        )
        if HISTORY_LATENT_SOURCE == "denoised":
            chunk_frames, chunk_latents = pipe_output
        else:
            chunk_frames = pipe_output
            chunk_latents = encode_chunk_to_latents(chunk_frames)

        # ── Clear history side-channel ────────────────────────────────
        pipe.dit.clear_helios_history()

        # ── Cache chunk latent for next chunk's history ───────────────
        accumulated_lats.append(chunk_latents)

        # ── Capture references for next iteration ─────────────────────
        if is_first_chunk and REF_STRATEGY == "fixed_first" and len(chunk_frames) > 0:
            fixed_ref = chunk_frames[0]
        if REF_STRATEGY == "rolling_last" and len(chunk_frames) > 0:
            rolling_ref = chunk_frames[-1]

        if is_first_chunk:
            all_frames.extend(chunk_frames)
        else:
            all_frames.extend(chunk_frames[1:])   # skip overlap frame

    return all_frames


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
os.makedirs(SAVE_PATH, exist_ok=True)
_ts = datetime.now().strftime("%m%d_%H%M")
for ablation_mode in HISTORY_ABLATION_MODES:
    all_frames = generate_one_ablation(ablation_mode)
    _tag = (
        f"{helios_checkpoint_tag(HELIOS_CHECKPOINT)}"
        f"_hist-{_safe_filename_tag(HISTORY_LATENT_SOURCE)}"
        f"_abl-{_safe_filename_tag(ablation_mode)}"
        f"_seed-{_safe_filename_tag(SEED_STRATEGY)}"
        f"_ref-{_safe_filename_tag(REF_STRATEGY)}"
    )
    out_name = f"helios_{_tag}_{_ts}.mp4"
    out_path = os.path.join(SAVE_PATH, out_name)
    save_video(all_frames, out_path, fps=FPS, quality=7)
    print(f"\nSaved {len(all_frames)} frames → {out_path}")
