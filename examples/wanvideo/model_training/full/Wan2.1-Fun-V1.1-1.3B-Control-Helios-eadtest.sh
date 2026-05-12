#!/bin/bash
# Sanity-check training for freeze-base Helios modules.
# Same setup as the previous EAD-clean run but:
#   * uses the new convex-blend noise formula (wan_video_helios_attention.py)
#   * gives history corruption an official-Helios-style clean skip probability
#   * freezes the base Wan-Control DiT and trains only Helios history patch modules
#   * disables history-key amplification to match official Helios stage configs
#   * enables mild saturation augmentation
# Goal: train ~100-300 steps and compare output quality vs full-DiT SFT.
#
# Run from the DiffSynth-Studio repo root:
#   bash examples/wanvideo/model_training/full/Wan2.1-Fun-V1.1-1.3B-Control-Helios-eadtest.sh

set -e

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
VIDEOXFUN_ROOT="/mnt/vita/scratch/vita-students/users/jinghao/code/VideoX-Fun"
MODEL_DIR="${VIDEOXFUN_ROOT}/models/Diffusion_Transformer/Wan2.1-Fun-V1.1-1.3B-Control"

DATASET_BASE="/mnt/vita/scratch/datasets/svi/wan-animate/high-quality-subset/preprocessed"
SRC_CSV="${VIDEOXFUN_ROOT}/datasets/antidrift_train.csv"

CONVERTED_CSV="/tmp/antidrift_train_diffsynth.csv"

# Distinct output dir so older checkpoints aren't touched
OUTPUT_DIR="./models/train/Wan2.1-Fun-V1.1-1.3B-Control-Helios-v11-freezebase"

RESUME_CKPT=""
# RESUME_CKPT="./models/train/Wan2.1-Fun-V1.1-1.3B-Control-Helios-v11-freezebase/step-450.safetensors"

# ---------------------------------------------------------------------------
# Convert CSV columns once (same dataset as antidrift)
# ---------------------------------------------------------------------------
if [ ! -f "${CONVERTED_CSV}" ]; then
  python3 -c "
import csv, sys
src, dst = sys.argv[1], sys.argv[2]
with open(src, newline='') as fin, open(dst, 'w', newline='') as fout:
    reader = csv.DictReader(fin)
    writer = csv.DictWriter(fout, fieldnames=['video', 'prompt', 'control_video'])
    writer.writeheader()
    n = sum(writer.writerow({'video': r['file_path'], 'prompt': r.get('text',''), 'control_video': r['control_file_path']}) or 1 for r in reader)
    print(f'[csv] {n} rows -> {dst}')
" "${SRC_CSV}" "${CONVERTED_CSV}"
else
  echo "[csv] ${CONVERTED_CSV} already exists, skipping conversion."
fi

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_P2P_LEVEL=0
export NCCL_SHM_DISABLE=1
export CUDA_VISIBLE_DEVICES=0

DIT_CKPT="${MODEL_DIR}/diffusion_pytorch_model.safetensors"
HELIOS_RESUME_ARGS=()
if [ -n "${RESUME_CKPT}" ]; then
  HELIOS_RESUME_ARGS=(--helios_resume_ckpt "${RESUME_CKPT}")
fi

# ---------------------------------------------------------------------------
# What's different from v10-eadclean:
#   - starts from the base DiT checkpoint instead of resuming v8 weights
#   - --helios_train_mode "helios_modules" freezes the base DiT
#   - --no-helios_amplify_history removes the extra history-key scaling path
#   - --helios_clean_history_prob 0.1 now also skips EAD corruption entirely
#     with 10% probability, matching official Helios' clean history branch
#   - keeps mild EAD ratios so the test isolates clean-skip behavior first
# Code-side fixes (apply automatically):
#   - convex-blend EAD formula        (was additive: x + σε)
#   - First-Frame Anchor              (lat_short carries [anchor, recent], fids [0, 7])
#   - Relative RoPE                   (current chunk freqs offset by 1 + sum(sizes))
# ---------------------------------------------------------------------------

# num_frames=81 -> 21 latent frames. With anchor + history [4,2,1] = 8
# history positions, the supervised target has 13 latent frames, matching
# inference CHUNK_FRAMES=49.
# num_epochs=1 + frequent saves -> easy to interrupt after ~100-200 steps
accelerate launch \
  --mixed_precision="bf16" \
  --num_processes=1 \
  examples/wanvideo/model_training/train_helios.py \
  --dataset_base_path "${DATASET_BASE}" \
  --dataset_metadata_path "${CONVERTED_CSV}" \
  --data_file_keys "video,control_video" \
  --height 480 \
  --width 832 \
  --num_frames 81 \
  --dataset_repeat 1 \
  --model_paths "[\"${DIT_CKPT}\",\"${MODEL_DIR}/models_t5_umt5-xxl-enc-bf16.pth\",\"${MODEL_DIR}/Wan2.1_VAE.pth\",\"${MODEL_DIR}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth\"]" \
  --tokenizer_path "${MODEL_DIR}/google/umt5-xxl" \
  --trainable_models "dit" \
  --extra_inputs "control_video" \
  --learning_rate 5e-5 \
  --num_epochs 1 \
  --early_save_steps 20 \
  --save_steps 50 \
  --gradient_accumulation_steps 4 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "${OUTPUT_DIR}" \
  --helios_train_mode "helios_modules" \
  --no-helios_amplify_history \
  --helios_history_sizes "4,2,1" \
  --helios_init_scale_logit -3.0 \
  --helios_corrupt_mode "random" \
  --helios_noise_ratio_short 0.05 \
  --helios_noise_ratio_mid   0.08 \
  --helios_noise_ratio_long  0.10 \
  --helios_apply_saturation \
  --helios_saturation_min 0.7 \
  --helios_saturation_max 1.3 \
  --helios_clean_history_prob 0.1 \
  --helios_use_first_frame_anchor \
  --helios_drop_t2v_ratio 0.10 \
  --helios_drop_i2v_ratio 0.05 \
  "${HELIOS_RESUME_ARGS[@]}" \
  --use_gradient_checkpointing \
  --find_unused_parameters
