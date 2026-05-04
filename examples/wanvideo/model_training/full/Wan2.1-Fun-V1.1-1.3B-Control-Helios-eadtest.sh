#!/bin/bash
# Sanity-check training for the EAD fixes.
# Same setup as Wan2.1-Fun-V1.1-1.3B-Control-Helios.sh (v4) but:
#   * uses the new convex-blend noise formula (wan_video_helios_attention.py)
#   * uses upstream-aligned EAD ratios from Helios correct.yaml
#   * enables saturation augmentation (was off in v4)
# Goal: train ~100-200 steps and compare loss curve / output quality vs v4.
#
# Run from the DiffSynth-Studio repo root:
#   bash examples/wanvideo/model_training/full/Wan2.1-Fun-V1.1-1.3B-Control-Helios-eadtest.sh

set -e

# ---------------------------------------------------------------------------
# Paths (identical to v4 for fair A/B)
# ---------------------------------------------------------------------------
VIDEOXFUN_ROOT="/mnt/vita/scratch/vita-students/users/jinghao/code/VideoX-Fun"
MODEL_DIR="${VIDEOXFUN_ROOT}/models/Diffusion_Transformer/Wan2.1-Fun-V1.1-1.3B-Control"

DATASET_BASE="/mnt/vita/scratch/datasets/svi/wan-animate/high-quality-subset/preprocessed"
SRC_CSV="${VIDEOXFUN_ROOT}/datasets/antidrift_train.csv"

CONVERTED_CSV="/tmp/antidrift_train_diffsynth.csv"

# Distinct output dir so v4 checkpoints aren't touched
OUTPUT_DIR="./models/train/Wan2.1-Fun-V1.1-1.3B-Control-Helios-v6-eadtest"

RESUME_CKPT=""

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
if [ -n "${RESUME_CKPT}" ]; then
  DIT_CKPT="${RESUME_CKPT}"
fi

# ---------------------------------------------------------------------------
# What's different from v4:
#   - --helios_corrupt_mode "random"          (v4: "noise")
#   - --helios_noise_ratio_short/mid/long 0.3333  (v4: 0.02/0.03/0.04, ~10x weaker)
#   - --helios_apply_saturation               (v4: omitted -> off)
#   - --helios_saturation_min 0.3 / max 1.7   (matches Helios correct.yaml)
#   - --helios_clean_history_prob 0.1
#   - --helios_use_first_frame_anchor         (v4: not present -> color drift)
# Code-side fixes (apply automatically):
#   - convex-blend EAD formula        (was additive: x + σε)
#   - First-Frame Anchor              (lat_short carries [anchor, recent], fids [0, 7])
#   - Relative RoPE                   (current chunk freqs offset by 1 + sum(sizes))
# ---------------------------------------------------------------------------

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
  --num_frames 77 \
  --dataset_repeat 1 \
  --model_paths "[\"${DIT_CKPT}\",\"${MODEL_DIR}/models_t5_umt5-xxl-enc-bf16.pth\",\"${MODEL_DIR}/Wan2.1_VAE.pth\",\"${MODEL_DIR}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth\"]" \
  --tokenizer_path "${MODEL_DIR}/google/umt5-xxl" \
  --trainable_models "dit" \
  --extra_inputs "control_video" \
  --learning_rate 1e-5 \
  --num_epochs 1 \
  --early_save_steps 20 \
  --save_steps 50 \
  --gradient_accumulation_steps 4 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "${OUTPUT_DIR}" \
  --helios_history_sizes "4,2,1" \
  --helios_init_scale_logit -3.0 \
  --helios_corrupt_mode "random" \
  --helios_noise_ratio_short 0.3333 \
  --helios_noise_ratio_mid   0.3333 \
  --helios_noise_ratio_long  0.3333 \
  --helios_apply_saturation \
  --helios_saturation_min 0.3 \
  --helios_saturation_max 1.7 \
  --helios_clean_history_prob 0.1 \
  --helios_use_first_frame_anchor \
  --helios_drop_t2v_ratio 0.10 \
  --helios_drop_i2v_ratio 0.05 \
  --use_gradient_checkpointing \
  --find_unused_parameters
