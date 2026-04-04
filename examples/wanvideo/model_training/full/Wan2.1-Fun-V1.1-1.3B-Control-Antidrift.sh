#!/bin/bash
# Anti-drift fine-tuning for Wan2.1-Fun-V1.1-1.3B-Control
# Uses dataset from VideoX-Fun and model weights from local VideoX-Fun checkout.
#
# Run from the DiffSynth-Studio repo root:
#   bash examples/wanvideo/model_training/full/Wan2.1-Fun-V1.1-1.3B-Control-Antidrift.sh

set -e
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
VIDEOXFUN_ROOT="/mnt/vita/scratch/vita-students/users/jinghao/code/VideoX-Fun"
MODEL_DIR="${VIDEOXFUN_ROOT}/models/Diffusion_Transformer/Wan2.1-Fun-V1.1-1.3B-Control"

DATASET_BASE="/mnt/vita/scratch/datasets/svi/wan-animate/high-quality-subset/preprocessed"
SRC_CSV="${VIDEOXFUN_ROOT}/datasets/antidrift_train.csv"

# Converted CSV (DiffSynth expects: video / prompt / control_video columns)
CONVERTED_CSV="/tmp/antidrift_train_diffsynth.csv"

OUTPUT_DIR="./models/train/Wan2.1-Fun-V1.1-1.3B-Control-Antidrift"

# Set to the latest checkpoint to resume training, or leave empty to start fresh.
# e.g. RESUME_CKPT="${OUTPUT_DIR}/step-9300.safetensors"
RESUME_CKPT="${OUTPUT_DIR}/step-9300.safetensors"

# ---------------------------------------------------------------------------
# Convert CSV columns once:
#   file_path        -> video
#   text             -> prompt
#   control_file_path -> control_video
#   (drop 'type')
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
export NCCL_P2P_LEVEL=0       # disable NVLink peer access
export NCCL_SHM_DISABLE=1     # fall back to socket transport
export CUDA_VISIBLE_DEVICES=0,1,2

accelerate launch \
  --mixed_precision="bf16" \
  --num_processes=1 \
  examples/wanvideo/model_training/train_antidrift.py \
  --dataset_base_path "${DATASET_BASE}" \
  --dataset_metadata_path "${CONVERTED_CSV}" \
  --data_file_keys "video,control_video" \
  --height 480 \
  --width 832 \
  --num_frames 65 \
  --dataset_repeat 1 \
  --model_paths "[\"${RESUME_CKPT:-${MODEL_DIR}/diffusion_pytorch_model.safetensors}\",\"${MODEL_DIR}/models_t5_umt5-xxl-enc-bf16.pth\",\"${MODEL_DIR}/Wan2.1_VAE.pth\",\"${MODEL_DIR}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth\"]" \
  --tokenizer_path "${MODEL_DIR}/google/umt5-xxl" \
  --trainable_models "dit" \
  --extra_inputs "control_video" \
  --learning_rate 2e-5 \
  --num_epochs 8 \
  --save_steps 100 \
  --gradient_accumulation_steps 4 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "${OUTPUT_DIR}" \
  --antidrift_history_frames 4 \
  --antidrift_drop_ref_ratio 0.10 \
  --antidrift_ref_noise_ratio 0.03 \
  --use_gradient_checkpointing
