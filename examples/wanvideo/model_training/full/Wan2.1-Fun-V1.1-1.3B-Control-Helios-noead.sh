#!/bin/bash
# Clean no-EAD Helios ablation.
#
# This keeps the Helios history path, first-frame anchor, relative RoPE, and
# history-key amplification training, but removes all EAD-style history
# corruption/saturation. It is meant to answer one question quickly:
#   can the current Helios port learn to use clean history without blurring
#   the background?
#
# Run from the DiffSynth-Studio repo root:
#   bash examples/wanvideo/model_training/full/Wan2.1-Fun-V1.1-1.3B-Control-Helios-noead.sh

set -e

VIDEOXFUN_ROOT="/mnt/vita/scratch/vita-students/users/jinghao/code/VideoX-Fun"
MODEL_DIR="${VIDEOXFUN_ROOT}/models/Diffusion_Transformer/Wan2.1-Fun-V1.1-1.3B-Control"

DATASET_BASE="/mnt/vita/scratch/datasets/svi/wan-animate/high-quality-subset/preprocessed"
SRC_CSV="${VIDEOXFUN_ROOT}/datasets/antidrift_train.csv"
CONVERTED_CSV="/tmp/antidrift_train_diffsynth.csv"

OUTPUT_DIR="./models/train/Wan2.1-Fun-V1.1-1.3B-Control-Helios-v9-noead"
RESUME_CKPT=""

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

# num_frames=81 -> 21 latent frames. With anchor + history [4,2,1] = 8
# history positions, the supervised target has 13 latent frames, matching
# inference CHUNK_FRAMES=49.
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
  --learning_rate 1e-5 \
  --num_epochs 1 \
  --early_save_steps 20 \
  --save_steps 50 \
  --gradient_accumulation_steps 4 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "${OUTPUT_DIR}" \
  --helios_history_sizes "4,2,1" \
  --helios_init_scale_logit -3.0 \
  --helios_corrupt_mode "none" \
  --helios_noise_ratio_short 0.0 \
  --helios_noise_ratio_mid 0.0 \
  --helios_noise_ratio_long 0.0 \
  --no-helios_apply_saturation \
  --helios_clean_history_prob 1.0 \
  --helios_use_first_frame_anchor \
  --helios_drop_t2v_ratio 0.0 \
  --helios_drop_i2v_ratio 0.0 \
  "${HELIOS_RESUME_ARGS[@]}" \
  --use_gradient_checkpointing \
  --find_unused_parameters
