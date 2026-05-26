# 14B LoRA + EAD training: cartoon style + Helios-style reference noise.
# Parallel to the 1.3B cartoon_ead.sh, but on the 14B base.
# Clean ablation — trains LoRA from scratch on base (no --lora_checkpoint);
# only diff vs the no-ref baseline (cartoon.sh) is: + reference_image input
# + EAD noise on ref_latents.
#
# Inputs:
#   1. reference_image (first frame of target video, geometric match with
#      chain-inference ref).
#   2. Helios EAD: corrupt reference_latents with sigma ~ U[0, corrupt_ratio]
#      during training, mimicking chain-inference artifacts so the model is
#      robust to imperfect refs at inference.
#
# Prerequisite (run once, in videox env):
#   python data/humanvid_diffsynth/extract_refs.py \
#       --src_csv data/cartoon_diffsynth/metadata.csv \
#       --base    /mnt/vita/scratch/datasets/svi/wan-animate/high-quality-subset/preprocessed \
#       --ref_dir data/cartoon_diffsynth/refs \
#       --out_csv data/cartoon_diffsynth/metadata.csv
# (Already done if you ran the 1.3B EAD recipe; PNGs are idempotent.)
#
# Memory note: 14B + ref + EAD is heavier than 14B alone. If OOM:
#   - Add --use_gradient_checkpointing_offload
#   - Bump NUM_GPUS to 4+
#
# Run from DiffSynth-Studio repo root.
# Override GPU count with:  NUM_GPUS=4 bash <this-script>
#
# Optional resume:
#   RESUME_FROM=models/train/.../step-N.safetensors bash <this-script>
#   - If unset → fresh training from base
#   - If set   → continues LoRA from that ckpt; saved files use cumulative step

NUM_GPUS=${NUM_GPUS:-2}
RESUME_FROM=${RESUME_FROM:-}

if [ -n "$RESUME_FROM" ]; then
    if [ ! -f "$RESUME_FROM" ]; then
        echo "[error] RESUME_FROM points to a non-existent file: $RESUME_FROM"
        exit 1
    fi
    CKPT_BASE=$(basename "$RESUME_FROM" .safetensors)
    STEP_OFFSET=$(echo "$CKPT_BASE" | sed -n 's/^step-\([0-9]\+\)$/\1/p')
    STEP_OFFSET=${STEP_OFFSET:-0}
    RESUME_ARG="--lora_checkpoint $RESUME_FROM --global_step_offset $STEP_OFFSET"
    OUT_SUFFIX="_resume_from_${CKPT_BASE}"
    echo "[resume] continuing LoRA from $RESUME_FROM  (step_offset=$STEP_OFFSET)"
else
    RESUME_ARG=""
    OUT_SUFFIX=""
    echo "[fresh] training LoRA from base model"
fi

accelerate launch \
  --num_processes=$NUM_GPUS \
  --mixed_precision=bf16 \
  examples/wanvideo/model_training/train.py \
  --dataset_base_path /mnt/vita/scratch/datasets/svi/wan-animate/high-quality-subset/preprocessed \
  --dataset_metadata_path ./data/cartoon_diffsynth/metadata.csv \
  --data_file_keys "video,control_video,reference_image" \
  --height 832 \
  --width 480 \
  --num_frames 49 \
  --dataset_repeat 10 \
  --model_id_with_origin_paths "PAI/Wan2.1-Fun-V1.1-14B-Control:diffusion_pytorch_model*.safetensors,PAI/Wan2.1-Fun-V1.1-14B-Control:models_t5_umt5-xxl-enc-bf16.pth,PAI/Wan2.1-Fun-V1.1-14B-Control:Wan2.1_VAE.pth,PAI/Wan2.1-Fun-V1.1-14B-Control:models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
  --learning_rate 1e-4 \
  --num_epochs 4 \
  --save_steps 200 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.1-Fun-V1.1-14B-Control_lora_cartoon_ead${OUT_SUFFIX}" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --extra_inputs "control_video,reference_image" \
  --use_gradient_checkpointing \
  --ead_corrupt_ratio 0.33 \
  --ead_clean_prob 0.10 \
  $RESUME_ARG
