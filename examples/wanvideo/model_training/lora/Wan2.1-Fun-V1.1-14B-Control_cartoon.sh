# LoRA fine-tune on Cartoon (200 samples) for Wan2.1-Fun-V1.1-14B-Control.
# Same data + training scheme as the 1.3B cartoon shell, just on the 14B base.
# Use this as the 14B LoRA baseline before any EAD / chain experiments.
#
# Memory: bf16 14B + LoRA + gradient checkpointing fits on a single H100 80GB
# (tight). If you hit OOM, options:
#   - Use 4+ GPUs with DeepSpeed ZeRO-2 (see full/accelerate_config_14B.yaml)
#   - Add --use_gradient_checkpointing_offload
#   - Drop --num_frames to 33 or smaller resolution
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
  --data_file_keys "video,control_video" \
  --height 832 \
  --width 480 \
  --num_frames 49 \
  --dataset_repeat 10 \
  --model_id_with_origin_paths "PAI/Wan2.1-Fun-V1.1-14B-Control:diffusion_pytorch_model*.safetensors,PAI/Wan2.1-Fun-V1.1-14B-Control:models_t5_umt5-xxl-enc-bf16.pth,PAI/Wan2.1-Fun-V1.1-14B-Control:Wan2.1_VAE.pth,PAI/Wan2.1-Fun-V1.1-14B-Control:models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
  --learning_rate 1e-4 \
  --num_epochs 2 \
  --save_steps 200 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.1-Fun-V1.1-14B-Control_lora_cartoon${OUT_SUFFIX}" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --extra_inputs "control_video" \
  --use_gradient_checkpointing \
  $RESUME_ARG
