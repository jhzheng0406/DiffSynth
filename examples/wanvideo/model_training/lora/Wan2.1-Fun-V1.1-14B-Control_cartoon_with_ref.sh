# 14B LoRA training: cartoon style + reference_image input, NO EAD.
# Ablation control between cartoon.sh (no ref, no EAD) and cartoon_ead.sh
# (ref + EAD). Trains LoRA from scratch on the 14B base — same step count
# as the EAD variant, so the only diff vs cartoon_ead is: EAD on/off.
#
# Inputs:
#   reference_image = first frame of target video (geometric match with
#   chain-inference ref). No corruption applied.
#
# Prerequisite (run once, in videox env):
#   python data/humanvid_diffsynth/extract_refs.py \
#       --src_csv data/cartoon_diffsynth/metadata.csv \
#       --base    /mnt/vita/scratch/datasets/svi/wan-animate/high-quality-subset/preprocessed \
#       --ref_dir data/cartoon_diffsynth/refs \
#       --out_csv data/cartoon_diffsynth/metadata.csv
#
# Memory note: 14B + ref is heavier than 14B alone. If OOM:
#   - Add --use_gradient_checkpointing_offload
#   - Bump NUM_GPUS to 4+
#
# Run from DiffSynth-Studio repo root.
# Override GPU count with:  NUM_GPUS=4 bash <this-script>
#
# Optional resume:
#   RESUME_FROM=models/train/.../step-2000.safetensors bash <this-script>
#   - If unset → fresh training from base, output goes to .../lora_cartoon_with_ref
#   - If set   → continues LoRA from that ckpt, output goes to
#                .../lora_cartoon_with_ref_resume_from_<basename>

NUM_GPUS=${NUM_GPUS:-2}
RESUME_FROM=${RESUME_FROM:-models/train/Wan2.1-Fun-V1.1-14B-Control_lora_cartoon_with_ref/step-400.safetensors}

if [ -n "$RESUME_FROM" ]; then
    if [ ! -f "$RESUME_FROM" ]; then
        echo "[error] RESUME_FROM points to a non-existent file: $RESUME_FROM"
        exit 1
    fi
    # Extract the step number from "step-N.safetensors" so saved checkpoints
    # in this run continue the cumulative count (step-N+200, step-N+400, ...).
    CKPT_BASE=$(basename "$RESUME_FROM" .safetensors)
    STEP_OFFSET=$(echo "$CKPT_BASE" | sed -n 's/^step-\([0-9]\+\)$/\1/p')
    STEP_OFFSET=${STEP_OFFSET:-0}
    RESUME_ARG="--lora_checkpoint $RESUME_FROM --global_step_offset $STEP_OFFSET"
    OUT_SUFFIX=""
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
  --output_path "./models/train/Wan2.1-Fun-V1.1-14B-Control_lora_cartoon_with_ref${OUT_SUFFIX}" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --extra_inputs "control_video,reference_image" \
  --use_gradient_checkpointing \
  $RESUME_ARG
