# 1.3B sink + recent LoRA, NO recent augmentation (--recent_aug_strength 0).
#
# This is a "clean" sink teacher: it learns sink anchoring + cartoon domain but
# is NOT trained to tolerate drifted recents. Used as the teacher for a clean
# asym-DMD ablation — with zero augment baked into the teacher, any robustness
# in the resulting student must come purely from the asymmetric augment
# mechanism during DMD (teacher clean recent / student augmented recent).
#
# Same recipe as Wan2.1-Fun-V1.1-1.3B-Control_cartoon_sink.sh except:
#   --recent_aug_strength 0           (no augment)
#   --output_path .../_sink_noaug     (separate dir)
#
# Run from DiffSynth-Studio repo root.  NUM_GPUS=4 bash <this-script>
# Optional resume: RESUME_FROM=models/train/.../step-N.safetensors bash <this-script>

NUM_GPUS=${NUM_GPUS:-4}
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
    echo "[resume] continuing LoRA from $RESUME_FROM  (step_offset=$STEP_OFFSET)"
else
    RESUME_ARG=""
    echo "[fresh] training clean (no-augment) sink LoRA from base model"
fi

accelerate launch \
  --num_processes=$NUM_GPUS \
  --mixed_precision=bf16 \
  examples/wanvideo/model_training/train_chunk_aware.py \
  --dataset_base_path . \
  --dataset_metadata_path ./data/cartoon_15s/metadata.csv \
  --data_file_keys "video,control_video" \
  --height 832 \
  --width 480 \
  --num_frames 49 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "PAI/Wan2.1-Fun-V1.1-1.3B-Control:diffusion_pytorch_model*.safetensors,PAI/Wan2.1-Fun-V1.1-1.3B-Control:models_t5_umt5-xxl-enc-bf16.pth,PAI/Wan2.1-Fun-V1.1-1.3B-Control:Wan2.1_VAE.pth,PAI/Wan2.1-Fun-V1.1-1.3B-Control:models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --save_steps 200 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.1-Fun-V1.1-1.3B-Control_lora_cartoon_sink_noaug" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --extra_inputs "control_video,reference_image,sink_reference_image" \
  --recent_aug_strength 0 \
  --use_gradient_checkpointing \
  $RESUME_ARG
