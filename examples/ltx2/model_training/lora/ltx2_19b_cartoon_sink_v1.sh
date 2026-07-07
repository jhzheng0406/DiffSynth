# LTX-2 19B teacher LoRA, NO-RECYCLE CONTROL — the LTX-2 analogue of Wan's
# sink_v2 (recent aug ON, recycle OFF). Required ablation pair for
# ltx2_19b_cartoon_sink_recycle_v1.sh: without this, the LTX-2 transfer
# experiment has no within-architecture recycle-vs-no-recycle comparison.
#
# Synced to the v4 recipe (2026-06-12): FFN targets, pose control, positive-shift
# sink (index 1 + time_shift 48). Identical to ltx2_19b_cartoon_sink_recycle_v4.sh except:
#   --recycle_alpha 0   (recycle hook not installed → pure chunk-aware + aug)
#   output dir
#
# Usage:
#   NUM_GPUS=8 bash <this-script>
#   RESUME_FROM=models/train/.../step-N.safetensors bash <this-script>

NUM_GPUS=${NUM_GPUS:-8}
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
    echo "[fresh] training LoRA from base model"
fi

accelerate launch \
  --num_processes=$NUM_GPUS \
  --mixed_precision=bf16 \
  examples/ltx2/model_training/train_chunk_aware_recycle_ltx2.py \
  --dataset_base_path . \
  --dataset_metadata_path ./data/cartoon_15s/metadata.csv \
  --data_file_keys "video,control_video" \
  --height 832 \
  --width 480 \
  --num_frames 49 \
  --frame_rate 24 \
  --dataset_repeat 1 \
  --model_paths '["models/Lightricks/LTX-2.3/ltx-2.3-22b-dev.safetensors"]' \
  --model_id_with_origin_paths "google/gemma-3-12b-it-qat-q4_0-unquantized:model-*.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 10 \
  --save_steps 200 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/LTX-2.3-22b_lora_cartoon_sink_v6_norecycle" \
  --lora_base_model "dit" \
  --lora_target_modules "to_k,to_q,to_v,to_out.0,net.0.proj,net.2" \
  --lora_rank 32 \
  --recent_aug_strength 0.5 \
  --sink_index 1 --time_shift_frames 48 \
  --input_images_strength 1.0 \
  --recycle_alpha 0 \
  --use_pose_control \
  --preset_lora_path "models/Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors" \
  --preset_lora_model dit \
  --use_gradient_checkpointing \
  --find_unused_parameters \
  $RESUME_ARG
