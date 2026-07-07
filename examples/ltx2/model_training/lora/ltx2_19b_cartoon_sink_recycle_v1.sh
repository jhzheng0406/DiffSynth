# LTX-2 19B teacher LoRA + sink/recent conditioning + latent error recycle.
# Cross-architecture port of Wan2.1-Fun-V1.1-1.3B-Control_cartoon_sink_recycle.sh.
#
# BEFORE first run:
#   1. Run the chaining sanity check (zero-shot, no training):
#        GPU=0 python examples/ltx2/model_inference/LTX-2-Chunk-Chain-Sanity.py
#      to validate the sink_index / strength mapping on the base model.
#   2. Mirror the cartoon metadata.csv — same format as the Wan one
#      (video, control_video, prompt, num_frames). Paths below assume the
#      same ./data/cartoon_15s/metadata.csv.
#
# Memory: 19B DiT bf16 ≈ 38 GB + gemma-3-12b text encoder in-loop. On H200
# (140 GB) single-stage training should fit with gradient checkpointing; if
# the text encoder pushes it over, move to the splited (cache) recipe — but
# note the recycle hook needs the VAE/image-encode units in the loop, so
# only the PROMPT side can be cached, not the conditioning latents.
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

# Pose control (optional, OFF for v1): add
#   --use_pose_control \
#   --preset_lora_path <union-control safetensors> --preset_lora_model dit \
# after verifying union-control handles cartoon pose at inference.

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
  --model_paths '["/mnt/vita/scratch/vita-students/users/jinghao/code/LTX-2-old/models/LTX-2/ltx-2-19b-dev.safetensors"]' \
  --model_id_with_origin_paths "google/gemma-3-12b-it-qat-q4_0-unquantized:model-*.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --save_steps 200 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/LTX-2-19b_lora_cartoon_sink_recycle_v1" \
  --lora_base_model "dit" \
  --lora_target_modules "to_k,to_q,to_v,to_out.0" \
  --lora_rank 32 \
  --recent_aug_strength 0.5 \
  --sink_index -49 \
  --input_images_strength 1.0 \
  --recycle_alpha 0.3 \
  --recycle_clean_prob 0.1 \
  --recycle_buffer_size 500 \
  --recycle_warmup 200 \
  --recycle_ramp 500 \
  --recycle_push_prob 0.5 \
  --recycle_collect_start_step 100 \
  --use_gradient_checkpointing \
  --find_unused_parameters \
  $RESUME_ARG
