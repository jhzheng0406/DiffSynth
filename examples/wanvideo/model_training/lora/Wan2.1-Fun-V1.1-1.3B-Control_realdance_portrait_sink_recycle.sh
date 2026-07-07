# Teacher LoRA (Stage A) — PORTRAIT real-human dance, benchmark-aligned track.
#
# vs the landscape mariam version (Wan2.1-Fun-..._realdance_sink_recycle.sh):
#   - data: data/realdance_portrait/metadata.csv  (UBC 3:4 + single-person 9:16,
#     ~800 portrait single-person videos; built by build_realdance_metadata.py
#     --subsets UBC,single --aspect portrait)
#   - resolution: PORTRAIT 832x480 (height>width) to match TikTok/UBC eval domain
#     (SVI dance benchmark is portrait). Landscape mariam was 480x832.
#   - output dir: ..._realdance_portrait_sink_recycle_v1
# Everything else (sink+recent ref, image aug, latent-space error recycle)
# identical to the landscape/cartoon recipe — control wiring unchanged.
#
# Usage:
#   bash <this-script>                                  # 4 GPUs default
#   NUM_GPUS=8 bash <this-script>
#   RESUME_FROM=models/train/.../step-N.safetensors bash <this-script>

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
    echo "[fresh] training LoRA from base model"
fi

accelerate launch \
  --num_processes=$NUM_GPUS \
  --mixed_precision=bf16 \
  examples/wanvideo/model_training/train_chunk_aware_recycle.py \
  --dataset_base_path . \
  --dataset_metadata_path ./data/realdance_portrait/metadata_staged.csv \
  --data_file_keys "video,control_video" \
  --height 832 \
  --width 480 \
  --num_frames 49 \
  --dataset_repeat 1 \
  --dataset_num_workers 4 \
  --model_id_with_origin_paths "PAI/Wan2.1-Fun-V1.1-1.3B-Control:diffusion_pytorch_model*.safetensors,PAI/Wan2.1-Fun-V1.1-1.3B-Control:models_t5_umt5-xxl-enc-bf16.pth,PAI/Wan2.1-Fun-V1.1-1.3B-Control:Wan2.1_VAE.pth,PAI/Wan2.1-Fun-V1.1-1.3B-Control:models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --save_steps 100 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.1-Fun-V1.1-1.3B-Control_lora_realdance_portrait_sink_recycle_v1" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --extra_inputs "control_video,reference_image,sink_reference_image" \
  --recent_aug_strength 0.5 \
  --recycle_alpha 0.3 \
  --recycle_clean_prob 0.1 \
  --recycle_buffer_size 500 \
  --recycle_warmup 200 \
  --recycle_ramp 500 \
  --recycle_push_prob 0.5 \
  --recycle_collect_start_step 100 \
  --plp \
  --aspect_crop \
  --use_gradient_checkpointing \
  $RESUME_ARG
