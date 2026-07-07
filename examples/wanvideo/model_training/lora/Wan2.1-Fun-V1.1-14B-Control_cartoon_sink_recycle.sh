# 14B teacher (Stage A) — recycle + PLP, CARTOON. Scale-generality point:
# shows our method (sink + latent-space error recycle + PLP) works at 14B, not
# just 1.3B. Mirrors the 1.3B recycle teacher recipe but with the 14B base.
# (The pre-existing 14B cartoon_sink teacher used plain train_chunk_aware.py —
#  no recycle/PLP — so it is NOT our method.)
#
# Cartoon (832x480) is already roughly portrait-matched → no --aspect_crop
# (that's only for 3:4 sources like UBC).
#
# Memory: 14B + LoRA + activations is heavy → gradient checkpointing on; if OOM,
# lower NUM_GPUS won't help (per-GPU memory), instead add --offload or reduce
# lora_rank. Budget far more than the 1.3B run.
#
# Usage:
#   NUM_GPUS=8 bash <this-script>
#   RESUME_FROM=models/train/.../step-N.safetensors bash <this-script>

NUM_GPUS=${NUM_GPUS:-8}
RESUME_FROM=${RESUME_FROM:-}

if [ -n "$RESUME_FROM" ]; then
    [ -f "$RESUME_FROM" ] || { echo "[error] RESUME_FROM not found: $RESUME_FROM"; exit 1; }
    STEP_OFFSET=$(basename "$RESUME_FROM" .safetensors | sed -n 's/^step-\([0-9]\+\)$/\1/p')
    RESUME_ARG="--lora_checkpoint $RESUME_FROM --global_step_offset ${STEP_OFFSET:-0}"
    echo "[resume] from $RESUME_FROM (offset=${STEP_OFFSET:-0})"
else
    RESUME_ARG=""
    echo "[fresh] 14B teacher from base"
fi

accelerate launch \
  --num_processes=$NUM_GPUS \
  --mixed_precision=bf16 \
  examples/wanvideo/model_training/train_chunk_aware_recycle.py \
  --dataset_base_path . \
  --dataset_metadata_path ./data/cartoon_15s/metadata.csv \
  --data_file_keys "video,control_video" \
  --height 832 \
  --width 480 \
  --num_frames 49 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "PAI/Wan2.1-Fun-V1.1-14B-Control:diffusion_pytorch_model*.safetensors,PAI/Wan2.1-Fun-V1.1-14B-Control:models_t5_umt5-xxl-enc-bf16.pth,PAI/Wan2.1-Fun-V1.1-14B-Control:Wan2.1_VAE.pth,PAI/Wan2.1-Fun-V1.1-14B-Control:models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --save_steps 100 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.1-Fun-V1.1-14B-Control_lora_cartoon_sink_recycle" \
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
  --dataset_num_workers 4 \
  --use_gradient_checkpointing \
  $RESUME_ARG
