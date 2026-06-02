# DMD + One-Forcing GAN + High-Frequency pixel aux loss   (v1, fr=full)
#
# Adds to the working GAN baseline (train_dmd_oneforcing.py) a Laplacian
# high-pass MSE against the ground-truth video:
#
#   gen_loss = dmd_g + 0.03 * gan_g + highfreq_weight * hf_loss
#
# Why: GAN already handles textures well but blurs in corners. A direct
# pixel-space high-pass target pushes student to match true edge / detail
# statistics of the dataset.
#
# Code: train_dmd_oneforcing_hf.py  (new file; oneforcing.py + HF block).
# Memory: VAE decode wrapped in torch.utils.checkpoint + chunk=4
# (same pattern as train_dmd_fd_v3.py — fits on 80-100GB H200).

OUTPUT_DIR="./models/train/wan1.3b_dmd_gan_hf_v1"
TEACHER_LORA="./models/train/Wan2.1-Fun-V1.1-1.3B-Control_lora_cartoon_sink_v2/step-1745.safetensors"

RESUME_STUDENT=${RESUME_STUDENT:-}
RESUME_CRITIC=${RESUME_CRITIC:-}

if [ -z "$RESUME_STUDENT" ] && [ -d "$OUTPUT_DIR" ]; then
    LATEST=$(ls "$OUTPUT_DIR"/step-*.safetensors 2>/dev/null | grep -v "_critic" \
             | sed -n 's/.*step-\([0-9]\+\)\.safetensors/\1/p' | sort -n | tail -1)
    if [ -n "$LATEST" ]; then
        RESUME_STUDENT="$OUTPUT_DIR/step-${LATEST}.safetensors"
        RESUME_CRITIC="$OUTPUT_DIR/step-${LATEST}_critic.safetensors"
        echo "[auto-resume] latest → step-${LATEST}"
    fi
elif [ "$RESUME_STUDENT" = "none" ]; then
    RESUME_STUDENT=""; RESUME_CRITIC=""; echo "[fresh] from scratch"
fi

RESUME_ARGS=""; GLOBAL_STEP_OFFSET=0
if [ -n "$RESUME_STUDENT" ]; then
    [ -f "$RESUME_STUDENT" ] || { echo "[error] not found: $RESUME_STUDENT"; exit 1; }
    GLOBAL_STEP_OFFSET=$(basename "$RESUME_STUDENT" .safetensors | sed -n 's/^step-\([0-9]\+\).*$/\1/p')
    GLOBAL_STEP_OFFSET=${GLOBAL_STEP_OFFSET:-0}
    RESUME_ARGS="$RESUME_ARGS --resume_student_from $RESUME_STUDENT"
fi
[ -n "$RESUME_CRITIC" ] && RESUME_ARGS="$RESUME_ARGS --resume_critic_from $RESUME_CRITIC"

NUM_GPUS=${NUM_GPUS:-8}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

accelerate launch \
    --num_processes=$NUM_GPUS --mixed_precision=bf16 \
    examples/wanvideo/model_training/train_dmd_oneforcing_hf.py \
    --dataset_metadata_path ./data/cartoon_15s/metadata.csv \
    --height 832 --width 480 --num_frames 49 --dataset_repeat 1 \
    --recent_aug_strength 0.5 \
    --teacher_lora_path "$TEACHER_LORA" \
    --lora_rank 32 --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
    --num_inference_steps 1 \
    --recent_augment_mode symmetric \
    --dfake_gen_update_ratio 5 --flow_shift 5.0 \
    --learning_rate_student 5e-6 --learning_rate_critic 5e-6 \
    --ema_start_step 999999999 \
    --gan_g_weight 0.03 --gan_d_weight 0.03 \
    --gan_feature_layers "13,21,29" --gan_ffn_dim 4096 \
    --highfreq_weight 1.0 \
    --hf_vae_chunk 4 \
    --num_epochs 5 --save_steps 50 \
    --global_step_offset $GLOBAL_STEP_OFFSET \
    --output_path "$OUTPUT_DIR" --use_gradient_checkpointing \
    $RESUME_ARGS
