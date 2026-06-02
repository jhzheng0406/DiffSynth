# DMD + FD-loss  v6:  fd_num_frames=16  +  fd_weight 0.03 → 0.1
#
# Goal: directly increase FD pull, since v3/v4/v5 results are still blurry.
# Hypothesis: with fd_weight=0.03 the FD term contributes ~same magnitude as
# dmd_g (~0.02-0.05). With 0.1, FD contributes ~3× dmd_g → meaningfully
# stronger pull toward the DINOv2 reference distribution.
#
# Risk: too much FD can hurt teacher-anchored DMD signal. v2 used 1.0 and was
# bad. 0.1 is a measured step (~3.3× current); if it works keep, if it
# over-pulls back off to 0.05.
#
# Code = train_dmd_fd_v3.py (same as v5, checkpoint-based VAE decode).
# Other hyperparams identical to v5.

OUTPUT_DIR="./models/train/wan1.3b_dmd_fd_v6_fr16_w0.1"
TEACHER_LORA="./models/train/Wan2.1-Fun-V1.1-1.3B-Control_lora_cartoon_sink_v2/step-1745.safetensors"
FD_STATS="./data/cartoon_15s/fd_stats_dinov2.npz"

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

[ -f "$FD_STATS" ] || { echo "[error] FD stats missing: $FD_STATS"; exit 1; }

NUM_GPUS=${NUM_GPUS:-8}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

accelerate launch \
    --num_processes=$NUM_GPUS --mixed_precision=bf16 \
    examples/wanvideo/model_training/train_dmd_fd_v3.py \
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
    --fd_stats_path "$FD_STATS" \
    --fd_weight 0.1 \
    --fd_feature_model dinov2_vitb14 \
    --fd_queue_size 5000 \
    --fd_num_frames 16 \
    --fd_vae_chunk 4 \
    --num_epochs 5 --save_steps 50 \
    --global_step_offset $GLOBAL_STEP_OFFSET \
    --output_path "$OUTPUT_DIR" --use_gradient_checkpointing \
    $RESUME_ARGS
