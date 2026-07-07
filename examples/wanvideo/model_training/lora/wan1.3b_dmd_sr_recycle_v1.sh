# DMD + One-Forcing (cls_branch GAN) + SVI-style Error Recycle + Self-Correcting Loss.
#
# Per step: TWO student rollouts SHARING the same initial noise + exit_idx,
# differing only in reference_latents (clean vs corrupt). Then DMD/GAN/sc loss.
#   1. CLEAN rollout (no grad):   x_pred_clean = student(noise, ref_clean)
#   2. INJECT error:               ref_corrupt = ref_clean + α·sampled_drift
#   3. CORRUPT rollout (grad):     x_pred = student(noise, ref_corrupt)
#      ★ same noise / exit_idx as clean rollout ★
#   4. critic + GAN-D unchanged
#   5. gen_loss = dmd_g + 0.03·gan_g + sc_weight·L1(x_pred, x_pred_clean.detach())
#   6. push (x_pred_clean_last - real_last) into buffer    ← drift direction = student - real
#
# SVI defaults (svi_shot.sh): buffer_k=500, warmup_iter=50, y_prob=0.9, clean_prob=0.2.
# We do deterministic injection (no clean_prob) since DMD already provides clean
# regularization via teacher-anchoring.
#
# Cost vs oneforcing: 2× student rollout (no grad one is cheaper, no critic backward).
# Expected ~1.3-1.5× slower per step.

OUTPUT_DIR="./models/train/wan1.3b_dmd_sr_recycle_v1"
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

accelerate launch \
    --num_processes=$NUM_GPUS --mixed_precision=bf16 \
    examples/wanvideo/model_training/train_dmd_sr_recycle.py \
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
    --sr_amount 0.5 --sr_sigma 1.2 \
    --error_alpha 0.25 --error_buffer_size 500 \
    --error_warmup_count 50 --error_alpha_ramp_steps 200 \
    --error_inject_prob 0.8 \
    --error_collect_start_step 100 \
    --sc_weight 0.5 \
    --num_epochs 5 --save_steps 50 \
    --global_step_offset $GLOBAL_STEP_OFFSET \
    --output_path "$OUTPUT_DIR" --use_gradient_checkpointing \
    $RESUME_ARGS
