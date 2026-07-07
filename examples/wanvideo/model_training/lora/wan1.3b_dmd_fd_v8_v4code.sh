# DMD + FD-loss  v8:  first run on CODE v4 (train_dmd_fd_v4.py).
#
# Same hyperparams as v7 (fd_weight=0.05, fr=16, lr 5e-6/5e-6, 1-NFE) so the
# comparison isolates the v4 protocol fixes:
#   - full-chunk causal VAE decode, frame subsample moved to pixel space
#     (v5-v7 all ran with the --fd_vae_chunk causality break → their FD was
#     partly measuring decode-protocol mismatch, not sample quality)
#   - queue warm-up gate (fd_min_pop=1536): no FD gradient until the pooled
#     covariance is full-rank; queue fills from the same gated rollout, so
#     cold-start and steady-state populations are the same distribution
#   - first-step-only FD gate (no-op here: num_inference_steps=1)
#   - eigenvalue clamp 1e-6 (kills exploding 1/sqrt(λ) grads near singularity)
#   - enqueue thinned to 8/chunk → queue spans ~2x more distinct videos
#
# PREREQ: regenerate diverse reference stats first (one-time, ~minutes).
# The current precompute also stores a feature bank for FD_OBJECTIVE=mmd.
#   python examples/wanvideo/model_training/precompute_fd_stats.py \
#       --metadata_path ./data/cartoon_15s/metadata.csv \
#       --output ./data/cartoon_15s/fd_stats_dinov2_v2.npz
#
# Reference stats stay REAL DATA: FD is the GAN substitute here — it injects
# the real-data distribution signal beyond the teacher ceiling (One-Forcing's
# GAN rationale), while DMD handles mode-seeking toward the teacher.

OUTPUT_DIR="./models/train/wan1.3b_dmd_fd_v8_v4code"
TEACHER_LORA="./models/train/Wan2.1-Fun-V1.1-1.3B-Control_lora_cartoon_sink_v2/step-1745.safetensors"
FD_STATS="./data/cartoon_15s/fd_stats_dinov2_v2.npz"
# frechet = original FD covariance+queue path; mmd = RDM-lite fresh-batch MMD.
FD_OBJECTIVE=${FD_OBJECTIVE:-frechet}

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

[ -f "$FD_STATS" ] || { echo "[error] FD stats missing: $FD_STATS — run precompute_fd_stats.py (see header)"; exit 1; }

NUM_GPUS=${NUM_GPUS:-8}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

accelerate launch \
    --num_processes=$NUM_GPUS --mixed_precision=bf16 \
    examples/wanvideo/model_training/train_dmd_fd_v4.py \
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
    --fd_weight 0.05 \
    --fd_feature_model dinov2_vitb14 \
    --fd_objective "$FD_OBJECTIVE" \
    --fd_queue_size 5000 \
    --fd_num_frames 16 \
    --fd_enqueue_per_chunk 8 \
    --fd_min_pop 1536 \
    --fd_eval_clamp 1e-6 \
    --num_epochs 5 --save_steps 50 \
    --global_step_offset $GLOBAL_STEP_OFFSET \
    --output_path "$OUTPUT_DIR" --use_gradient_checkpointing \
    $RESUME_ARGS
