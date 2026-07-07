# DMD + MMD (RDM-lite)  v2:  causal-WINDOW grad decode — fixes the OOM/starvation
# of the v1 runs (tileckpt & friends).
#
# What changed vs v1_49f_8gpu_tileckpt:
#   - decode_window (wan_video_vae.py): no_grad prefix primes the causal cache,
#     only fd_window_latents=4 latents decode WITH grad at a random position.
#     Window pixels are BIT-IDENTICAL to the full decode (verified diff=0.0),
#     so the reference-stats domain still matches; grad memory scales with the
#     window, not the 49-frame chunk.
#   - No FD-dedicated GPUs (fd_device_offset gone) → 8 training ranks, not 4.
#   - No tiled grad decode (tiling never reduced the BACKWARD peak anyway).
#   - fd_num_frames 2 → 16: every step now has 16×8 = 128 MMD samples instead
#     of 8 (repulsion term was pure noise at 8).
#   - ×num_proc gradient compensation restored: fd_weight now truly means its
#     nominal value at any world size (v1 runs effectively trained at w/4).
#   - dfake_gen_update_ratio back to 5 (1 was a debug setting).
#
# Tuning headroom: if step-1/2 memory prints show plenty free, raise
# FD_WINDOW to 6 and FD_FRAMES to 24 (→192 samples/step).

TEACHER_LORA="./models/train/Wan2.1-Fun-V1.1-1.3B-Control_lora_cartoon_sink_recycle_v1/step-875.safetensors"
FD_STATS="./data/cartoon_15s/fd_stats_dinov2_v2.npz"

# mmd (default) = RDM-lite fresh-batch MMD; frechet = FD covariance+queue path
# (queue/min_pop/enqueue args use the parser defaults). Separate output dirs
# so auto-resume never crosses objectives.
FD_OBJECTIVE=${FD_OBJECTIVE:-mmd}
if [ "$FD_OBJECTIVE" = "mmd" ]; then
    OUTPUT_DIR="./models/train/wan1.3b_dmd_mmd_recycleteacher_v2_window"
else
    OUTPUT_DIR="./models/train/wan1.3b_dmd_frechet_recycleteacher_v2_window"
fi

NUM_GPUS=${NUM_GPUS:-8}
MASTER_PORT=${MASTER_PORT:-29548}
FD_WEIGHT=${FD_WEIGHT:-0.02}
FD_WINDOW=${FD_WINDOW:-4}
FD_FRAMES=${FD_FRAMES:-16}

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
[ -f "$TEACHER_LORA" ] || { echo "[error] teacher LoRA missing: $TEACHER_LORA"; exit 1; }

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Persistent torch.hub cache (pod HOME=/home/runai-home is ephemeral →
# every fresh pod re-downloads DINOv2 and multi-rank races on the extract).
export TORCH_HOME=${TORCH_HOME:-/home/jzheng/.cache/torch}

python -m torch.distributed.run \
    --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT \
    examples/wanvideo/model_training/train_dmd_fd_v4.py \
    --dataset_metadata_path ./data/cartoon_15s/metadata.csv \
    --height 832 --width 480 --num_frames 49 --dataset_repeat 1 \
    --recent_aug_strength 0.5 \
    --teacher_lora_path "$TEACHER_LORA" \
    --lora_rank 32 --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
    --num_inference_steps 1 \
    --recent_augment_mode symmetric \
    --dfake_gen_update_ratio 5 \
    --flow_shift 5.0 \
    --learning_rate_student 5e-6 --learning_rate_critic 5e-6 \
    --ema_start_step 999999999 \
    --fd_stats_path "$FD_STATS" \
    --fd_objective "$FD_OBJECTIVE" \
    --fd_weight $FD_WEIGHT \
    --fd_feature_model dinov2_vitb14 \
    --fd_window_latents $FD_WINDOW \
    --fd_num_frames $FD_FRAMES \
    --fd_mmd_max_ref_features 20000 \
    --fd_mmd_ref_chunk 4096 \
    --num_epochs 5 --save_steps 50 \
    --global_step_offset $GLOBAL_STEP_OFFSET \
    --output_path "$OUTPUT_DIR" \
    --use_gradient_checkpointing \
    --memory_debug --memory_debug_steps 2 \
    $RESUME_ARGS
