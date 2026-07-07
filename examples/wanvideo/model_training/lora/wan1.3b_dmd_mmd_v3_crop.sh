# DMD + MMD  v3:  crop-view experiments (fixes the blur-blind feature space).
#
# Diagnosis of v2_window ("还是好糊"): the whole-frame path squash-resizes
# 832x480 → 224² (3.7x low-pass) before DINOv2-cls — the feature space barely
# sees blur, so MMD converged (fd_raw -0.5 → -0.57, floor -0.61) without
# sharpening anything. Crop view = K random 224px crops at NATIVE resolution:
# high frequencies stay visible AND samples/step go 128 → 512 (toward iRDM's
# working batch regime; their N=512 arm at true-independent samples was the
# minimum that didn't regress).
#
# Two experiments (user-requested A/B):
#   MODE=crop  (default): crop view ONLY          (fd_weight=0, crop_weight)
#   MODE=dual           : global + crop two-view  (both weights; the mini
#                         "encoder battery" — global keeps layout stats,
#                         crop brings texture/blur sensitivity)
#
# PREREQ (one-time): crop-protocol reference stats
#   python examples/wanvideo/model_training/precompute_fd_stats.py \
#       --metadata_path ./data/cartoon_15s/metadata.csv \
#       --output ./data/cartoon_15s/fd_stats_dinov2_crop.npz \
#       --crops_per_frame 4 --max_frames 60000
#
# Usage:
#   bash wan1.3b_dmd_mmd_v3_crop.sh                  # crop-only
#   MODE=dual bash wan1.3b_dmd_mmd_v3_crop.sh        # global + crop

TEACHER_LORA="./models/train/Wan2.1-Fun-V1.1-1.3B-Control_lora_cartoon_sink_recycle_v1/step-875.safetensors"
FD_STATS="./data/cartoon_15s/fd_stats_dinov2_v2.npz"
FD_CROP_STATS="./data/cartoon_15s/fd_stats_dinov2_crop.npz"

MODE=${MODE:-crop}
NUM_GPUS=${NUM_GPUS:-8}
MASTER_PORT=${MASTER_PORT:-29549}
FD_CROP_WEIGHT=${FD_CROP_WEIGHT:-0.05}
# 0.5x the bank's median-heuristic bandwidth (55.07 → 27.5): blur-sensitivity
# diagnostic (2026-07-06) showed MMD^2 contrast between clean and blurred
# crops PEAKS at 0.5x (sigma=3: 0.053 @ 1x → 0.081 @ 0.5x; 0.25x saturates).
# 0 = use the bandwidth saved in the stats file (1x median).
FD_CROP_BW=${FD_CROP_BW:-27.5}
FD_CROPS=${FD_CROPS:-4}
FD_WINDOW=${FD_WINDOW:-4}
FD_FRAMES=${FD_FRAMES:-16}
SAVE_STEPS=${SAVE_STEPS:-50}
# Tail-biased window sampling (P(j) ∝ 1 + bias·j/J): concentrates FD/MMD
# gradient on late-chunk frames where 1-NFE blur accumulates. 0 = uniform.
FD_TAIL_BIAS=${FD_TAIL_BIAS:-0.0}

if [ "$MODE" = "crop" ]; then
    FD_WEIGHT=${FD_WEIGHT:-0.0}
    OUTPUT_DIR="./models/train/wan1.3b_dmd_mmd_v3_croponly"
elif [ "$MODE" = "dual" ]; then
    FD_WEIGHT=${FD_WEIGHT:-0.02}
    OUTPUT_DIR="./models/train/wan1.3b_dmd_mmd_v3_dual"
else
    echo "[error] MODE must be 'crop' or 'dual', got: $MODE"; exit 1
fi
# Separate dir per bias setting so auto-resume never mixes arms (string
# match, no bc dependency in the pod image).
case "$FD_TAIL_BIAS" in
    0|0.0|"") ;;
    *) OUTPUT_DIR="${OUTPUT_DIR}_tail${FD_TAIL_BIAS}";;
esac

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

[ -f "$FD_CROP_STATS" ] || { echo "[error] crop stats missing: $FD_CROP_STATS — run the PREREQ in this file's header"; exit 1; }
# global stats only needed when the global term is active (dual mode)
if [ "$MODE" = "dual" ]; then
    [ -f "$FD_STATS" ] || { echo "[error] FD stats missing: $FD_STATS"; exit 1; }
fi
[ -f "$TEACHER_LORA" ] || { echo "[error] teacher LoRA missing: $TEACHER_LORA"; exit 1; }

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Persistent torch.hub cache (pod HOME=/home/runai-home is ephemeral →
# every fresh pod re-downloads DINOv2 and multi-rank races on the extract).
export TORCH_HOME=${TORCH_HOME:-/home/jzheng/.cache/torch}

echo "[run] MODE=$MODE  fd_weight=$FD_WEIGHT  crop_weight=$FD_CROP_WEIGHT  → $OUTPUT_DIR"

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
    --fd_objective mmd \
    --fd_weight $FD_WEIGHT \
    --fd_crop_stats_path "$FD_CROP_STATS" \
    --fd_crop_weight $FD_CROP_WEIGHT \
    --fd_crop_mmd_bandwidth $FD_CROP_BW \
    --fd_crops_per_frame $FD_CROPS \
    --fd_feature_model dinov2_vitb14 \
    --fd_window_latents $FD_WINDOW \
    --fd_window_tail_bias $FD_TAIL_BIAS \
    --fd_num_frames $FD_FRAMES \
    --fd_mmd_max_ref_features 20000 \
    --fd_mmd_ref_chunk 4096 \
    --num_epochs 5 --save_steps $SAVE_STEPS \
    --global_step_offset $GLOBAL_STEP_OFFSET \
    --output_path "$OUTPUT_DIR" \
    --use_gradient_checkpointing \
    --memory_debug --memory_debug_steps 2 \
    $RESUME_ARGS
