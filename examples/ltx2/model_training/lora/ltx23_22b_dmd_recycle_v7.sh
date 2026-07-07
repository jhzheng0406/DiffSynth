# LTX-2.3 22B DMD student + SVI recycle (Stage B) — cross-architecture port of
# wan1.3b_dmd_recycle_v2.sh.
#
# v7 (2026-06-16): DIVERGENCE FIX. v6 blew up — err_norm grew monotonically
# (0.29→0.46→0.63→0.71→3.98→5.84) and dmd_g exploded (-0.18→+0.22→+9.06) right
# as the error_alpha ramp hit max (~step 350: collect_start 100 + warmup 50 +
# ramp 200). The recycle feedback loop (student error → buffer → injection →
# more error) went unstable on 22B under full-strength injection. Fixes:
#   - learning_rate 5e-6 → 2e-6   (slower student, less runaway)
#   - error_alpha 0.25 → 0.1      (weaker injection, weaker feedback)
#   - error_alpha_ramp 200 → 400  (more gradual onset, no sudden full strength)
# Last HEALTHY v6 ckpt was step-300; 400+ degrading, 700+ fully diverged.
# (No grad-clip in the script — if v7 still drifts, add clip_grad_norm_.)
#
# PREREQUISITES (in order — do not skip):
#   1. Chaining sanity passed (LTX-2-Chunk-Chain-Sanity.py, sink_index 三档).
#   2. Stage A teacher trained: ltx23_22b_cartoon_sink_recycle_v6.sh
#      (this script auto-picks its latest step-N checkpoint).
#
# Memory: shared 19B base + student/critic LoRA adapters (teacher = adapters
# off) ≈ 38 GB dit + 24 GB gemma + activations ≈ 70-80 GB / GPU. Fits H200.
# --offload_text_encoder moves gemma to CPU (cache-miss shuttling only) if
# you need the headroom.
#
# Speed: ~17 19B forwards per step (rollouts + 5×critic + 5×GAN-D ×2-batch +
# DMD scores + GAN-G) — expect minutes-per-step territory, an order of
# magnitude slower than the 1.3B Wan runs. Budget accordingly.
#
# Usage:
#   NUM_GPUS=8 bash <this-script>
#   TEACHER_STEP=800 bash <this-script>
#   RESUME_STUDENT=none bash <this-script>     # force fresh start

OUTPUT_DIR="./models/train/ltx23_22b_dmd_recycle_v7"
TEACHER_DIR="./models/train/LTX-2.3-22b_lora_cartoon_sink_recycle_v6"

# Resolve teacher LoRA: TEACHER_LORA > TEACHER_STEP > auto-pick latest
if [ -z "$TEACHER_LORA" ]; then
    if [ -n "$TEACHER_STEP" ]; then
        if [[ "$TEACHER_STEP" == epoch-* ]]; then
            TEACHER_LORA="$TEACHER_DIR/${TEACHER_STEP}.safetensors"
        else
            TEACHER_LORA="$TEACHER_DIR/step-${TEACHER_STEP}.safetensors"
        fi
    else
        LATEST_T=$(ls "$TEACHER_DIR"/step-*.safetensors 2>/dev/null \
                   | sed -n 's/.*step-\([0-9]\+\)\.safetensors/\1/p' | sort -n | tail -1)
        [ -z "$LATEST_T" ] && { echo "[error] no step-*.safetensors in $TEACHER_DIR — train Stage A first"; exit 1; }
        TEACHER_LORA="$TEACHER_DIR/step-${LATEST_T}.safetensors"
        echo "[auto-teacher] latest → step-${LATEST_T}"
    fi
fi
[ -f "$TEACHER_LORA" ] || { echo "[error] TEACHER_LORA not found: $TEACHER_LORA"; exit 1; }
echo "[teacher] $TEACHER_LORA"

RESUME_STUDENT=${RESUME_STUDENT:-}
RESUME_CRITIC=${RESUME_CRITIC:-}

if [ -z "$RESUME_STUDENT" ] && [ -d "$OUTPUT_DIR" ]; then
    LATEST=$(ls "$OUTPUT_DIR"/step-*.safetensors 2>/dev/null | grep -v "_critic\|_ema" \
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
    examples/ltx2/model_training/train_dmd_recycle_ltx2.py \
    --dataset_metadata_path ./data/cartoon_15s/metadata.csv \
    --height 832 --width 480 --num_frames 49 --frame_rate 24 --dataset_repeat 1 \
    --recent_aug_strength 0.5 \
    --model_paths '["models/Lightricks/LTX-2.3/ltx-2.3-22b-dev.safetensors"]' \
    --teacher_lora_path "$TEACHER_LORA" \
    --lora_rank 32 --lora_target_modules "to_k,to_q,to_v,to_out.0,net.0.proj,net.2" \
    --sink_index 1 --time_shift_frames 48 --input_images_strength 1.0 \
    --use_pose_control \
    --ic_lora_path "models/Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors" \
    --num_inference_steps 1 \
    --recent_augment_mode symmetric \
    --dfake_gen_update_ratio 5 \
    --learning_rate_student 2e-6 --learning_rate_critic 2e-6 \
    --ema_start_step 999999999 \
    --gan_g_weight 0.03 --gan_d_weight 0.03 \
    --gan_feature_layers "21,34,47" --gan_ffn_dim 4096 --gan_num_heads 32 \
    --error_alpha 0.1 --error_buffer_size 500 \
    --error_warmup_count 50 --error_alpha_ramp_steps 400 \
    --error_inject_prob 0.8 \
    --error_collect_start_step 100 \
    --sc_weight 0.5 \
    --offload_text_encoder \
    --num_epochs 5 --save_steps 100 \
    --global_step_offset $GLOBAL_STEP_OFFSET \
    --output_path "$OUTPUT_DIR" --use_gradient_checkpointing \
    $RESUME_ARGS
