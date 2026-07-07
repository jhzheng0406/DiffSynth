# DMD + One-Forcing GAN + TEACHER TRAJECTORY ANCHOR (1-step, Progressive-Distill style).
#
#   teacher  = base + sink_v2 LoRA (frozen): DMD real score AND the anchor target
#              (teacher K-step ODE from the student's OWN initial noise)
#   student  = base + sink_v2 + LoRA (trained, 1-step rollout)
#   critic   = base + sink_v2 + LoRA (DMD critic + cls_branch GAN-D)
#
# NEW vs oneforcing: anchor_loss = L1(x_pred, teacher_Kstep(same_init_noise)).
#   Targets the measured ~20% sharpness gap (teacher 50-step > student 1-step).
#   Inference UNCHANGED (1-NFE) — detail is distilled into student weights.
#
# Key knobs (see train_dmd_refine.py docstring):
#   --anchor_weight 0.25     auxiliary weight (DMD stays main loss)
#   --anchor_steps 4         teacher ODE steps for the target
#   --anchor_highfreq        match only x-blur(x) (detail), DMD owns structure
#   --anchor_start_step 200  skip early (rough x_pred)
#
# ABLATION on top of oneforcing — set --anchor_weight 0 to recover baseline.

OUTPUT_DIR="./models/train/wan1.3b_dmd_refine_v1"
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
    examples/wanvideo/model_training/train_dmd_refine.py \
    --dataset_metadata_path ./data/cartoon_15s/metadata.csv \
    --height 832 --width 480 --num_frames 49 --dataset_repeat 1 \
    --recent_aug_strength 0.5 \
    --teacher_lora_path "$TEACHER_LORA" \
    --lora_rank 32 --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
    --num_inference_steps 1 \
    --recent_augment_mode symmetric \
    --dfake_gen_update_ratio 5 --flow_shift 5.0 \
    --learning_rate_student 2e-6 --learning_rate_critic 4e-7 \
    --ema_start_step 999999999 \
    --gan_g_weight 0.03 --gan_d_weight 0.03 \
    --gan_feature_layers "13,21,29" --gan_ffn_dim 4096 \
    --anchor_weight 0.25 \
    --anchor_steps 4 \
    --anchor_start_step 200 \
    --anchor_highfreq \
    --anchor_hf_kernel 3 \
    --anchor_cfg 0 \
    --num_epochs 5 --save_steps 50 \
    --global_step_offset $GLOBAL_STEP_OFFSET \
    --output_path "$OUTPUT_DIR" --use_gradient_checkpointing \
    $RESUME_ARGS
