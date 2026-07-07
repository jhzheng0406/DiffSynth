# LTX-2 Stage B student 1-step 链推理(pod 内,单卡)— Wan FewStep 脚本的 LTX 版
#
#   GPU=0 bash examples/ltx2/model_inference/run_chain_studentB.sh                # 最新 student,8 chunk
#   GPU=0 STEP=400 bash ...                                                       # 指定 student step
#   GPU=0 NUM_CHUNKS=90 bash ...                                                  # 180s 长链
#   GPU=0 TEACHER_STEP=1740 bash ...                                              # 指定 teacher ckpt
#
# LoRA 叠放(必须与训练一致):base + IC-LoRA(control)+ Stage-A v3 teacher
# + Stage-B student,1-NFE/chunk。student 蒸在哪个 teacher 上就 fuse 哪个
# (对应 Wan 侧 "各 fuse 各自 sink" 的公平对比规则)。

set -e
cd /mnt/vita/scratch/vita-students/users/jinghao/code/DiffSynth-Studio

GPU=${GPU:-0}
NUM_CHUNKS=${NUM_CHUNKS:-8}
TEACHER_DIR=${TEACHER_DIR:-models/train/LTX-2.3-22b_lora_cartoon_sink_recycle_v6}
STUDENT_DIR=${STUDENT_DIR:-models/train/ltx23_22b_dmd_recycle_v1}
CONTROL_VIDEO=${CONTROL_VIDEO:-asset/pose_loop.mp4}
PROMPT="[VISUAL]:Anime style. A girl with short purple hair is dancing lightly. She wears a black headband, a white dress with a black vest, and a pink bow on her chest. The background is a large pink circle, simple and soft. [SOUNDS]:soft background music"

pick_latest() {  # pick_latest DIR [STEP]
    if [ -n "$2" ]; then echo "$1/step-$2.safetensors"; return; fi
    local latest=$(ls "$1"/step-*.safetensors 2>/dev/null | grep -v "_critic\|_ema\|_cls\|_state" \
                   | sed -n 's/.*step-\([0-9]\+\)\.safetensors/\1/p' | sort -n | tail -1)
    [ -z "$latest" ] && { echo ""; return; }
    echo "$1/step-${latest}.safetensors"
}

TEACHER_LORA=$(pick_latest "$TEACHER_DIR" "$TEACHER_STEP")
STUDENT_LORA=$(pick_latest "$STUDENT_DIR" "$STEP")
[ -f "$TEACHER_LORA" ] || { echo "[error] teacher LoRA not found in $TEACHER_DIR"; exit 1; }
[ -f "$STUDENT_LORA" ] || { echo "[error] student LoRA not found in $STUDENT_DIR"; exit 1; }
echo "[student-1step] teacher=$TEACHER_LORA"
echo "[student-1step] student=$STUDENT_LORA"
TAG="$(basename "$STUDENT_DIR")_$(basename "$STUDENT_LORA" .safetensors)"

mkdir -p outputs
CUDA_VISIBLE_DEVICES=$GPU python examples/ltx2/model_inference/LTX-2-Chunk-Chain-Sanity.py \
    --initial_ref asset/6.png \
    --prompt "$PROMPT" \
    --num_chunks $NUM_CHUNKS \
    --chunk_frames 49 \
    --height 832 --width 480 \
    --steps 1 --cfg 1.0 --seed 42 \
    --sink_index 1 --time_shift_frames 48 \
    --control_video "$CONTROL_VIDEO" \
    --base 2.3 --ic_lora "models/Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors" \
    --lora_path "$TEACHER_LORA,$STUDENT_LORA" \
    --output "outputs/ltx2_studentB_${TAG}_${NUM_CHUNKS}ch_1step.mp4"
