# LTX-2 Stage A teacher 链推理评估(pod 内,单卡)
#
#   GPU=0 bash examples/ltx2/model_inference/run_chain_stageA.sh                 # 最新 ckpt,8 chunk
#   GPU=0 STEP=400 bash ...                                                      # 指定 step
#   GPU=0 NUM_CHUNKS=20 bash ...                                                 # 长链看一致性
#   GPU=0 LORA_DIR=models/train/LTX-2-19b_lora_cartoon_sink_v1 bash ...          # 评 no-recycle 对照
#
# 看什么(对照 zero-shot sanity 的结果):
#   1. 域:卡通风格是否回来了(base 是 2.5D 半写实 → LoRA 应拉回平面卡通)
#   2. sink:-49 的 token 训后是否从"轻微有害"变"有益"(对比 zero-shot sink-49)
#   3. 跨 chunk 一致性 vs zero-shot nosink 基线
# 注意:这是 50/30-step teacher 评估,不是 1-step;Stage B 之后才有 1-step 链。

set -e
cd /mnt/vita/scratch/vita-students/users/jinghao/code/DiffSynth-Studio

GPU=${GPU:-0}
NUM_CHUNKS=${NUM_CHUNKS:-8}
# 50 = 对齐 Wan 侧 teacher 评估约定(infer_teacher_* 全是 50 NFE);
# LTX-2 官方管线默认是 30,STEPS=30 可切回省时间。
STEPS=${STEPS:-50}
# NO_SINK=1:同一 LoRA 推理时去掉 sink token(负 RoPE 运动扰动的 A/B)
if [ "${NO_SINK:-0}" = "1" ]; then SINK_ARG="--no_sink"; SINK_TAG="nosink"; else SINK_ARG="--sink_index 1 --time_shift_frames 48"; SINK_TAG="shift48"; fi
# CONTROL=1(default for v5): pose 驱动；CONTROL_VIDEO/IC_LORA 可替换。
IC_LORA=${IC_LORA:-models/Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors}
if [ "${CONTROL:-1}" = "1" ]; then
    CTRL_ARG="--control_video ${CONTROL_VIDEO:-asset/pose_loop.mp4} --ic_lora $IC_LORA"; SINK_TAG="${SINK_TAG}_ctrl"
else CTRL_ARG=""; fi
LORA_DIR=${LORA_DIR:-models/train/LTX-2.3-22b_lora_cartoon_sink_recycle_v6}
PROMPT="[VISUAL]:Anime style. A girl with short purple hair is dancing lightly. She wears a black headband, a white dress with a black vest, and a pink bow on her chest. The background is a large pink circle, simple and soft. [SOUNDS]:soft background music"

# 选 ckpt:STEP 指定 > 自动捡最新 step-N
if [ -n "$STEP" ]; then
    LORA="$LORA_DIR/step-${STEP}.safetensors"
else
    LATEST=$(ls "$LORA_DIR"/step-*.safetensors 2>/dev/null \
             | sed -n 's/.*step-\([0-9]\+\)\.safetensors/\1/p' | sort -n | tail -1)
    [ -z "$LATEST" ] && { echo "[error] no step-*.safetensors in $LORA_DIR"; exit 1; }
    LORA="$LORA_DIR/step-${LATEST}.safetensors"
fi
[ -f "$LORA" ] || { echo "[error] not found: $LORA"; exit 1; }
TAG=$(basename "$LORA_DIR" | sed 's/LTX-2-19b_lora_cartoon_//')_$(basename "$LORA" .safetensors)

mkdir -p outputs
echo "[stageA-eval] lora=$LORA chunks=$NUM_CHUNKS GPU=$GPU"
CUDA_VISIBLE_DEVICES=$GPU python examples/ltx2/model_inference/LTX-2-Chunk-Chain-Sanity.py \
    --base 2.3 \
    --initial_ref asset/6.png \
    --prompt "$PROMPT" \
    --num_chunks $NUM_CHUNKS \
    --chunk_frames 49 \
    --height 832 --width 480 \
    --steps $STEPS --cfg 3.0 --seed 42 \
    $SINK_ARG \
    $CTRL_ARG \
    --lora_path "$LORA" \
    --output "outputs/ltx2_stageA_${TAG}_${NUM_CHUNKS}ch_${STEPS}step_${SINK_TAG}_control_${CTRL_TAG}.mp4"
