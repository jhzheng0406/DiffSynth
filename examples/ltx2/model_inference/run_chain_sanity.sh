# LTX-2 chaining sanity — 三档速查(pod 内跑,单卡)
#
#   GPU=0 bash examples/ltx2/model_inference/run_chain_sanity.sh            # no-sink(默认)
#   GPU=0 ARM=sink49 bash examples/ltx2/model_inference/run_chain_sanity.sh # sink_index=-49
#   GPU=0 ARM=sink8  bash examples/ltx2/model_inference/run_chain_sanity.sh # sink_index=-8
#   GPU=0 ARM=long   bash examples/ltx2/model_inference/run_chain_sanity.sh # no-sink × 20 chunk
#                      (≈40s@24fps:看无 recycle 的 drift 基线——4 chunk 太短,
#                       只能验映射验不出漂移;泛化实验需要确认 LTX-2 上有靶子)
#
# 结果速记(2026-06-11/12 已跑):nosink 4ch 稳 ✅;sink-8 第3-4 chunk 洗白 ❌;
# sink-49 那次没传 --initial_ref(默认 prompt,不可比),ARM=sink49 是同图重跑。
#
# 看什么:
#   no-sink  = 纯 recent 链(LTX 官方 i2v 用法,零风险 fallback)→ 看 chunk 间
#              是否续上 + 无 recycle 的漂移基线
#   sink49/8 = 负 index sink token → 对比 no-sink 有没有 artifact / 一致性收益
#
# 注意:首次加载要从 NFS 读 43GB(dit)+ 24GB(gemma),可能 15-30 分钟;
# 生成 4 chunk × 30 step × CFG 双 forward,19B 上预计再 ~20-40 分钟。

set -e
cd /mnt/vita/scratch/vita-students/users/jinghao/code/DiffSynth-Studio

GPU=${GPU:-0}
ARM=${ARM:-nosink}
PROMPT="[VISUAL]:Anime style. A girl with short purple hair is dancing lightly. She wears a black headband, a white dress with a black vest, and a pink bow on her chest. The background is a large pink circle, simple and soft. [SOUNDS]:soft background music"

NUM_CHUNKS=${NUM_CHUNKS:-4}
case "$ARM" in
    nosink) EXTRA="--no_sink";        OUT="ltx2_chain_sanity_nosink.mp4" ;;
    sink49) EXTRA="--sink_index -49"; OUT="ltx2_chain_sanity_sink-49.mp4" ;;
    sink8)  EXTRA="--sink_index -8";  OUT="ltx2_chain_sanity_sink-8.mp4" ;;
    long)   EXTRA="--no_sink"; NUM_CHUNKS=20; OUT="ltx2_chain_sanity_nosink_20ch.mp4" ;;
    # zero-shot gate for the pose-control route: does union-control IC-LoRA
    # follow CARTOON OPENPOSE at all (official example was real-domain depth)?
    # First run downloads the IC-LoRA weights from modelscope.
    control) EXTRA="--no_sink --control_video data/cartoon_15s/poses/000000_poses.mp4"; OUT="ltx2_chain_sanity_control.mp4" ;;
    *) echo "[error] ARM must be nosink|sink49|sink8|long|control"; exit 1 ;;
esac

mkdir -p outputs
echo "[sanity] ARM=$ARM GPU=$GPU → $OUT"
CUDA_VISIBLE_DEVICES=$GPU python examples/ltx2/model_inference/LTX-2-Chunk-Chain-Sanity.py \
    --initial_ref asset/6.png \
    --prompt "$PROMPT" \
    --num_chunks $NUM_CHUNKS \
    --chunk_frames 49 \
    --height 832 --width 480 \
    --steps 30 --cfg 3.0 --seed 42 \
    $EXTRA \
    --output "outputs/$OUT"
