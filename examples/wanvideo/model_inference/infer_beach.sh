#!/bin/bash
# 海边场景 10min 横评:v2 / asym / sym / control 同步 4 卡并行
# Logs → ./logs/beach_<exp>.log(脚本所在目录的 logs/ 子目录)
#
# 改 STEP / PROMPT / REF 就能跑别的场景或别的 checkpoint。

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd /mnt/vita/scratch/vita-students/users/jinghao/code/DiffSynth-Studio

LOGDIR="$SCRIPT_DIR/logs"
mkdir -p "$LOGDIR"

LONG=examples/wanvideo/model_inference/run_long_chain.sh
SK_V2=models/train/Wan2.1-Fun-V1.1-1.3B-Control_lora_cartoon_sink_v2/step-1745.safetensors
SK_NOAUG=models/train/Wan2.1-Fun-V1.1-1.3B-Control_lora_cartoon_sink_noaug/step-1745.safetensors

# ─── 场景配置(改这块就换场景)───
DURATION=${DURATION:-600}
STEPS=${STEPS:-4}
REF=${REF:-asset/8.png}
POSE=${POSE:-asset/pose_loop_600s.mp4}
TAG=${TAG:-beach}
PROMPT=${PROMPT:-"动漫风格，紫色长发少女正在海边轻盈起舞。她身穿白色无袖连衣裙，胸前系着深蓝色蝴蝶结，发丝随风飘动。背景是蔚蓝的大海与天空，海岸边盛开着粉色花朵，画面明亮清新。"}

# ─── checkpoint 配置(改 STEP 跑别的)───
STEP_V2=${STEP_V2:-1200}
STEP_NOAUG=${STEP_NOAUG:-1600}

# 没建过对应长度的 pose 就先建
[ -f "$POSE" ] || python asset/make_pose_long.py --duration $DURATION --output "$POSE"

echo "[$(date +%H:%M:%S)] launching 4 jobs:  TAG=$TAG  DUR=${DURATION}s  STEPS=$STEPS"

# === GPU 0:v2(原 SOTA)===
CUDA_VISIBLE_DEVICES=0 bash $LONG --duration $DURATION --steps $STEPS \
    --control_video "$POSE" \
    --initial_ref "$REF" --prompt "$PROMPT" --tag "$TAG" \
    --sink_lora "$SK_V2" \
    --student_dir models/train/Wan2.1-Fun-V1.1-1.3B-Control_lora_cartoon_sink_dmd_v2 \
    --student $STEP_V2 > "$LOGDIR/${TAG}_v2.log" 2>&1 &

# === GPU 1:asym(主结果)===
CUDA_VISIBLE_DEVICES=1 bash $LONG --duration $DURATION --steps $STEPS \
    --control_video "$POSE" \
    --initial_ref "$REF" --prompt "$PROMPT" --tag "$TAG" \
    --sink_lora "$SK_NOAUG" \
    --student_dir models/train/Wan2.1-Fun-V1.1-1.3B-Control_lora_cartoon_dmd_noaugteacher_asym \
    --student $STEP_NOAUG > "$LOGDIR/${TAG}_asym.log" 2>&1 &

# === GPU 2:sym(失败基线)===
CUDA_VISIBLE_DEVICES=2 bash $LONG --duration $DURATION --steps $STEPS \
    --control_video "$POSE" \
    --initial_ref "$REF" --prompt "$PROMPT" --tag "$TAG" \
    --sink_lora "$SK_NOAUG" \
    --student_dir models/train/Wan2.1-Fun-V1.1-1.3B-Control_lora_cartoon_dmd_noaug_sym \
    --student $STEP_NOAUG > "$LOGDIR/${TAG}_sym.log" 2>&1 &

# === GPU 3:control(下界)===
CUDA_VISIBLE_DEVICES=3 bash $LONG --duration $DURATION --steps $STEPS \
    --control_video "$POSE" \
    --initial_ref "$REF" --prompt "$PROMPT" --tag "$TAG" \
    --sink_lora "$SK_NOAUG" \
    --student_dir models/train/Wan2.1-Fun-V1.1-1.3B-Control_lora_cartoon_dmd_noaug_control \
    --student $STEP_NOAUG > "$LOGDIR/${TAG}_control.log" 2>&1 &

echo "monitor:  tail -f $LOGDIR/${TAG}_{v2,asym,sym,control}.log"
wait
echo "[$(date +%H:%M:%S)] done.  outputs in DiffSynth-Studio/samples/dmd_fewstep_validation/"
