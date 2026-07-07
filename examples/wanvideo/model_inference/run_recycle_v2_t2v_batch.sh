#!/bin/bash
# Batch T2V runs for recycle_v2 step-850: two per-GPU queues.
# Jobs inside one GPU run sequentially; the two GPUs run in parallel.
#
# The script stays in the FOREGROUND until all queues finish. Do NOT detach
# with nohup: this node reaps background processes that leave the login
# session (verified 2026-06-12 — nohup'd queues were SIGKILLed ~30s in,
# while an identical foreground run completed). To keep it running with the
# terminal closed, use tmux:
#
#   tmux new -s t2v
#   bash examples/wanvideo/model_inference/run_recycle_v2_t2v_batch.sh
#   # detach: Ctrl-B then D;  reattach later: tmux attach -t t2v
#
#   STEP=950 bash run_recycle_v2_t2v_batch.sh   # other checkpoint
#
# Monitor (from another terminal):
#   tail -f examples/wanvideo/model_inference/logs/batch_gpu0.log
#   tail -f examples/wanvideo/model_inference/logs/batch_gpu1.log
# Outputs: samples/onestep_clarity/01_recycle_t2v/  (filenames carry _t2v + tag)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
T2V="$SCRIPT_DIR/infer_wan1.3b_dmd_recycle_v2_t2v.sh"
LOGDIR="$SCRIPT_DIR/logs"
mkdir -p "$LOGDIR"

STEP=${STEP:-850}

PROMPT_PURPLE="动漫风格，高清二次元插画质感。一位紫色波波头短发的少女正在镜头前轻盈起舞，\
发丝随动作自然飘动，刘海下露出明亮的紫色大眼睛。她头戴黑色细发箍，身穿及膝的白色连衣裙，\
裙摆层叠带蕾丝花边，外搭一件修身的黑色无袖背心，胸前系着一个大大的粉色蝴蝶结。\
她的动作流畅优雅，双臂舒展，裙摆随旋转轻轻扬起。背景是一个柔和的粉色大圆圈，\
周围漂浮着细小的光点，画面简洁干净，色调柔和明亮，光影细腻，线条清晰。"

PROMPT_SAILOR="动漫风格，高清二次元插画质感。一位金色长直发的少女正在镜头前轻盈起舞，\
长发垂至腰间，随舞步柔顺摆动，碧蓝色的大眼睛神采明亮。她身穿经典的深蓝色水手服，\
白色衣领上有两道蓝色条纹，胸前系着鲜红色的领结，蓝色百褶短裙随旋转展开，\
脚上是白色及膝袜和棕色小皮鞋。她的舞姿轻快活泼，脚尖轻点，裙摆飞扬。\
背景是一个温暖的鹅黄色大圆圈，点缀着零星的星星图案，画面简洁柔和，\
光线均匀通透，细节丰富，线条干净利落。"

PROMPT_KIMONO="动漫风格，高清二次元插画质感。一位黑色双马尾的少女正在镜头前轻盈起舞，\
两束长马尾用金色铃铛发饰束起，随动作左右摇曳，深红色的眼眸温柔明亮。\
她身穿绣有白色鹤纹的绯红色和服，宽大的振袖随手臂动作如蝶翼般展开，\
腰间系着金色织锦腰带，打成精致的太鼓结，足踏白色足袋和红色木屐。\
她的舞姿端庄优雅，动作舒缓连贯，衣袖翻飞间若隐若现。背景是一个淡雅的浅绿色大圆圈，\
飘落着几片粉色樱花花瓣，画面简洁唯美，色彩和谐，光影柔和，线条精致。"

# GPU 0: detailed purple-girl prompt, 180s then 600s (long-chain drift check)
bash -c "
    GPU=0 STEP=$STEP DURATION=180 TAG=purple PROMPT='$PROMPT_PURPLE' bash '$T2V'
    GPU=0 STEP=$STEP DURATION=600 TAG=purple PROMPT='$PROMPT_PURPLE' bash '$T2V'
" > "$LOGDIR/batch_gpu0.log" 2>&1 &
PID0=$!

# GPU 1: two new outfits at 180s (prompt-generalization check)
bash -c "
    GPU=1 STEP=$STEP DURATION=180 TAG=sailor PROMPT='$PROMPT_SAILOR' bash '$T2V'
    GPU=1 STEP=$STEP DURATION=180 TAG=kimono PROMPT='$PROMPT_KIMONO' bash '$T2V'
" > "$LOGDIR/batch_gpu1.log" 2>&1 &
PID1=$!

echo "running: GPU0 queue pid=$PID0  GPU1 queue pid=$PID1  (step-$STEP)"
echo "logs:    $LOGDIR/batch_gpu0.log  $LOGDIR/batch_gpu1.log"
echo "abort:   Ctrl-C here, or: kill $PID0 $PID1"
echo "waiting for both queues to finish (keep this terminal / tmux session alive)..."
wait $PID0 $PID1
echo "[$(date +%H:%M:%S)] all queues done"
