#!/bin/bash
# Conditioning ablation (sink / recent / aug) — progressively add each, matched
# config, 50-NFE teacher (these are Stage-A teacher LoRAs, NOT 1-step distilled;
# 50-step is their valid mode AND isolates conditioning from 1-step artifacts).
#
# Multi-chunk (drift only shows across chunks). After all 4 run, extracts a LATE
# frame from each + builds a side-by-side grid for slides.
#
#   GPU=0 NUM_CHUNKS=20 bash ablation_conditioning.sh
#
# 5 levels — full sink×recent 2x2 + aug; baseline = NOTHING (pose only):
#   nothing       cartoon_chunked_noref (1600)  --no_recent --no_sink  pose only
#   recent only   cartoon_with_ref      (4975)  --no_sink              recent (i2v chain)
#   sink only     cartoon_sinkonly      (1745)  --no_recent            sink
#   sink+recent   cartoon_sink_noaug    (1745)  (default)              sink + recent
#   +aug          cartoon_sink_v2       (1745)  (default)              sink + recent + aug
#
# (Models trained for different #steps — config mismatch; note in figure caption.)

set -e
cd /mnt/vita/scratch/vita-students/users/jinghao/code/DiffSynth-Studio
PY=/home/jzheng/miniconda3/envs/diffsynth/bin/python
FEWSTEP=examples/wanvideo/model_inference/Wan2.1-Fun-V1.1-1.3B-Control-DMD-Sink-FewStep.py
OUT=samples/ablation_conditioning
mkdir -p "$OUT"

GPUS=${GPUS:-0}                  # "0" single GPU; "0,1" splits across two
NUM_CHUNKS=${NUM_CHUNKS:-20}     # ~60s; drift shows by late chunks
STEPS=${STEPS:-50}
SEED=${SEED:-42}
POSE=${POSE:-asset/pose_loop.mp4}
REF=${REF:-asset/6.png}
IFS=',' read -ra GPU_ARR <<< "$GPUS"
NGPU=${#GPU_ARR[@]}

M=models/train
LOGDIR=examples/wanvideo/model_inference/logs; mkdir -p "$LOGDIR"
run_one () {  # $1=tag  $2=lora  $3=extra_flags  $4=gpu
    local tag="$1" lora="$2" extra="$3" gpu="$4"
    [ -f "$lora" ] || { echo "[skip] $tag — lora not found: $lora"; return; }
    echo "── [$tag] GPU $gpu  $lora  ($extra) ──"
    CUDA_VISIBLE_DEVICES=$gpu $PY "$FEWSTEP" \
        --sink_lora "$lora" --student "" \
        --control_video "$POSE" --initial_ref "$REF" \
        --height 832 --width 480 \
        --num_chunks "$NUM_CHUNKS" --steps "$STEPS" --seed "$SEED" \
        --save_dir "$OUT" --tag "abl_$tag" $extra \
        > "$LOGDIR/abl_${tag}.log" 2>&1
}

# Four (tag, lora, flags) jobs; dispatched round-robin onto GPU_ARR. Each GPU
# runs its jobs SEQUENTIALLY (no per-GPU memory contention); GPUs run in PARALLEL.
TAGS=(0_nothing 1_recent)
LORAS=(
  "$M/Wan2.1-Fun-V1.1-1.3B-Control_lora_cartoon_chunked_noref/step-1600.safetensors"
  "$M/Wan2.1-Fun-V1.1-1.3B-Control_lora_cartoon_with_ref/step-4975.safetensors"
)
FLAGS=("--no_recent --no_sink" "--no_sink")

# Per-GPU worker: runs all jobs assigned to it, sequentially.
worker () {  # $1 = gpu slot index
    local slot="$1" gpu="${GPU_ARR[$1]}"
    for j in "${!TAGS[@]}"; do
        if [ $(( j % NGPU )) -eq "$slot" ]; then
            run_one "${TAGS[$j]}" "${LORAS[$j]}" "${FLAGS[$j]}" "$gpu"
        fi
    done
}

echo "[dispatch] $NGPU GPU(s): $GPUS  | logs → $LOGDIR/abl_*.log"
for s in "${!GPU_ARR[@]}"; do worker "$s" & done
wait
echo "[all jobs done]"

echo "=== building comparison grid ==="
$PY - "$OUT" "$NUM_CHUNKS" <<'PYEOF'
import sys, os, glob, decord, numpy as np
from PIL import Image, ImageDraw
out_dir, nchunks = sys.argv[1], int(sys.argv[2])
labels = [("0_nothing","pose only"), ("1_recent","recent only (i2v)")]
# late frame (~last chunk) where drift is visible; early frame for reference
total = 49 + (nchunks-1)*48
for tag_when, fi in [("early", 24), ("late", total-12)]:
    cols=[]
    for tag,_ in labels:
        hits=glob.glob(os.path.join(out_dir, f"*abl_{tag}*.mp4"))
        if not hits: print(f"[miss] {tag}"); continue
        vr=decord.VideoReader(sorted(hits)[-1]); f=vr[min(fi,len(vr)-1)].asnumpy()
        cols.append((tag, Image.fromarray(f)))
    if not cols: continue
    w,h=cols[0][1].size
    grid=Image.new("RGB",(w*len(cols), h+28),"white")
    d=ImageDraw.Draw(grid)
    for i,(tag,im) in enumerate(cols):
        grid.paste(im,(i*w,28))
        lab=dict(labels).get(tag,tag)
        d.text((i*w+4,8), lab, fill="black")
    p=os.path.join(out_dir, f"ablation_grid_{tag_when}_f{fi}.png")
    grid.save(p); print("[grid]", p, "| order:", " | ".join(t for t,_ in cols))
PYEOF
echo "done → $OUT/ablation_grid_*.png"
