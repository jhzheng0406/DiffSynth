"""Reusable comparison grid: rows = different videos (labeled), cols = frames of
the SAME timeline (so you read drift left→right, methods top→bottom).

Edit FIGS below and run:
  /home/jzheng/miniconda3/envs/diffsynth/bin/python notes/analysis/make_compare_grid.py
"""
import os, decord, numpy as np
from PIL import Image, ImageDraw

ROOT = "/mnt/vita/scratch/vita-students/users/jinghao/code/DiffSynth-Studio"
OUT  = f"{ROOT}/notes/analysis"
CELL_W = 200          # per-frame cell width (px); height auto from aspect
LABEL_W = 130         # left label column width
TOP_H = 24            # top time-label strip

def build(name, rows, frac_cols=(0.0, 0.33, 0.66, 0.98), col_labels=None):
    """rows = [(label, abspath), ...]; frac_cols = fractions along the video."""
    rows = [(l, p) for l, p in rows if os.path.isfile(p)]
    if not rows:
        print(f"[{name}] no valid videos, skip"); return
    # frame indices from the FIRST video's length (assume comparable timelines)
    vr0 = decord.VideoReader(rows[0][1]); n0 = len(vr0)
    idxs = [min(int(f * (n0 - 1)), n0 - 1) for f in frac_cols]
    if col_labels is None:
        col_labels = [f"{int(i)}f" for i in idxs]

    def cell(path, fi):
        vr = decord.VideoReader(path)
        im = Image.fromarray(vr[min(fi, len(vr) - 1)].asnumpy())
        w, h = im.size
        return im.resize((CELL_W, int(h * CELL_W / w)), Image.LANCZOS)

    ch = cell(rows[0][1], 0).size[1]
    W = LABEL_W + CELL_W * len(idxs)
    H = TOP_H + ch * len(rows)
    grid = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(grid)
    for j, cl in enumerate(col_labels):
        d.text((LABEL_W + j * CELL_W + 4, 6), cl, fill="black")
    for i, (lab, p) in enumerate(rows):
        d.text((4, TOP_H + i * ch + ch // 2 - 4), lab, fill="black")
        for j, fi in enumerate(idxs):
            grid.paste(cell(p, fi), (LABEL_W + j * CELL_W, TOP_H + i * ch))
    out = f"{OUT}/{name}.png"
    grid.save(out)
    print(f"[{name}] {len(rows)} rows x {len(idxs)} cols -> {out}")


A = f"{ROOT}/samples/ablation_conditioning"
S = f"{ROOT}/samples/onestep_clarity"

# ── FIG1: teacher conditioning (sink/ref/aug[/recycle]) — 50-step teacher ──
build("fig1_teacher_conditioning", [
    ("recent (i2v)",     f"{A}/dmd_sink-with_ref_nostudent_50step_cfg5.0_chunks20x49_832x480_seed42_abl_1_recent.mp4"),
    ("sink only",        f"{A}/dmd_sink-sinkonly_norecent_nostudent_50step_cfg5.0_chunks20x49_832x480_seed42_abl_1_sink.mp4"),
    ("sink+recent",      f"{A}/dmd_sink-sink_noaug_nostudent_50step_cfg5.0_chunks20x49_832x480_seed42_abl_2_recent.mp4"),
    ("sink+recent+aug",  f"{A}/dmd_sink-sink_v2_nostudent_50step_cfg5.0_chunks20x49_832x480_seed42_abl_3_aug.mp4"),
    ("+recycle (teacher)", f"{A}/dmd_sink-sink_recycle_v1_nostudent_50step_cfg5.0_chunks20x49_832x480_seed42_abl_4_recycle.mp4"),
], frac_cols=(0.025, 0.325, 0.675, 0.975),
   col_labels=["1.5s (chunk0)", "19.5s (chunk6)", "40.5s (chunk13)", "58.5s (chunk19)"])

# ── FIG3: recycle placement — 1-step students, 200 chunks (drift more visible) ──
RP = f"{ROOT}/samples/recycle_placement_200"
build("fig3_recycle_placement", [
    ("One-Forcing (1-step)",  f"{RP}/dmd_sink-sink_v2_exp-Wan2.1-Fun-V1.1-1.3B-Control_lora_cartoon_sink_dmd_oneforcing_1step_step-850_1step_cfg1.0_chunks200x49_832x480_seed42_rp_norecycle.mp4"),
    ("+ student recycle (v1)", f"{RP}/dmd_sink-sink_v2_exp-wan1.3b_dmd_recycle_v1_step-850_1step_cfg1.0_chunks200x49_832x480_seed42_rp_student.mp4"),
    ("+ teacher+student (v2)", f"{RP}/dmd_sink-sink_recycle_v1_exp-wan1.3b_dmd_recycle_v2_step-850_1step_cfg1.0_chunks200x49_832x480_seed42_rp_both.mp4"),
], frac_cols=(0.2025, 0.4525, 0.8025, 0.9975),
   col_labels=["121.5s (chunk40)", "271.5s (chunk90)", "481.5s (chunk160)", "598.5s (chunk199)"])

# ── FIG2: distillation NFE — CLARITY (early chunks of the EXISTING long videos,
# pre-drift; recycle N/A). All chunks60 / seed42 → matched. 1-step+GAN beats
# 2/4-step no-GAN → GAN > NFE for sharpness.
DFV = f"{ROOT}/samples/dmd_fewstep_validation"
build("fig2_nfe_clarity", [
    ("1-step (+GAN)", f"{S}/01_recycle/dmd_sink-sink_recycle_v1_exp-wan1.3b_dmd_recycle_v2_step-750_1step_cfg1.0_chunks60x49_832x480_seed42.mp4"),
    ("2-step (no GAN)", f"{DFV}/01_v2_2step/dmd_sink-sink_v2_exp-Wan2.1-Fun-V1.1-1.3B-Control_lora_cartoon_sink_dmd_2step_step-1800_2step_cfg1.0_chunks60x49_832x480_seed42.mp4"),
    ("4-step (no GAN)", f"{ROOT}/samples/nfe_compare/dmd_sink-sink_v2_exp-Wan2.1-Fun-V1.1-1.3B-Control_lora_cartoon_sink_dmd_v2_step-1700_4step_cfg1.0_chunks60x49_832x480_seed42_nfe_4step.mp4"),
    ("teacher 50-step", f"{S}/_teacher_ref_50step/dmd_sink-sink_v2_nostudent_50step_cfg5.0_chunks60x49_832x480_seed42.mp4"),
], frac_cols=(0.025, 0.075, 0.125, 0.175), col_labels=["1.5s (chunk0)", "4.5s (chunk1)", "7.5s (chunk2)", "10.5s (chunk3)"])
