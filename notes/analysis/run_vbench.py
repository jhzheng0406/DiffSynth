"""VBench++ technical-quality eval (custom_input — no prompt suite / no GT).

The 6 dims are SVI's VBench++ set, all reference-free:
  subject_consistency  (DINO)      background_consistency (CLIP)
  motion_smoothness    (AMT)       dynamic_degree         (RAFT)
  aesthetic_quality    (LAION)     imaging_quality        (MUSIQ)

Two modes:
  --mode whole  : one score per video (overall + per-video table).
  --mode drift  : slice each video into temporal windows, score each window,
                  write a per-window CSV (+ a plot) → SVI Fig5 quality-vs-time
                  drift curve. All windows go into ONE dir so VBench loads each
                  model once.

Run in the ISOLATED vbench env (NOT diffsynth):
  ENV=/mnt/.../envs/vbench310
  # whole-video, full 6 dims
  $ENV/bin/python notes/analysis/run_vbench.py --name full --mode whole \
      --videos "samples/recycle_placement_200/*_rp_*.mp4"
  # drift curve (window = 49f every 196f ≈ every 4 chunks)
  $ENV/bin/python notes/analysis/run_vbench.py --name drift --mode drift \
      --window_frames 49 --window_stride 196 \
      --videos "samples/recycle_placement_200/*_rp_norecycle.mp4" \
               "samples/recycle_placement_200/*_rp_student.mp4" \
               "samples/recycle_placement_200/*_rp_both.mp4"
"""
import argparse, os, glob, json, re

# Pin VBench's weight cache to a project-local dir BEFORE importing vbench, so it
# doesn't resolve to a HOME-relative ~/.cache/vbench (HOME differs across the
# runai container vs login shell → it'd re-trigger the missing-`wget` download).
# Pre-downloaded weights (amt/musiq/raft) live here.
os.environ.setdefault(
    "VBENCH_CACHE_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "vbench_cache"),
)

import torch

# VBench's checkpoints (AMT/RAFT/MUSIQ) are full pickles (contain typing.OrderedDict
# etc.), which fail under torch>=2.6's default weights_only=True. These are trusted
# files we downloaded ourselves → force weights_only=False for all of VBench's loads.
_orig_torch_load = torch.load
def _torch_load_compat(*a, **k):
    k.setdefault("weights_only", False)
    return _orig_torch_load(*a, **k)
torch.load = _torch_load_compat

DEFAULT_DIMS = [
    "subject_consistency", "background_consistency", "motion_smoothness",
    "dynamic_degree", "aesthetic_quality", "imaging_quality",
]


def find_full_info():
    import vbench
    d = os.path.dirname(vbench.__file__)
    for root, _, files in os.walk(d):
        if "VBench_full_info.json" in files:
            return os.path.join(root, "VBench_full_info.json")
    return None


def expand_videos(patterns):
    vids = []
    for v in patterns:
        hits = sorted(glob.glob(v))
        vids.extend(hits if hits else ([v] if os.path.isfile(v) else []))
    return [os.path.abspath(v) for v in vids]


def short_tag(path):
    """Compact, filesystem-safe label per source video (for grouping clips).
    Must DIFFER between the videos being compared, else they collapse to one line."""
    b = os.path.splitext(os.path.basename(path))[0]
    m = re.search(r"_rp_([a-z]+)", b)              # recycle_placement naming
    if m:
        return m.group(1)
    m = re.search(r"exp-(.+?)_step", b)            # experiment (student) dir name
    if m:
        return re.sub(r"[^A-Za-z0-9]+", "-", m.group(1))
    return re.sub(r"[^A-Za-z0-9]+", "-", b)[:80]   # fallback: longer slice


def write_clip(src, start, n, out_path):
    """Read [start, start+n) from src and write a clip. decord read + imageio write."""
    import decord, numpy as np
    vr = decord.VideoReader(src)
    fps = float(vr.get_avg_fps()) or 24.0
    end = min(start + n, len(vr))
    if end - start < 8:                         # too short to score
        return False
    idx = list(range(start, end))
    frames = vr.get_batch(idx).asnumpy()        # [T,H,W,3] uint8
    try:
        import imageio
        with imageio.get_writer(out_path, fps=fps, macro_block_size=1) as w:
            for f in frames:
                w.append_data(f)
    except Exception:
        import torchvision
        torchvision.io.write_video(out_path, torch.from_numpy(frames), fps=round(fps))
    return True


def run_vbench(videos_dir, name, dims, out_dir):
    from vbench import VBench
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full_info = find_full_info()
    print(f"[run_vbench] device={device}  full_info={full_info}  dir={videos_dir}")
    vb = VBench(device, full_info, out_dir)
    vb.evaluate(videos_path=videos_dir, name=name, dimension_list=list(dims), mode="custom_input")
    res_path = os.path.join(out_dir, f"{name}_eval_results.json")
    if not os.path.isfile(res_path):
        cands = sorted(glob.glob(os.path.join(out_dir, "*_eval_results.json")), key=os.path.getmtime)
        res_path = cands[-1] if cands else None
    return json.load(open(res_path)) if res_path else {}


def parse_per_video(res):
    """-> {dim: {video_basename: score}} and {dim: overall}."""
    overall, per = {}, {}
    for dim, val in res.items():
        if isinstance(val, (list, tuple)):
            overall[dim] = val[0]
            if len(val) > 1:
                for item in val[1]:
                    vp = os.path.basename(item.get("video_path", "?"))
                    sc = item.get("video_results", item.get("video_score"))
                    if isinstance(sc, (int, float)):
                        per.setdefault(dim, {})[vp] = sc
        else:
            overall[dim] = val
    return overall, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--out", default="notes/analysis/vbench_out")
    ap.add_argument("--name", default="run")
    ap.add_argument("--dimensions", nargs="+", default=DEFAULT_DIMS)
    ap.add_argument("--mode", choices=["whole", "drift"], default="whole")
    ap.add_argument("--window_frames", type=int, default=49)
    ap.add_argument("--window_stride", type=int, default=196)
    args = ap.parse_args()

    vids = expand_videos(args.videos)
    assert vids, f"no videos matched: {args.videos}"
    os.makedirs(args.out, exist_ok=True)

    # ---------------- WHOLE ----------------
    if args.mode == "whole":
        vdir = os.path.join(args.out, f"videos_{args.name}")
        os.makedirs(vdir, exist_ok=True)
        for v in vids:
            dst = os.path.join(vdir, os.path.basename(v))
            if not os.path.exists(dst):
                os.symlink(v, dst)
        print(f"[whole] {len(vids)} videos, dims={args.dimensions}")
        res = run_vbench(vdir, args.name, args.dimensions, args.out)
        overall, per = parse_per_video(res)
        print("\n%-24s overall" % "dimension"); print("-" * 33)
        for d in args.dimensions:
            print(f"{d:24s} {overall.get(d, float('nan')):.4f}")
        print("\nper-video:")
        for v in vids:
            b = os.path.basename(v)
            sc = "  ".join(f"{d[:11]}={per.get(d, {}).get(b, float('nan')):.3f}" for d in args.dimensions)
            print(f"  {b[:50]:50s} {sc}")
        return

    # ---------------- DRIFT ----------------
    import decord
    cdir = os.path.join(args.out, f"clips_{args.name}")
    os.makedirs(cdir, exist_ok=True)
    # clip name: {tag}__w{idx:04d}.mp4  -> tag = source line, idx = window
    clip_meta = {}   # clip_basename -> (tag, win_idx, start)
    for v in vids:
        tag = short_tag(v)
        n = len(decord.VideoReader(v))
        starts = list(range(0, max(1, n - args.window_frames + 1), args.window_stride))
        for wi, s in enumerate(starts):
            cn = f"{tag}__w{wi:04d}.mp4"
            op = os.path.join(cdir, cn)
            if not os.path.isfile(op):
                if not write_clip(v, s, args.window_frames, op):
                    continue
            clip_meta[cn] = (tag, wi, s)
    print(f"[drift] {len(vids)} videos -> {len(clip_meta)} window-clips "
          f"(win={args.window_frames}f stride={args.window_stride}f), dims={args.dimensions}")

    res = run_vbench(cdir, args.name, args.dimensions, args.out)
    _, per = parse_per_video(res)

    # write CSV: tag, win_idx, start_frame, <dims...>
    import csv as _csv
    csv_path = os.path.join(args.out, f"vbench_drift_{args.name}.csv")
    rows = []
    for cn, (tag, wi, s) in sorted(clip_meta.items(), key=lambda kv: (kv[1][0], kv[1][1])):
        row = {"video": tag, "window": wi, "start_frame": s}
        for d in args.dimensions:
            row[d] = per.get(d, {}).get(cn, "")
        rows.append(row)
    with open(csv_path, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["video", "window", "start_frame"] + list(args.dimensions))
        w.writeheader(); w.writerows(rows)
    print(f"[drift] CSV -> {csv_path}")

    # plot: one subplot per dim, one line per video tag
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        tags = sorted({r["video"] for r in rows})
        nd = len(args.dimensions)
        fig, axes = plt.subplots(nd, 1, figsize=(9, 2.4 * nd), sharex=True)
        if nd == 1:
            axes = [axes]
        for ax, d in zip(axes, args.dimensions):
            for tg in tags:
                xs = [r["window"] for r in rows if r["video"] == tg and r[d] != ""]
                ys = [r[d] for r in rows if r["video"] == tg and r[d] != ""]
                ax.plot(xs, ys, marker=".", label=tg)
            ax.set_ylabel(d, fontsize=8); ax.grid(alpha=0.3); ax.legend(fontsize=7)
        axes[-1].set_xlabel("window index (time →)")
        png = os.path.join(args.out, f"vbench_drift_{args.name}.png")
        fig.tight_layout(); fig.savefig(png, dpi=130)
        print(f"[drift] plot -> {png}")
    except Exception as e:
        print(f"[drift] plot skipped ({e}); CSV is ready for your own plotting.")


if __name__ == "__main__":
    main()
