"""
Yoyo-extend a pose video to a target DURATION (instead of num_chunks).
Same forward-backward cycle logic as make_loop_pose.py (boundary-aware,
no duplicate first/last frames at the joins).

Usage:
    python asset/make_pose_long.py --duration 180
    python asset/make_pose_long.py --input asset/pose.mp4 --duration 300 \
        --output asset/pose_loop_5min.mp4
"""
import argparse
import numpy as np
from decord import VideoReader
import imageio


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input",    default="asset/pose.mp4")
    p.add_argument("--output",   default="asset/pose_loop_180s.mp4")
    p.add_argument("--duration", type=float, default=180.0,
                   help="Target output duration in seconds (at source fps).")
    args = p.parse_args()

    vr = VideoReader(args.input)
    fps = vr.get_avg_fps()
    frames = vr[:].asnumpy()
    print(f"Source: {len(frames)} frames @ {fps:.2f} fps "
          f"= {len(frames)/fps:.2f}s")

    # forward + backward cycle, skipping first/last on reverse to avoid duplicating
    # boundary frames (matches make_loop_pose.py exactly)
    fwd = frames
    bwd = frames[-2:0:-1]
    cycle = np.concatenate([fwd, bwd], axis=0)
    print(f"Yoyo cycle: {len(cycle)} frames = {len(cycle)/fps:.2f}s")

    # tile cycles, trim to exactly target duration
    needed = int(round(args.duration * fps))
    repeats = -(-needed // len(cycle))                # ceil division
    tiled = np.tile(cycle, (repeats, 1, 1, 1))[:needed]
    print(f"Output: {len(tiled)} frames = {len(tiled)/fps:.2f}s "
          f"(~{len(tiled)/len(cycle):.2f} cycles, trimmed)")

    imageio.mimwrite(args.output, tiled, fps=fps, codec="libx264",
                     output_params=["-crf", "18", "-pix_fmt", "yuv420p"])
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
