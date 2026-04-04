"""
Make a forward-backward looping pose video long enough for N chunks.

Usage:
    python asset/make_loop_pose.py \
        --input  asset/pose.mp4 \
        --output asset/pose_loop.mp4 \
        --num_chunks 10 \
        --chunk_frames 49
"""
import argparse
import numpy as np
from decord import VideoReader
import imageio

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",        default="asset/pose.mp4")
    parser.add_argument("--output",       default="asset/pose_loop.mp4")
    parser.add_argument("--num_chunks",   type=int, default=10)
    parser.add_argument("--chunk_frames", type=int, default=49)
    args = parser.parse_args()

    # total frames needed: first chunk full, remaining chunks each add (L-1) new frames
    L = args.chunk_frames
    needed = L + (args.num_chunks - 1) * (L - 1)   # = 481 for 10 chunks of 49
    print(f"Need {needed} frames for {args.num_chunks} chunks × {L} frames")

    # read source video
    vr = VideoReader(args.input)
    fps = vr.get_avg_fps()
    frames = vr[:].asnumpy()   # [T, H, W, 3]  uint8
    print(f"Source: {len(frames)} frames @ {fps:.1f} fps")

    # build forward-backward cycle (avoid duplicating boundary frames)
    fwd = frames                    # [T]
    bwd = frames[-2:0:-1]           # [T-2]  reverse without first/last
    cycle = np.concatenate([fwd, bwd], axis=0)
    print(f"One cycle: {len(cycle)} frames")

    # tile cycles until we have enough
    repeats = -(-needed // len(cycle))   # ceiling division
    tiled = np.tile(cycle, (repeats, 1, 1, 1))[:needed]
    print(f"Output: {len(tiled)} frames")

    # write
    imageio.mimwrite(args.output, tiled, fps=fps, codec="libx264",
                     output_params=["-crf", "18", "-pix_fmt", "yuv420p"])
    print(f"Saved → {args.output}")

if __name__ == "__main__":
    main()
