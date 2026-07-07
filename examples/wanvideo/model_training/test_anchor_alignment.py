"""Verify the trajectory-anchor PREMISE before training: from the SAME initial
noise, does the student's 1-step output structurally ALIGN with the teacher's
K-step output (only sharper), or does it land at a different pose/composition?

  aligned + teacher sharper  → L1 anchor will transfer detail. Train it.
  pose/composition differs    → L1 would fight over structure. Don't train as-is
                                (use --anchor_highfreq, or rethink).

Pipeline:
  1. teacher (base+sink), student (base+sink+student LoRA)
  2. one dataset chunk → encode conditioning
  3. init_noise = randn  (shared)
  4. x_pred    = student 1-step rollout(init_noise)
  5. x_teacher = teacher K-step ODE(init_noise)         [--anchor_steps]
  6. decode both → side-by-side + Sobel + high-freq-residual energy

Usage:
  CUDA_VISIBLE_DEVICES=0 python examples/wanvideo/model_training/test_anchor_alignment.py \
      --student_lora models/train/wan1.3b_dmd_recycle_v1/step-850.safetensors \
      --chunk_idx 20 --anchor_steps 4 --anchor_cfg 0 --frame 24
"""
import argparse, os, sys
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_dmd_refine import (
    build_pipe, fuse_sink_lora_into_pipe, add_trainable_lora,
    load_lora_into_trainable, encode_batch, encode_prompt,
    rollout_student, teacher_trajectory, highfreq_residual,
)
from train_chunk_aware import ChunkAwareDataset
from diffsynth.core import load_state_dict

SINK_LORA = "models/train/Wan2.1-Fun-V1.1-1.3B-Control_lora_cartoon_sink_v2/step-1745.safetensors"
NEG = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，"
       "JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，"
       "形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走")

SX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32); SY = SX.T
def _cv(g, k):
    p = np.pad(g, 1, mode="edge"); H, W = g.shape; o = np.zeros((H, W), np.float32)
    for i in range(3):
        for j in range(3): o += k[i, j] * p[i:i+H, j:j+W]
    return o
def sobel(pil):
    a = np.asarray(pil, np.float32); g = 0.299*a[..., 0]+0.587*a[..., 1]+0.114*a[..., 2]
    return float(np.sqrt(_cv(g, SX)**2 + _cv(g, SY)**2).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student_lora", required=True)
    ap.add_argument("--sink_lora", default=SINK_LORA)
    ap.add_argument("--dataset_metadata_path", default="./data/cartoon_15s/metadata.csv")
    ap.add_argument("--chunk_idx", type=int, default=20)
    ap.add_argument("--height", type=int, default=832)
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--num_frames", type=int, default=49)
    ap.add_argument("--flow_shift", type=float, default=5.0)
    ap.add_argument("--anchor_steps", type=int, default=4)
    ap.add_argument("--anchor_cfg", type=float, default=0.0)
    ap.add_argument("--frame", type=int, default=24)
    ap.add_argument("--out_dir", default="./samples/refine_test")
    ap.add_argument("--lora_rank", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device, dtype = "cuda", torch.bfloat16
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    import random as _r; _r.seed(args.seed)
    tm = ["q", "k", "v", "o", "ffn.0", "ffn.2"]

    print("[build] teacher = base + sink ...")
    teacher = build_pipe(device, dtype); fuse_sink_lora_into_pipe(teacher, args.sink_lora)
    teacher.dit.requires_grad_(False); teacher.dit.eval()

    print("[build] student = base + sink + student LoRA ...")
    student = build_pipe(device, dtype); fuse_sink_lora_into_pipe(student, args.sink_lora)
    student.dit = add_trainable_lora(student.dit, tm, args.lora_rank)
    load_lora_into_trainable(student.dit, load_state_dict(args.student_lora))
    student.dit.eval()

    ds = ChunkAwareDataset(csv_path=args.dataset_metadata_path, height=args.height,
                           width=args.width, chunk_frames=args.num_frames,
                           recent_aug_strength=0.5, dataset_repeat=1)
    idx = args.chunk_idx % len(ds); batch = ds[idx]
    print(f"[data] chunk {idx}: {batch['prompt'][:60]}...")

    # student single step + teacher K-step schedule (force aligned start)
    student.scheduler.set_timesteps(num_inference_steps=1, shift=args.flow_shift)
    denoising_step_list = [float(t) for t in student.scheduler.timesteps]
    student.scheduler.set_timesteps(num_inference_steps=args.anchor_steps, shift=args.flow_shift)
    teacher_step_list = [float(t) for t in student.scheduler.timesteps]
    teacher_step_list[0] = denoising_step_list[0]
    print(f"[schedule] student 1-step t0={denoising_step_list[0]:.1f}")
    print(f"[schedule] teacher {args.anchor_steps}-step: {[round(t,1) for t in teacher_step_list]}")

    neg = encode_prompt(teacher, NEG, dtype, device) if args.anchor_cfg else None

    with torch.no_grad():
        cond = encode_batch(student, batch, dtype, device)
        init_noise = torch.randn(tuple(cond["target_latents"].shape), dtype=dtype, device=device)

        x_pred, t_gen = rollout_student(
            student.dit, denoising_step_list, tuple(cond["target_latents"].shape),
            cond["prompt_embed"], cond["control_latents"],
            cond["reference_latents_aug"], cond["clip_feature_aug"],
            device, dtype, initial_noise=init_noise,
        )
        x_teacher = teacher_trajectory(
            teacher.dit, init_noise, teacher_step_list,
            cond["prompt_embed"], cond["control_latents"],
            cond["reference_latents_clean"], cond["clip_feature_clean"],
            device, dtype, neg_prompt_embed=neg, cfg_scale=args.anchor_cfg,
        )

        # high-freq energy (latent) — how much detail each carries
        hf_s = float(highfreq_residual(x_pred,    3).abs().mean())
        hf_t = float(highfreq_residual(x_teacher, 3).abs().mean())
        # structural distance latent (how aligned): normalized L1
        align = float((x_pred - x_teacher).abs().mean())
        print(f"\n[latent] student hf={hf_s:.4f}  teacher hf={hf_t:.4f}  "
              f"(teacher/student={hf_t/max(hf_s,1e-6):.2f}×)")
        print(f"[latent] |x_pred - x_teacher| L1 = {align:.4f}  (lower = better aligned)")

        def decode_frame(latent, tag):
            vid = teacher.vae.decode(latent, device=device, tiled=True)
            frames = teacher.vae_output_to_video(vid, pattern="B C T H W")
            fr = frames[min(args.frame, len(frames) - 1)]
            fr.save(os.path.join(args.out_dir, f"align_chunk{idx}_{tag}.png"))
            return fr, sobel(fr)

        fr_s, s_s = decode_frame(x_pred, "student_1step")
        fr_t, s_t = decode_frame(x_teacher, f"teacher_{args.anchor_steps}step")

    w, h = fr_s.size
    grid = Image.new("RGB", (w*2, h), "white")
    grid.paste(fr_s, (0, 0)); grid.paste(fr_t, (w, 0))
    gp = os.path.join(args.out_dir, f"align_chunk{idx}_grid.png")
    grid.save(gp)
    print(f"\n[pixel] student_1step sobel={s_s:.3f}   teacher_{args.anchor_steps}step sobel={s_t:.3f}")
    print(f"[grid] LEFT=student_1step  RIGHT=teacher_{args.anchor_steps}step  → {gp}")
    print("\nRead it: same pose/composition + teacher sharper → ALIGNED, train the anchor.")
    print("         different pose → NOT aligned, L1 fights structure (use --anchor_highfreq or rethink).")


if __name__ == "__main__":
    main()
