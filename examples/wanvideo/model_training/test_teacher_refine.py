"""Offline sanity check for teacher_refine: does renoise→teacher-1step actually
SHARPEN a 1-step student's output (裙边 / detail)?

Pipeline:
  1. build teacher (base + sink_v2, frozen) and student (base + sink_v2 + student LoRA)
  2. take one chunk from ChunkAwareDataset → encode conditioning
  3. student 1-step rollout → x_pred  (the actual blurry output)
  4. teacher_refine(x_pred) at a SWEEP of refine_t levels
  5. VAE-decode x_pred and each refined → save side-by-side PNGs + Sobel numbers

If the refined frames aren't visibly sharper than x_pred, the whole
train_dmd_refine approach has a low ceiling and isn't worth running.

Usage (single GPU):
  CUDA_VISIBLE_DEVICES=0 python examples/wanvideo/model_training/test_teacher_refine.py \
      --student_lora models/train/wan1.3b_dmd_clip_gan_v4/step-850.safetensors \
      --chunk_idx 20 --refine_t 150,300,500 --frame 24
"""
import argparse, os, sys
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_dmd_refine import (
    build_pipe, fuse_sink_lora_into_pipe, add_trainable_lora,
    load_lora_into_trainable, encode_batch, rollout_student, teacher_refine,
)
from train_chunk_aware import ChunkAwareDataset
from diffsynth.core import load_state_dict

SINK_LORA = "models/train/Wan2.1-Fun-V1.1-1.3B-Control_lora_cartoon_sink_v2/step-1745.safetensors"


# ---- Sobel sharpness (numpy, no cv2/scipy) ----
SX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
SY = SX.T
def _conv(img, k):
    p = np.pad(img, 1, mode="edge"); H, W = img.shape
    out = np.zeros((H, W), np.float32)
    for i in range(3):
        for j in range(3):
            out += k[i, j] * p[i:i+H, j:j+W]
    return out
def sobel(pil):
    a = np.asarray(pil, np.float32)
    g = 0.299*a[..., 0] + 0.587*a[..., 1] + 0.114*a[..., 2]
    return float(np.sqrt(_conv(g, SX)**2 + _conv(g, SY)**2).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student_lora", required=True, help="student LoRA .safetensors")
    ap.add_argument("--sink_lora", default=SINK_LORA)
    ap.add_argument("--dataset_metadata_path", default="./data/cartoon_15s/metadata.csv")
    ap.add_argument("--chunk_idx", type=int, default=20, help="which dataset chunk")
    ap.add_argument("--height", type=int, default=832)
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--num_frames", type=int, default=49)
    ap.add_argument("--num_inference_steps", type=int, default=1)
    ap.add_argument("--flow_shift", type=float, default=5.0)
    ap.add_argument("--refine_t", default="150,300,500",
                    help="comma-sep timesteps to sweep (each tested as a 1-step refine)")
    ap.add_argument("--refine_steps2", default="",
                    help="optional: a 2-step descending list, e.g. '350,150', tested in addition")
    ap.add_argument("--frame", type=int, default=24, help="latent->pixel frame index to visualize")
    ap.add_argument("--out_dir", default="./samples/refine_test")
    ap.add_argument("--lora_rank", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    import random as _r; _r.seed(args.seed)

    target_modules = ["q", "k", "v", "o", "ffn.0", "ffn.2"]

    # ---- teacher (frozen) ----
    print("[build] teacher = base + sink ...")
    teacher = build_pipe(device, dtype)
    fuse_sink_lora_into_pipe(teacher, args.sink_lora)
    teacher.dit.requires_grad_(False); teacher.dit.eval()

    # ---- student (base + sink + trained LoRA) ----
    print("[build] student = base + sink + student LoRA ...")
    student = build_pipe(device, dtype)
    fuse_sink_lora_into_pipe(student, args.sink_lora)
    student.dit = add_trainable_lora(student.dit, target_modules, args.lora_rank)
    load_lora_into_trainable(student.dit, load_state_dict(args.student_lora))
    student.dit.eval()

    # ---- one chunk ----
    ds = ChunkAwareDataset(
        csv_path=args.dataset_metadata_path, height=args.height, width=args.width,
        chunk_frames=args.num_frames, recent_aug_strength=0.5, dataset_repeat=1,
    )
    idx = args.chunk_idx % len(ds)
    batch = ds[idx]
    print(f"[data] chunk {idx}: {batch['prompt'][:60]}...")

    # ---- schedule ----
    student.scheduler.set_timesteps(num_inference_steps=args.num_inference_steps, shift=args.flow_shift)
    denoising_step_list = [float(t) for t in student.scheduler.timesteps]
    print(f"[schedule] {denoising_step_list}")

    with torch.no_grad():
        cond = encode_batch(student, batch, dtype, device)
        ref_student  = cond["reference_latents_aug"]
        clip_student = cond["clip_feature_aug"]
        ref_teacher  = cond["reference_latents_clean"]
        clip_teacher = cond["clip_feature_clean"]
        prompt_embed = cond["prompt_embed"]
        control_latents = cond["control_latents"]

        # student 1-step output
        x_pred, t_gen = rollout_student(
            student.dit, denoising_step_list, tuple(cond["target_latents"].shape),
            prompt_embed, control_latents, ref_student, clip_student, device, dtype,
        )
        print(f"[student] x_pred ready (exit t={t_gen})")

        # decode helper
        def decode_frame(latent, tag):
            vid = teacher.vae.decode(latent, device=device, tiled=True)  # [B,C,T,H,W] in [-1,1]
            frames = teacher.vae_output_to_video(vid, pattern="B C T H W")
            fr = frames[min(args.frame, len(frames) - 1)]
            s = sobel(fr)
            path = os.path.join(args.out_dir, f"chunk{idx}_{tag}.png")
            fr.save(path)
            return fr, s, path

        results = []  # (label, PIL, sobel)
        fr0, s0, p0 = decode_frame(x_pred, "student_x0")
        results.append(("student_x0", fr0, s0))
        print(f"  student_x0          sobel={s0:.3f}  {p0}")

        # 1-step refine sweep
        for t in [float(x) for x in args.refine_t.split(",") if x.strip()]:
            x_ref = teacher_refine(teacher.dit, x_pred, [t],
                                   prompt_embed, control_latents, ref_teacher, clip_teacher,
                                   device, dtype)
            fr, s, p = decode_frame(x_ref, f"refine_t{int(t)}")
            results.append((f"refine_t{int(t)}", fr, s))
            print(f"  refine_t{int(t):<4d}        sobel={s:.3f}  (Δ vs student {s - s0:+.3f})  {p}")

        # optional 2-step
        if args.refine_steps2.strip():
            tl = [float(x) for x in args.refine_steps2.split(",")]
            x_ref = teacher_refine(teacher.dit, x_pred, tl,
                                   prompt_embed, control_latents, ref_teacher, clip_teacher,
                                   device, dtype)
            fr, s, p = decode_frame(x_ref, f"refine_2step_{'_'.join(str(int(t)) for t in tl)}")
            results.append((f"refine_2step", fr, s))
            print(f"  refine_2step {tl}  sobel={s:.3f}  (Δ {s - s0:+.3f})  {p}")

    # ---- side-by-side grid (student | each refine) ----
    w, h = results[0][1].size
    n = len(results)
    grid = Image.new("RGB", (w * n, h))
    for i, (label, fr, s) in enumerate(results):
        grid.paste(fr, (i * w, 0))
    grid_path = os.path.join(args.out_dir, f"chunk{idx}_grid.png")
    grid.save(grid_path)
    print(f"\n[grid] {' | '.join(l for l, _, _ in results)}")
    print(f"[grid] saved → {grid_path}")
    print(f"\nSobel summary:")
    for label, _, s in results:
        print(f"  {label:<22s} {s:.3f}")


if __name__ == "__main__":
    main()
