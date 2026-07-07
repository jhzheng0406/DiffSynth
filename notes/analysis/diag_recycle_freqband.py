"""DIAGNOSTIC (read-only): does the recycle residual have separable low- vs
high-frequency structure? If yes → frequency-band recycle (#2) is justified;
if the residual is uniform → don't bother.

Mechanism: load a trained recycle student, run its 1-step generation on real
chunks, compute the SAME residual recycle collects:
    err = x_pred_clean[:, :, -1] - target_latents[:, :, -1]     (last latent frame)
then split each err into low/high spatial-frequency bands and report:
  - energy fraction in each band (is there substantial energy in BOTH?)
  - how stable the split is across samples (variance of high-frac)
  - per-band mean spatial energy map (saved as a figure)

Run (1 GPU; does NOT touch training code, only imports its helpers):
  cd DiffSynth-Studio
  CUDA_VISIBLE_DEVICES=0 python notes/analysis/diag_recycle_freqband.py \
      --student_ckpt models/train/wan1.3b_dmd_recycle_v2/step-850.safetensors \
      --sink_lora models/train/Wan2.1-Fun-V1.1-1.3B-Control_lora_cartoon_sink_recycle_v1/step-875.safetensors \
      --metadata data/cartoon_15s/metadata.csv --height 832 --width 480 --n 80
"""
import argparse, os, sys
import torch
import torch.nn.functional as F

# import the student's own machinery (module-level defs are import-safe; main()
# is guarded). Path: this file lives in notes/analysis/, training code in examples/.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "examples", "wanvideo", "model_training"))
os.chdir(ROOT)

from train_dmd_recycle import (                       # noqa: E402
    build_pipe, fuse_sink_lora_into_pipe, add_trainable_lora,
    load_lora_into_trainable, encode_batch, rollout_student,
)
from train_chunk_aware import ChunkAwareDataset        # noqa: E402
from diffsynth.core import load_state_dict             # noqa: E402


def split_bands(x, k):
    """x: [C,h,w] → (low, high). low = box-blur via avg_pool(k)+nearest-up."""
    xb = x.unsqueeze(0)
    low = F.avg_pool2d(xb, kernel_size=k, stride=k, ceil_mode=True)
    low = F.interpolate(low, size=x.shape[-2:], mode="nearest")
    low = low.squeeze(0)
    return low, x - low


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student_ckpt",
                    default="models/train/wan1.3b_dmd_recycle_v2/step-850.safetensors")
    ap.add_argument("--sink_lora",
                    default="models/train/Wan2.1-Fun-V1.1-1.3B-Control_lora_cartoon_sink_recycle_v1/step-875.safetensors")
    ap.add_argument("--metadata", default="data/cartoon_15s/metadata.csv")
    ap.add_argument("--height", type=int, default=832)
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--num_frames", type=int, default=49)
    ap.add_argument("--n", type=int, default=80, help="#chunks to sample")
    ap.add_argument("--k", type=int, default=2, help="low-pass pool kernel (band cutoff)")
    ap.add_argument("--lora_rank", type=int, default=32)
    ap.add_argument("--lora_target_modules", default="q,k,v,o,ffn.0,ffn.2")
    ap.add_argument("--flow_shift", type=float, default=5.0)
    ap.add_argument("--plp", action="store_true", help="match a PLP-trained ckpt")
    ap.add_argument("--out", default="notes/analysis/diag_recycle_freqband")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    # ---- build student exactly as training does ----
    pipe = build_pipe(device, dtype)
    fuse_sink_lora_into_pipe(pipe, args.sink_lora)
    pipe.dit = add_trainable_lora(pipe.dit, args.lora_target_modules.split(","), args.lora_rank)
    load_lora_into_trainable(pipe.dit, load_state_dict(args.student_ckpt))
    pipe.dit.eval()
    print(f"[diag] student loaded: {args.student_ckpt}")

    pipe.scheduler.set_timesteps(num_inference_steps=1, shift=args.flow_shift)
    denoising_step_list = [float(t) for t in pipe.scheduler.timesteps]
    print(f"[diag] 1-step denoising_step_list={[round(t,1) for t in denoising_step_list]}")

    ds = ChunkAwareDataset(args.metadata, args.height, args.width,
                           chunk_frames=args.num_frames, plp=args.plp)
    n = min(args.n, len(ds))

    # ---- collect residuals + band energies ----
    low_frac, high_frac = [], []
    sum_low_map = sum_high_map = None
    count = 0
    for i in range(n):
        try:
            batch = ds[i * max(1, len(ds) // n)]            # spread across dataset
            cond = encode_batch(pipe, batch, dtype, device, no_recent=False, plp=args.plp)
            tgt = cond["target_latents"]
            with torch.no_grad():
                x_pred, _ = rollout_student(
                    pipe.dit, denoising_step_list, tuple(tgt.shape),
                    cond["prompt_embed"], cond["control_latents"],
                    cond["reference_latents_clean"], cond["clip_feature_clean"],
                    device, dtype, initial_noisy=torch.randn(tgt.shape, dtype=dtype, device=device),
                    exit_idx=0,
                )
            err = (x_pred[:, :, -1] - tgt[:, :, -1]).float().squeeze(0)   # [C,h,w]
        except Exception as e:
            print(f"  [skip {i}] {e!r}")
            continue
        low, high = split_bands(err, args.k)
        el, eh = (low ** 2).sum().item(), (high ** 2).sum().item()
        tot = el + eh + 1e-12
        low_frac.append(el / tot); high_frac.append(eh / tot)
        lm = (low ** 2).mean(0); hm = (high ** 2).mean(0)              # [h,w] per-band energy
        sum_low_map = lm if sum_low_map is None else sum_low_map + lm
        sum_high_map = hm if sum_high_map is None else sum_high_map + hm
        count += 1

    if count == 0:
        print("[diag] no residuals collected — check paths."); return

    import numpy as np
    lf, hf = np.array(low_frac), np.array(high_frac)
    print(f"\n[diag] residuals collected: {count}  (k={args.k})")
    print(f"  low-band  energy fraction:  mean={lf.mean():.3f}  std={lf.std():.3f}")
    print(f"  high-band energy fraction:  mean={hf.mean():.3f}  std={hf.std():.3f}")
    print(f"  → both bands substantial? {'YES' if min(lf.mean(), hf.mean()) > 0.15 else 'marginal'}")
    print(f"  → split stable across samples (low std)? {'YES' if hf.std() < 0.12 else 'noisy'}")

    # ---- figure: per-band mean spatial energy map + high-frac histogram ----
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(13, 4))
        ax[0].imshow((sum_low_map / count).cpu().numpy());  ax[0].set_title(f"low-band energy (k={args.k})")
        ax[1].imshow((sum_high_map / count).cpu().numpy()); ax[1].set_title("high-band energy")
        ax[2].hist(hf, bins=20); ax[2].set_title("high-band energy fraction"); ax[2].set_xlabel("frac")
        for a in ax[:2]: a.axis("off")
        fig.tight_layout(); fig.savefig(f"{args.out}.png", dpi=130)
        print(f"[diag] figure → {args.out}.png")
    except Exception as e:
        print(f"[diag] plot skipped: {e}")

    print("\n[interpretation] If BOTH bands carry substantial, stable energy, a "
          "two-bank (low/high) recycle with separate α_low/α_high is justified — "
          "low-band ↔ background/stability drift, high-band ↔ detail/clarity drift. "
          "If high-frac ≈ 0 or wildly varies, keep flat FIFO.")


if __name__ == "__main__":
    main()
