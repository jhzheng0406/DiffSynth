"""
Compare v6 ckpt against the original base Wan ckpt to see what actually moved.

If base Wan params barely moved AND helios params didn't move,
then the v6 ckpt is essentially identical to base+patch → loading it should
behave like zero-shot. If it produces noise instead, the bug is in the LOAD path
(missing keys, key naming, dtype mismatch).

Run from DiffSynth-Studio repo root:
    python test_v6_vs_base.py
"""
import sys
sys.path.insert(0, "/mnt/vita/scratch/vita-students/users/jinghao/code/DiffSynth-Studio")
import torch
from safetensors.torch import load_file

V6   = "models/train/Wan2.1-Fun-V1.1-1.3B-Control-Helios-v6-eadtest/step-200.safetensors"
BASE = "/mnt/vita/scratch/vita-students/users/jinghao/code/VideoX-Fun/models/Diffusion_Transformer/Wan2.1-Fun-V1.1-1.3B-Control/diffusion_pytorch_model.safetensors"

print("Loading v6 ...")
v6 = load_file(V6)
print(f"v6 keys: {len(v6)}")

print("Loading base ...")
base = load_file(BASE)
print(f"base keys: {len(base)}\n")

# Strip "pipe.dit." prefix from base if present (v6 was saved with prefix removed)
base = {k.removeprefix("pipe.dit."): v for k, v in base.items()}

shared = set(v6) & set(base)
v6_only = set(v6) - set(base)
base_only = set(base) - set(v6)

print(f"Shared keys:   {len(shared)}")
print(f"In v6 only:    {len(v6_only)}  (helios params, not in base)")
print(f"In base only:  {len(base_only)}  (base param NOT saved in v6 -> stays at base value when loading)")

if v6_only:
    print(f"\nv6-only keys (sample):")
    for k in sorted(v6_only)[:5]:
        print(f"  {k}")

if base_only:
    print(f"\n⚠️  base-only keys (sample) — these stay at base init when loading v6:")
    for k in sorted(base_only)[:10]:
        print(f"  {k}")

# Compare shared params: how much did v6 move from base?
print("\n" + "=" * 60)
print("How much did each shared param move in v6 vs base? (top 20 by L2 delta)")
print("=" * 60)
moved = []
for k in shared:
    a = v6[k].float()
    b = base[k].float()
    if a.shape != b.shape:
        moved.append((k, float('nan'), 'shape mismatch'))
        continue
    delta = (a - b).norm().item()
    rel   = delta / max(b.norm().item(), 1e-9)
    moved.append((k, delta, rel))

moved.sort(key=lambda x: -x[1] if not isinstance(x[2], str) else 0)
print(f"{'param':60s} {'L2 delta':>12s} {'rel':>10s}")
for k, d, r in moved[:20]:
    if isinstance(r, str):
        print(f"  {k:58s} {d!s:>12s} {r}")
    else:
        print(f"  {k:58s} {d:12.4f} {r:10.6f}")

print(f"\nMedian rel delta over {len(moved)} params: "
      f"{torch.tensor([r for _, _, r in moved if not isinstance(r, str)]).median().item():.6f}")
print("Bigger numbers = more training movement.")
