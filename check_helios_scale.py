"""
Track how Helios trainable params evolve across checkpoints.

Reports per checkpoint:
  - history_key_scale: mean / std spread / range / drift-from-init
  - patch_helios_{short,mid,long}.weight: drift-from-init (L2 of weight delta)

The "drift-from-init" lets you confirm parameters are actually being optimized
(not stuck at init dtype-rounding noise).

Run from DiffSynth-Studio repo root:
    python check_helios_scale.py [CKPT_DIR] [INIT_LOGIT]

Examples:
    python check_helios_scale.py
    python check_helios_scale.py models/train/Wan2.1-Fun-V1.1-1.3B-Control-Helios-v7-soft -5.0
"""

import sys
import torch
from safetensors.torch import load_file
from pathlib import Path

# Defaults — override via argv
CKPT_DIR = "models/train/Wan2.1-Fun-V1.1-1.3B-Control-Helios-v7-soft"
INIT_LOGIT = -5.0   # the value you passed to --helios_init_scale_logit
MAX_HISTORY_SCALE = 10.0   # patch default
STEP_INTERVAL = 1   # show every Nth step

if len(sys.argv) > 1:
    CKPT_DIR = sys.argv[1]
if len(sys.argv) > 2:
    INIT_LOGIT = float(sys.argv[2])

INIT_SIGMOID = torch.sigmoid(torch.tensor(INIT_LOGIT)).item()
INIT_FINAL_SCALE = 1.0 + INIT_SIGMOID * (MAX_HISTORY_SCALE - 1.0)

# ----------------------------------------------------------------------------
ckpt_paths = sorted(
    Path(CKPT_DIR).glob("step-*.safetensors"),
    key=lambda p: int(p.stem.split("-")[1]),
)
ckpt_paths = [p for p in ckpt_paths if int(p.stem.split("-")[1]) % STEP_INTERVAL == 0]

if not ckpt_paths:
    print(f"No step-*.safetensors found in {CKPT_DIR}")
    sys.exit(1)

print(f"CKPT_DIR    : {CKPT_DIR}")
print(f"INIT_LOGIT  : {INIT_LOGIT}  ->  sigmoid={INIT_SIGMOID:.4f}  ->  final_scale={INIT_FINAL_SCALE:.4f}")
print(f"Found {len(ckpt_paths)} checkpoints\n")

# Header
print(f"{'step':>6} | "
      f"{'logit_mean':>12s} {'logit_std':>10s} {'logit_range':>20s}  | "
      f"{'final_scale_mean':>18s} {'fs_max':>10s}  | "
      f"{'long_drift':>10s} {'mid_drift':>10s} {'short_drift':>11s}")
print("-" * 130)

# Cache earliest checkpoint's patch_helios_*.weight to compute drift
init_phl = init_phm = init_phs = None

for path in ckpt_paths:
    step = int(path.stem.split("-")[1])
    ckpt = load_file(str(path))

    # ── history_key_scale ───────────────────────────────────────────
    hks_keys = sorted([k for k in ckpt if "history_key_scale" in k])
    logits = torch.stack([ckpt[k].cpu().float() for k in hks_keys])  # [n_blocks, n_heads]
    sigmoids = torch.sigmoid(logits)
    final = 1.0 + sigmoids * (MAX_HISTORY_SCALE - 1.0)

    logit_mean = logits.mean().item()
    logit_std  = logits.std().item()
    logit_min  = logits.min().item()
    logit_max  = logits.max().item()
    fs_mean    = final.mean().item()
    fs_max     = final.max().item()

    # Drift from init logit (positive = moved up = stronger attn to history)
    drift_from_init = logit_mean - INIT_LOGIT
    drift_marker = " ↑" if drift_from_init > 0.01 else (" ↓" if drift_from_init < -0.01 else " =")

    # ── patch_helios_{short,mid,long}.weight drift ─────────────────
    phl = ckpt.get("patch_helios_long.weight",  torch.zeros(1)).float()
    phm = ckpt.get("patch_helios_mid.weight",   torch.zeros(1)).float()
    phs = ckpt.get("patch_helios_short.weight", torch.zeros(1)).float()

    if init_phl is None:
        init_phl, init_phm, init_phs = phl.clone(), phm.clone(), phs.clone()
        long_drift = mid_drift = short_drift = 0.0
    else:
        long_drift  = (phl - init_phl).norm().item()
        mid_drift   = (phm - init_phm).norm().item()
        short_drift = (phs - init_phs).norm().item()

    print(f"{step:>6} | "
          f"{logit_mean:>+12.5f} {logit_std:>10.5f} "
          f"[{logit_min:>+8.5f},{logit_max:>+8.5f}]  | "
          f"{fs_mean:>18.4f} {fs_max:>10.4f}  | "
          f"{long_drift:>10.4f} {mid_drift:>10.4f} {short_drift:>11.4f}"
          + drift_marker)

print()
print(f"Init values:  logit={INIT_LOGIT:.4f}  sigmoid={INIT_SIGMOID:.4f}  final_scale={INIT_FINAL_SCALE:.4f}")
print(f"Reading guide:")
print(f"  - logit_mean rising         => model wants STRONGER attention to helios history")
print(f"  - logit_std rising          => different blocks specialize differently")
print(f"  - patch_helios_*_drift > 0  => convs are actually being optimized (vs stuck at init)")
print(f"  - if logit stuck at {INIT_LOGIT:.3f} after many steps, optimizer may be missing those params")
