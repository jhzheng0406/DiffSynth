"""
Check how history_key_scale evolves across Helios checkpoints.
Run from DiffSynth-Studio repo root:
    python check_helios_scale.py
"""

import torch
from safetensors.torch import load_file
from pathlib import Path

CKPT_DIR = "models/train/Wan2.1-Fun-V1.1-1.3B-Control-Helios-v6-eadtest"  # Adjust if your checkpoints are in a different location
# Check every N steps to keep output manageable (set to 1 to show all early checkpoints)
STEP_INTERVAL = 1

ckpt_paths = sorted(
    Path(CKPT_DIR).glob("step-*.safetensors"),
    key=lambda p: int(p.stem.split("-")[1]),
)
ckpt_paths = [p for p in ckpt_paths if int(p.stem.split("-")[1]) % STEP_INTERVAL == 0]

print(f"{'Step':>6}  {'mean logit':>14}  {'mean scale':>14}  {'min scale':>14}  {'max scale':>14}")
print("-" * 68)

for path in ckpt_paths:
    step = int(path.stem.split("-")[1])
    ckpt = load_file(str(path))
    logits = torch.stack([v.cpu().float() for k, v in ckpt.items() if "history_key_scale" in k])
    scales = torch.sigmoid(logits)
    print(
        f"{step:>6}  "
        f"{logits.mean().item():>14.7f}  "
        f"{scales.mean().item():>14.7f}  "
        f"{scales.min().item():>14.7f}  "
        f"{scales.max().item():>14.7f}"
    )

print()
print(f"Total history_key_scale params: {logits.numel()}")
print(f"Init baseline: logit=-3.0  →  scale={torch.sigmoid(torch.tensor(-3.0)).item():.4f}")
