"""
Diagnose the v6 trained checkpoint to figure out why inference is noise.

Checks:
  1. State-dict load: are there missing/unexpected keys?
  2. Helios-specific weight stats: did training actually move them?
  3. patch_helios_long sample output: is it producing reasonable activations?
  4. history_key_scale post-sigmoid value: is it sane (0.01..1.0 range)?

Run from DiffSynth-Studio repo root:
    python test_v6_ckpt.py
"""
import sys
import torch

sys.path.insert(0, "/mnt/vita/scratch/vita-students/users/jinghao/code/DiffSynth-Studio")
from safetensors.torch import load_file

CKPT = "models/train/Wan2.1-Fun-V1.1-1.3B-Control-Helios-v6-eadtest/step-200.safetensors"

print(f"Loading: {CKPT}\n")
sd = load_file(CKPT)
print(f"Total keys: {len(sd)}\n")

# ---------------------------------------------------------------------------
# 1. Helios-specific keys present?
# ---------------------------------------------------------------------------
print("=" * 60)
print("1. Helios-specific keys in checkpoint:")
print("=" * 60)
helios_keys = [k for k in sd if 'patch_helios' in k or 'history_key_scale' in k]
print(f"Found {len(helios_keys)} helios keys")
for k in helios_keys[:6]:
    print(f"  {k:60s} shape={tuple(sd[k].shape)}")
if len(helios_keys) > 6:
    print(f"  ... and {len(helios_keys) - 6} more")

# ---------------------------------------------------------------------------
# 2. Helios weight stats — are they trained or stuck at init?
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("2. Weight stats:")
print("=" * 60)

for kind in ['patch_helios_short', 'patch_helios_mid', 'patch_helios_long']:
    weights = {k: v for k, v in sd.items() if kind in k and 'weight' in k}
    for k, v in weights.items():
        v = v.float()
        print(f"  {k:50s} mean={v.mean().item():+.5f} std={v.std().item():.5f} "
              f"abs_max={v.abs().max().item():.4f}")

# history_key_scale (one per block, all should have similar magnitudes)
hks_keys = sorted([k for k in sd if 'history_key_scale' in k])
print(f"\n  history_key_scale ({len(hks_keys)} blocks):")
if hks_keys:
    logits = torch.stack([sd[k].float() for k in hks_keys])  # [n_blocks, n_heads]
    scales = torch.sigmoid(logits)                            # [0, 1]
    final  = 1.0 + scales * (10.0 - 1.0)                      # match get_scale_key formula

    print(f"    raw logit  mean={logits.mean().item():+.4f}  range=[{logits.min().item():+.4f}, {logits.max().item():+.4f}]")
    print(f"    sigmoid    mean={scales.mean().item():.4f}   range=[{scales.min().item():.4f}, {scales.max().item():.4f}]")
    print(f"    final scale mean={final.mean().item():.4f}  range=[{final.min().item():.4f}, {final.max().item():.4f}]")
    print(f"    (init was logit=-3.0 → sigmoid=0.047 → final scale=1.42)")
    if abs(logits.mean().item() - (-3.0)) < 0.01:
        print(f"    ⚠️  history_key_scale stuck at init — optimizer never updated it!")
    else:
        print(f"    ✓ history_key_scale moved from init")

# ---------------------------------------------------------------------------
# 3. NaN / Inf check
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("3. NaN / Inf check:")
print("=" * 60)
nan_keys = [k for k, v in sd.items() if torch.isnan(v).any() or torch.isinf(v).any()]
if nan_keys:
    print(f"  ❌ {len(nan_keys)} keys with NaN/Inf:")
    for k in nan_keys[:10]:
        print(f"    {k}")
else:
    print("  ✓ no NaN / Inf anywhere")

# ---------------------------------------------------------------------------
# 4. Compare patch_helios_long vs base patch_embedding scale
# ---------------------------------------------------------------------------
# We loaded the FT'd state. Patch_embedding is in state too (shared with base Wan).
print("\n" + "=" * 60)
print("4. Scale comparison (patch_helios_* vs patch_embedding):")
print("=" * 60)

if 'patch_embedding.weight' in sd:
    pe = sd['patch_embedding.weight'].float()
    print(f"  patch_embedding.weight     mean={pe.mean().item():+.5f}  std={pe.std().item():.5f}")
    for kind in ['patch_helios_short', 'patch_helios_mid', 'patch_helios_long']:
        wkey = f'{kind}.weight'
        if wkey in sd:
            w = sd[wkey].float()
            ratio_std = w.std().item() / max(pe.std().item(), 1e-9)
            tag = "OK" if 0.1 < ratio_std < 10 else "⚠️  out of expected scale"
            print(f"  {kind:30s} mean={w.mean().item():+.5f}  std={w.std().item():.5f}  "
                  f"std_ratio_vs_pe={ratio_std:.3f}  [{tag}]")
else:
    print("  patch_embedding.weight not in ckpt — checkpoint may only contain trainable subset")

print("\nDone.")
