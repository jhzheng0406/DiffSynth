"""
Sanity check for the First-Frame Anchor + Relative RoPE refactor.

Verifies:
  1. prepare_helios_history with anchor returns lat_short of shape [B,C,2,H,W]
     and fids_short = [0, 1+sizes[0]+sizes[1]] (non-contiguous).
  2. split_into_helios_history_and_target picks the very first latent of the
     clip as anchor and slices target after position 1+total_hist.
  3. The two paths produce IDENTICAL frame-id layouts so train/inference match.
  4. Anchor identity is preserved (anchor latent != recent latent).
  5. Without anchor, layouts revert to legacy contiguous behavior.
"""
import sys
import torch

sys.path.insert(0, "/mnt/vita/scratch/vita-students/users/jinghao/code/DiffSynth-Studio")
from diffsynth.models.wan_video_helios_attention import (
    prepare_helios_history,
    split_into_helios_history_and_target,
)

torch.manual_seed(0)

SIZES = (4, 2, 1)              # matches user's training script
TOTAL = sum(SIZES)             # 7
B, C, H, W = 1, 16, 32, 32

# ---------------------------------------------------------------------------
# Test 1: inference path with anchor — accumulated 3 chunks, anchor preserved
# ---------------------------------------------------------------------------
chunk0 = torch.randn(B, C, 13, H, W) * 0.5 + 100  # marker: high mean
chunk1 = torch.randn(B, C, 11, H, W) * 0.5 + 50
chunk2 = torch.randn(B, C, 11, H, W) * 0.5 + 10

accumulated = [chunk0, chunk1, chunk2]
lat_long, lat_mid, lat_short, fids_long, fids_mid, fids_short = prepare_helios_history(
    accumulated, history_sizes=SIZES, use_first_frame_anchor=True,
)

print("=== Inference (with anchor) ===")
print(f"lat_long.shape  = {tuple(lat_long.shape)}    fids_long  = {fids_long.tolist()}")
print(f"lat_mid.shape   = {tuple(lat_mid.shape)}     fids_mid   = {fids_mid.tolist()}")
print(f"lat_short.shape = {tuple(lat_short.shape)}    fids_short = {fids_short.tolist()}")
assert lat_long.shape  == (B, C, 4, H, W),  "long should be 4 frames"
assert lat_mid.shape   == (B, C, 2, H, W),  "mid should be 2 frames"
assert lat_short.shape == (B, C, 2, H, W),  "short should be 2 frames (anchor + recent)"
assert fids_long.tolist()  == [1, 2, 3, 4],  "long fids must be [1..4]"
assert fids_mid.tolist()   == [5, 6],        "mid fids must be [5..6]"
assert fids_short.tolist() == [0, 7],        "short fids must be [0, 7] (non-contiguous)"

# Anchor must be the very first latent of chunk0 (mean ~100)
anchor = lat_short[:, :, 0]
recent = lat_short[:, :, 1]
print(f"  anchor mean = {anchor.mean().item():.2f}  (expected ~100, from chunk0[0])")
print(f"  recent mean = {recent.mean().item():.2f}  (expected ~10,  from end of chunk2)")
assert anchor.mean().item() > 80, "anchor should be from chunk0 (mean ~100)"
assert recent.mean().item() < 30, "recent should be from chunk2 (mean ~10)"
assert not torch.allclose(anchor, recent), "anchor and recent must be distinct latents"
print("  ✓ inference: anchor preserved from chunk0, recent from chunk2\n")

# ---------------------------------------------------------------------------
# Test 2: training split with anchor
# ---------------------------------------------------------------------------
# Need 1 + total_hist + window latents. With sizes=[4,2,1] total_hist=7, so need 1+7+window.
# Use window=12 → need 20 latents.
F = 20
clip = torch.arange(F, dtype=torch.float32).view(1, 1, F, 1, 1).expand(B, C, F, H, W).clone()

target_lat, lat_long, lat_mid, lat_short, fids_long, fids_mid, fids_short, fids_target = (
    split_into_helios_history_and_target(
        clip, history_sizes=SIZES, latent_window_size=12,
        is_random_drop=False, use_first_frame_anchor=True,
    )
)

print("=== Training split (with anchor) ===")
print(f"lat_long.shape  = {tuple(lat_long.shape)}    fids_long  = {fids_long.tolist()}")
print(f"lat_mid.shape   = {tuple(lat_mid.shape)}     fids_mid   = {fids_mid.tolist()}")
print(f"lat_short.shape = {tuple(lat_short.shape)}    fids_short = {fids_short.tolist()}")
print(f"target.shape    = {tuple(target_lat.shape)}    fids_target = {fids_target.tolist()}")

# Each latent's "value" is its original temporal index (we encoded that into the data)
def src_pos(t):
    return int(t[0, 0, 0, 0, 0].item())
print(f"  anchor source position = {src_pos(lat_short[:, :, :1])}  (expected 0)")
print(f"  recent source position = {src_pos(lat_short[:, :, 1:])}  (expected 7)")
print(f"  long source positions  = {[src_pos(lat_long[:, :, i:i+1]) for i in range(4)]}  (expected [1,2,3,4])")
print(f"  mid source positions   = {[src_pos(lat_mid[:, :, i:i+1])  for i in range(2)]}  (expected [5,6])")
print(f"  target source positions= [{src_pos(target_lat[:, :, :1])}..{src_pos(target_lat[:, :, -1:])}]  (expected [8..19])")

assert fids_long.tolist()   == [1, 2, 3, 4]
assert fids_mid.tolist()    == [5, 6]
assert fids_short.tolist()  == [0, 7]
assert src_pos(lat_short[:, :, :1])  == 0  # anchor = first latent
assert src_pos(lat_short[:, :, 1:])  == 7  # recent = position 7
assert src_pos(target_lat[:, :, :1]) == 8  # target starts at 8
assert src_pos(target_lat[:, :, -1:]) == 19
print("  ✓ training: anchor at position 0, target shifted to start at 8\n")

# ---------------------------------------------------------------------------
# Test 3: layouts match between inference and training
# ---------------------------------------------------------------------------
# The fids returned by both functions should be identical for the same sizes.
# We already asserted both produce [1..4] / [5..6] / [0,7]. ✓
print("=== Layout consistency ===")
print("  ✓ inference fids_short == training fids_short == [0, 7]")
print("  ✓ inference fids_long  == training fids_long  == [1, 2, 3, 4]\n")

# ---------------------------------------------------------------------------
# Test 4: legacy mode still works
# ---------------------------------------------------------------------------
lat_long, lat_mid, lat_short, fids_long, fids_mid, fids_short = prepare_helios_history(
    accumulated, history_sizes=SIZES, use_first_frame_anchor=False,
)
print("=== Inference (legacy, no anchor) ===")
print(f"lat_short.shape = {tuple(lat_short.shape)}    fids_short = {fids_short.tolist()}")
assert lat_short.shape == (B, C, 1, H, W),  "legacy short should be 1 frame"
assert fids_short.tolist() == [6],          "legacy short fid must be [6]"
assert fids_long.tolist()  == [0, 1, 2, 3]
assert fids_mid.tolist()   == [4, 5]
print("  ✓ legacy mode unchanged (contiguous fids 0..6)\n")

print("ALL TESTS PASSED ✓")
