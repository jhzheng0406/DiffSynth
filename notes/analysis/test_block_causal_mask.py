"""Verify the block-causal flex_attention path used by the causal-AR baseline:
  1. default (block_mask=None) == plain SDPA (working path untouched);
  2. block-causal flex == dense-masked SDPA reference (same numbers);
  3. the mask structure is right: ref prefix attends ref-only; target frame fi
     attends ref + target blocks <= fi; no future leakage.
Run: /home/jzheng/miniconda3/envs/diffsynth/bin/python notes/analysis/test_block_causal_mask.py
"""
import torch, torch.nn.functional as F
from einops import rearrange
from diffsynth.models.wan_video_dit import (
    flash_attention, build_block_causal_mask, _FLEX_AVAILABLE,
)

assert _FLEX_AVAILABLE, "flex_attention not available"
torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev)
H, Wd, nh, hd = 2, 3, 4, 16          # tiny: 6 tokens/frame, 4 heads, head_dim 16
S_f = H * Wd
T_ref, f, bf = 2, 4, 1               # 2 ref frames, 4 target frames, block=1
S = (T_ref + f) * S_f
dim = nh * hd

_dt = torch.bfloat16 if dev == "cuda" else torch.float32
q = torch.randn(1, S, dim, device=dev, dtype=_dt)
k = torch.randn(1, S, dim, device=dev, dtype=_dt)
v = torch.randn(1, S, dim, device=dev, dtype=_dt)

# ---- dense reference mask (the intended semantics) ----
fr = torch.arange(S, device=dev) // S_f
q_ref = fr < T_ref
allow = torch.zeros(S, S, dtype=torch.bool, device=dev)
for i in range(S):
    qf = fr[i].item(); qref = qf < T_ref
    for j in range(S):
        kf = fr[j].item(); kref = kf < T_ref
        if qref:
            allow[i, j] = kref                                  # ref → ref only
        else:
            qb = (qf - T_ref) // bf; kb = (kf - T_ref) // bf
            allow[i, j] = kref or (kb <= qb)                    # target → ref + causal blocks

def sdpa_dense(mask):
    qq = rearrange(q, "b s (n d) -> b n s d", n=nh).float()
    kk = rearrange(k, "b s (n d) -> b n s d", n=nh).float()
    vv = rearrange(v, "b s (n d) -> b n s d", n=nh).float()
    bias = torch.where(mask, 0.0, float("-inf")).view(1, 1, S, S)
    out = F.scaled_dot_product_attention(qq, kk, vv, attn_mask=bias)
    return rearrange(out, "b n s d -> b s (n d)", n=nh)

# 1. default path unchanged
ref_full = sdpa_dense(torch.ones(S, S, dtype=torch.bool, device=dev))
got_full = flash_attention(q, k, v, num_heads=nh).float()
print("1) default==full-SDPA  max abs diff:", (ref_full - got_full).abs().max().item())

# 2. block-causal flex == dense-masked SDPA
bm = build_block_causal_mask(T_ref, f, H, Wd, block_frames=bf, device=dev)
got_causal = flash_attention(q, k, v, num_heads=nh, block_mask=bm).float()
ref_causal = sdpa_dense(allow)
print("2) flex-causal==dense  max abs diff:", (ref_causal - got_causal).abs().max().item())

# 3. structural sanity: last target frame's allow row sees everything;
#    first target frame must NOT see later target frames.
last_q = (T_ref + f - 1) * S_f          # a token in last target frame
first_t = T_ref * S_f                    # a token in first target frame
print("3) last-target sees all keys:", bool(allow[last_q].all()))
print("   first-target blocks future target:",
      not bool(allow[first_t, (T_ref + 1) * S_f]))
print("   ref blocks any target key:", not bool(allow[0, T_ref * S_f]))
