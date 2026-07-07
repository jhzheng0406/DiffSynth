"""
FD-loss components — v2, fixing two bugs in fd_loss.py vs the paper:

  Bug 1 (CRITICAL):  loss magnitude not normalized.
      paper:   fd_loss = fid / (fid.detach() + eps)         (always ≈ 1)
      v1:      fd_loss = fid                                 (scale wild)

  Bug 2:  no cross-rank all_gather of features.
      paper:   features from ALL ranks pooled before FD     (stable stats)
      v1:      each rank computes FD on its own 8 features  (noisy)

This file is SELF-CONTAINED (no import from fd_loss.py) so it doesn't break
if v1 changes.  Reference: Jiawei-Yang/FD-loss frechet_distance/losses.py
"""
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Differentiable all-gather  (FD-loss losses.py _DiffAllGather)
# ---------------------------------------------------------------------------
class _DiffAllGather(torch.autograd.Function):
    """All-gather across ranks; backward routes grad only to the local chunk."""
    @staticmethod
    def forward(ctx, tensor, scale_backward: bool = True):
        ctx.rank = dist.get_rank()
        ctx.batch = tensor.shape[0]
        ctx.world_size = dist.get_world_size()
        ctx.scale_backward = bool(scale_backward)
        gathered = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, tensor.contiguous())
        # Replace own slot with the original (preserves autograd graph)
        gathered[ctx.rank] = tensor
        return torch.cat(gathered, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        n = ctx.batch
        grad = grad_output[ctx.rank * n : (ctx.rank + 1) * n].contiguous()
        if ctx.scale_backward:
            # The training scripts manually average parameter grads after
            # backward. Each rank computes the same global gathered loss but only
            # owns the local feature graph, so scale here to make the subsequent
            # all_reduce(mean) equal the desired sum of per-rank partial grads.
            grad = grad * ctx.world_size
        return grad, None


def diff_all_gather(tensor: torch.Tensor, scale_backward: bool = True) -> torch.Tensor:
    """Cross-rank all-gather of features. No-op on single-GPU."""
    if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
        return tensor
    return _DiffAllGather.apply(tensor, scale_backward)


# ---------------------------------------------------------------------------
# Feature queue (unchanged from v1)
# ---------------------------------------------------------------------------
class FeatureQueue(nn.Module):
    def __init__(self, size: int, feat_dim: int):
        super().__init__()
        self.size = int(size)
        self.feat_dim = int(feat_dim)
        self.register_buffer("buf",   torch.zeros(self.size, self.feat_dim))
        self.register_buffer("ptr",   torch.zeros(1, dtype=torch.long))
        self.register_buffer("count", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def enqueue(self, new_feats: torch.Tensor):
        N = new_feats.shape[0]
        if N == 0:
            return
        new_feats = new_feats.detach().to(self.buf.dtype).to(self.buf.device)
        ptr = int(self.ptr.item())
        end = ptr + N
        if end <= self.size:
            self.buf[ptr:end] = new_feats
        else:
            first = self.size - ptr
            self.buf[ptr:] = new_feats[:first]
            self.buf[: N - first] = new_feats[first:]
        self.ptr[0] = (ptr + N) % self.size
        self.count[0] = min(int(self.count.item()) + N, self.size)

    def get_valid(self) -> torch.Tensor:
        n = int(self.count.item())
        return self.buf[:n].detach()


# ---------------------------------------------------------------------------
# Fréchet distance (unchanged math from v1)
# ---------------------------------------------------------------------------
def _compute_fd_raw(mu_ref, sigma_ref, all_feats, sigma_ref_sqrt=None,
                    eval_clamp_min: float = 0.0):
    """Differentiable Fréchet distance; returns scalar FD (NOT normalized).

    eval_clamp_min: clamp floor for the eigenvalues of sigma·sigma_ref before
    sqrt. The backward of sqrt(λ) is 1/(2√λ) — near-zero eigenvalues (rank-
    deficient sigma from a small population) produce exploding gradients.
    Clamping kills the gradient for eigendirections below the floor instead
    (clamp backward is 0 below min). Default 0.0 keeps v2/v3 behavior."""
    n_samples = all_feats.shape[0]
    if n_samples < 2:
        return torch.tensor(1e6, device=all_feats.device,
                            dtype=torch.float32, requires_grad=True)

    feats64 = all_feats.double()
    mu = feats64.mean(dim=0)
    centered = feats64 - mu
    sigma = (centered.T @ centered) / (n_samples - 1)

    mu_ref    = mu_ref   .to(dtype=torch.float64, device=feats64.device)
    sigma_ref = sigma_ref.to(dtype=torch.float64, device=feats64.device)
    diff = mu - mu_ref
    mean_term = diff.dot(diff)

    if sigma_ref_sqrt is not None:
        S = sigma_ref_sqrt.to(dtype=torch.float64, device=feats64.device)
        M = S @ sigma @ S
        M = 0.5 * (M + M.T)
        evals = torch.linalg.eigvalsh(M)
        evals = torch.clamp(evals, min=eval_clamp_min)
        tr_covmean = torch.sum(torch.sqrt(evals))
    else:
        product = sigma @ sigma_ref
        if not torch.isfinite(product).all():
            return torch.tensor(1e6, device=feats64.device, dtype=torch.float32)
        evals = torch.linalg.eigvals(product).real
        evals = torch.clamp(evals, min=eval_clamp_min)
        tr_covmean = torch.sum(torch.sqrt(evals))

    trace_term = sigma.trace() + sigma_ref.trace() - 2.0 * tr_covmean
    return (mean_term + trace_term).float()


def compute_fd_loss_normalized(mu_ref, sigma_ref, all_feats,
                               sigma_ref_sqrt=None, eps: float = 1e-3,
                               eval_clamp_min: float = 0.0):
    """FD loss with paper's normalization:  fid / (fid.detach() + eps).
    Returns (normalized_loss, raw_fid) so caller can log raw value.
    eval_clamp_min is forwarded to _compute_fd_raw (see its docstring)."""
    fid_raw = _compute_fd_raw(mu_ref, sigma_ref, all_feats, sigma_ref_sqrt,
                              eval_clamp_min=eval_clamp_min)
    fid_norm = fid_raw / (fid_raw.detach() + eps)
    return fid_norm, fid_raw.detach()


def precompute_sigma_ref_sqrt(sigma_ref: torch.Tensor) -> torch.Tensor:
    sigma_ref = sigma_ref.to(torch.float64)
    eigvals, eigvecs = torch.linalg.eigh(sigma_ref)
    eigvals = torch.clamp(eigvals, min=0)
    return eigvecs @ torch.diag(eigvals.sqrt()) @ eigvecs.T


# ---------------------------------------------------------------------------
# Native-resolution random crops (the whole-frame resize path low-passes the
# frames 3.7x before the encoder, making the feature space nearly BLIND to
# blur; 224px crops at native resolution keep the high frequencies visible).
# ---------------------------------------------------------------------------
def sample_crop_offsets(B, T, K, H, W, crop_size, generator=None):
    """Random top-left corners for K crops per frame. Returns [B,T,K,2] (y,x)
    on CPU. Sample OUTSIDE torch.utils.checkpoint and pass in as an arg so the
    backward recompute crops the same regions."""
    if crop_size > H or crop_size > W:
        raise ValueError(
            f"crop_size={crop_size} exceeds frame size {H}x{W} — "
            f"lower --fd_crop_size (or the precompute --crop_size) below "
            f"min(H, W) for this resolution."
        )
    ys = torch.randint(0, H - crop_size + 1, (B, T, K), generator=generator)
    xs = torch.randint(0, W - crop_size + 1, (B, T, K), generator=generator)
    return torch.stack([ys, xs], dim=-1)


def crop_frames(pixels: torch.Tensor, offsets: torch.Tensor, crop_size: int):
    """pixels [B,T,3,H,W], offsets [B,T,K,2] → crops [B, T*K, 3, c, c].
    Plain slicing per crop (B*T*K is small); differentiable."""
    B, T, C, H, W = pixels.shape
    K = offsets.shape[2]
    out = []
    for b in range(B):
        for t in range(T):
            for k in range(K):
                y = int(offsets[b, t, k, 0])
                x = int(offsets[b, t, k, 1])
                out.append(pixels[b, t, :, y:y + crop_size, x:x + crop_size])
    return torch.stack(out, dim=0).reshape(B, T * K, C, crop_size, crop_size)


# ---------------------------------------------------------------------------
# RDM-lite / MMD objective
# ---------------------------------------------------------------------------
def estimate_rbf_bandwidth(feats: torch.Tensor, max_samples: int = 4096,
                           eps: float = 1e-6) -> torch.Tensor:
    """Median-heuristic RBF bandwidth on frozen reference features."""
    if feats.shape[0] < 2:
        return torch.tensor(1.0, device=feats.device, dtype=torch.float32)
    x = feats.float()
    if x.shape[0] > max_samples:
        x = x[:max_samples]
    d = torch.pdist(x, p=2)
    d = d[d > 0]
    if d.numel() == 0:
        return torch.tensor(1.0, device=feats.device, dtype=torch.float32)
    return d.median().clamp_min(eps)


def _rbf_from_sqdist(sqdist: torch.Tensor, bandwidth: torch.Tensor) -> torch.Tensor:
    bw2 = bandwidth.to(device=sqdist.device, dtype=sqdist.dtype).clamp_min(1e-12).pow(2)
    return torch.exp(-sqdist / (2.0 * bw2))


def compute_mmd_loss_normalized(
    ref_feats: torch.Tensor,
    gen_feats: torch.Tensor,
    bandwidth=None,
    ref_chunk: int = 4096,
    eps: float = 1e-3,
):
    """RDM-style MMD objective with exact generated repulsion and frozen real attraction.

    This is a practical RDM-lite path: it uses a fixed bank of real features from
    precompute_fd_stats.py instead of a full Nyström kernel mean. It drops the
    constant real-real term, so the raw objective can be negative; the normalized
    loss keeps that sign and only rescales the gradient magnitude.
    """
    if gen_feats.shape[0] < 1 or ref_feats.shape[0] < 1:
        raw = torch.tensor(0.0, device=gen_feats.device, dtype=torch.float32)
        return raw, raw.detach()

    gen = gen_feats.float()
    ref = ref_feats.to(device=gen.device, dtype=torch.float32)
    if bandwidth is None:
        bandwidth = estimate_rbf_bandwidth(ref)
    if not torch.is_tensor(bandwidth):
        bandwidth = torch.tensor(float(bandwidth), device=gen.device, dtype=torch.float32)
    else:
        bandwidth = bandwidth.to(device=gen.device, dtype=torch.float32)

    gg_sq = torch.cdist(gen, gen, p=2).pow(2)
    repulsion = _rbf_from_sqdist(gg_sq, bandwidth).mean()

    attraction_sum = gen.new_tensor(0.0)
    n_ref = ref.shape[0]
    ref_chunk = max(1, int(ref_chunk))
    for start in range(0, n_ref, ref_chunk):
        chunk = ref[start:start + ref_chunk]
        gr_sq = torch.cdist(gen, chunk, p=2).pow(2)
        attraction_sum = attraction_sum + _rbf_from_sqdist(gr_sq, bandwidth).sum()
    attraction = attraction_sum / (gen.shape[0] * n_ref)

    raw = repulsion - 2.0 * attraction
    norm = raw / (raw.detach().abs() + eps)
    return norm, raw.detach()


# ---------------------------------------------------------------------------
# Video feature extractor (unchanged from v1)
# ---------------------------------------------------------------------------
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class VideoFeatureExtractor(nn.Module):
    def __init__(self, model_name: str = "dinov2_vitb14",
                 resize_to: int = 224, dtype=torch.bfloat16):
        super().__init__()
        if model_name.startswith("dinov2"):
            self.backbone = torch.hub.load("facebookresearch/dinov2", model_name)
        else:
            raise NotImplementedError(f"Unsupported feature model: {model_name}")
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.to(dtype=dtype)
        self.resize_to = int(resize_to)
        self.feat_dim = self.backbone.embed_dim
        self.dtype = dtype

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = pixels.shape
        x = pixels.reshape(B * T, C, H, W).to(self.dtype)
        if H != self.resize_to or W != self.resize_to:
            x = F.interpolate(x.float(), size=(self.resize_to, self.resize_to),
                              mode="bilinear", align_corners=False).to(self.dtype)
        mean = _IMAGENET_MEAN.to(x.device).to(x.dtype)
        std  = _IMAGENET_STD .to(x.device).to(x.dtype)
        x = (x - mean) / std
        return self.backbone(x)


# ---------------------------------------------------------------------------
# Reference-stats I/O (unchanged; identical files between v1 and v2)
# ---------------------------------------------------------------------------
def load_fd_stats(path: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"FD reference stats not found at {path}")
    data = np.load(path)
    mu_ref    = torch.from_numpy(data["mu"]).to(torch.float64)
    sigma_ref = torch.from_numpy(data["sigma"]).to(torch.float64)
    return mu_ref, sigma_ref


def load_fd_reference_features(path: str, max_features: int = 0):
    """Returns (features, bandwidth, protocol). `protocol` is the dict saved
    by precompute_fd_stats.py describing HOW the features were extracted
    (crops_per_frame, crop_size, feature_model, vae_roundtrip, ...), or None
    for stats files predating protocol stamping. Callers should validate it —
    a whole-frame bank silently matched against crop features (or vice versa)
    pulls the student toward the wrong distribution."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"FD/RDM reference stats not found at {path}")
    data = np.load(path)
    if "features" not in data:
        raise KeyError(
            f"{path} has no 'features' array. Regenerate it with "
            "precompute_fd_stats.py from this code version (or use --fd_objective frechet)."
        )
    feats = torch.from_numpy(data["features"]).float()
    if max_features and max_features > 0:
        feats = feats[:max_features]
    bandwidth = None
    if "mmd_bandwidth" in data:
        bandwidth = torch.tensor(float(data["mmd_bandwidth"]), dtype=torch.float32)
    protocol = None
    if "protocol" in data:
        protocol = json.loads(str(data["protocol"]))
    return feats, bandwidth, protocol


def save_fd_stats(path: str, mu, sigma, num_samples: int,
                  features=None, mmd_bandwidth=None, protocol: dict = None):
    Path(os.path.dirname(path) or ".").mkdir(parents=True, exist_ok=True)
    payload = {
        "mu": np.asarray(mu, dtype=np.float64),
        "sigma": np.asarray(sigma, dtype=np.float64),
        "num_samples": int(num_samples),
    }
    if features is not None:
        payload["features"] = np.asarray(features, dtype=np.float32)
    if mmd_bandwidth is not None:
        payload["mmd_bandwidth"] = np.asarray(float(mmd_bandwidth), dtype=np.float32)
    if protocol is not None:
        payload["protocol"] = np.asarray(json.dumps(protocol))
    np.savez(path, **payload)
