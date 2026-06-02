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
    def forward(ctx, tensor):
        ctx.rank = dist.get_rank()
        ctx.batch = tensor.shape[0]
        gathered = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, tensor.contiguous())
        # Replace own slot with the original (preserves autograd graph)
        gathered[ctx.rank] = tensor
        return torch.cat(gathered, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        n = ctx.batch
        return grad_output[ctx.rank * n : (ctx.rank + 1) * n].contiguous()


def diff_all_gather(tensor: torch.Tensor) -> torch.Tensor:
    """Cross-rank all-gather of features. No-op on single-GPU."""
    if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
        return tensor
    return _DiffAllGather.apply(tensor)


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
def _compute_fd_raw(mu_ref, sigma_ref, all_feats, sigma_ref_sqrt=None):
    """Differentiable Fréchet distance; returns scalar FD (NOT normalized)."""
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
        evals = torch.clamp(evals, min=0)
        tr_covmean = torch.sum(torch.sqrt(evals))
    else:
        product = sigma @ sigma_ref
        if not torch.isfinite(product).all():
            return torch.tensor(1e6, device=feats64.device, dtype=torch.float32)
        evals = torch.linalg.eigvals(product).real
        evals = torch.clamp(evals, min=0)
        tr_covmean = torch.sum(torch.sqrt(evals))

    trace_term = sigma.trace() + sigma_ref.trace() - 2.0 * tr_covmean
    return (mean_term + trace_term).float()


def compute_fd_loss_normalized(mu_ref, sigma_ref, all_feats,
                               sigma_ref_sqrt=None, eps: float = 1e-3):
    """FD loss with paper's normalization:  fid / (fid.detach() + eps).
    Returns (normalized_loss, raw_fid) so caller can log raw value."""
    fid_raw = _compute_fd_raw(mu_ref, sigma_ref, all_feats, sigma_ref_sqrt)
    fid_norm = fid_raw / (fid_raw.detach() + eps)
    return fid_norm, fid_raw.detach()


def precompute_sigma_ref_sqrt(sigma_ref: torch.Tensor) -> torch.Tensor:
    sigma_ref = sigma_ref.to(torch.float64)
    eigvals, eigvecs = torch.linalg.eigh(sigma_ref)
    eigvals = torch.clamp(eigvals, min=0)
    return eigvecs @ torch.diag(eigvals.sqrt()) @ eigvecs.T


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
