"""
FD-loss components for video DMD distillation.

Adapted from Jiawei-Yang/FD-loss (frechet_distance/losses.py, judges.py)
plus our chunk+VAE video setup.

Pipeline (in the generator's training step):
    latent(x_pred, with grad)
       └─ VAE.decode → [B, T, 3, H, W] pixels in [0,1]
                       │
       ┌──────────────┴───────────────┐
       │                              │
    queue (detached, K previous feats)│
       │       ┌───────────────────────┘
       │       │ DINOv2 feature extract per-frame
       │       │ → new_feats [B*T, D]  (with grad)
       │       ▼
       └──── concat → all_feats [K+B*T, D]
                       │
                       ▼
              compute_fd_loss(mu_ref, sigma_ref, all_feats)
                       │
                       ▼
              fd_loss = scalar (backprop through new_feats only)

queue.enqueue(new_feats.detach())  # after backward
"""

import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Feature queue: ring buffer of detached features, decouples population size
#                from batch size (FD-loss paper's key trick).
# ---------------------------------------------------------------------------
class FeatureQueue(nn.Module):
    """Holds up to `size` features of dim `feat_dim`. Older entries get
    overwritten ring-buffer style."""
    def __init__(self, size: int, feat_dim: int):
        super().__init__()
        self.size = int(size)
        self.feat_dim = int(feat_dim)
        # Stored on CPU by default; .cuda() will move them
        self.register_buffer("buf",   torch.zeros(self.size, self.feat_dim))
        self.register_buffer("ptr",   torch.zeros(1, dtype=torch.long))
        self.register_buffer("count", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def enqueue(self, new_feats: torch.Tensor):
        """new_feats: [N, D].  Always detached / no_grad context expected."""
        N = new_feats.shape[0]
        if N == 0:
            return
        new_feats = new_feats.detach().to(self.buf.dtype).to(self.buf.device)
        # Wrap-around copy in <= 2 contiguous chunks
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
        """Return only the populated portion of the buffer."""
        n = int(self.count.item())
        return self.buf[:n].detach()


# ---------------------------------------------------------------------------
# Fréchet distance (differentiable via gradient through input features)
# ---------------------------------------------------------------------------
def compute_fd_loss(mu_ref, sigma_ref, all_feats, sigma_ref_sqrt=None):
    """Differentiable Fréchet distance:
        ||mu - mu_ref||² + tr(sigma + sigma_ref - 2·sqrt(sigma · sigma_ref))
    Gradients flow through `all_feats` (the queue portion should be detached).

    Args:
        mu_ref, sigma_ref: precomputed reference stats on real data.
        all_feats: [N, D] tensor (mix of detached queue + grad-enabled new).
        sigma_ref_sqrt: optional precomputed sqrt(sigma_ref) — faster path.
    """
    n_samples = all_feats.shape[0]
    if n_samples < 2:
        return torch.tensor(1e6, device=all_feats.device,
                            dtype=torch.float32, requires_grad=True)

    feats64 = all_feats.double()
    mu      = feats64.mean(dim=0)
    centered = feats64 - mu
    sigma    = (centered.T @ centered) / (n_samples - 1)

    mu_ref     = mu_ref.to(dtype=torch.float64, device=feats64.device)
    sigma_ref  = sigma_ref.to(dtype=torch.float64, device=feats64.device)
    diff       = mu - mu_ref
    mean_term  = diff.dot(diff)

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


def precompute_sigma_ref_sqrt(sigma_ref: torch.Tensor) -> torch.Tensor:
    """One-time eigendecomposition of sigma_ref for the fast FD path."""
    sigma_ref = sigma_ref.to(torch.float64)
    eigvals, eigvecs = torch.linalg.eigh(sigma_ref)
    eigvals = torch.clamp(eigvals, min=0)
    return eigvecs @ torch.diag(eigvals.sqrt()) @ eigvecs.T


# ---------------------------------------------------------------------------
# Video feature extractor:  treat each decoded frame as one sample
# ---------------------------------------------------------------------------
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class VideoFeatureExtractor(nn.Module):
    """Frozen 2D backbone applied per-frame. Default: DINOv2 ViT-B/14 (768d).

    Forward input:  pixels in [0, 1], shape [B, T, 3, H, W].
    Forward output: features [B*T, feat_dim].
    """
    def __init__(self, model_name: str = "dinov2_vitb14",
                 resize_to: int = 224, dtype=torch.bfloat16):
        super().__init__()
        # Load via torch.hub (DINOv2 official models)
        if model_name.startswith("dinov2"):
            self.backbone = torch.hub.load("facebookresearch/dinov2", model_name)
        else:
            raise NotImplementedError(f"Unsupported feature model: {model_name}")
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.to(dtype=dtype)
        self.resize_to = int(resize_to)
        self.feat_dim = self.backbone.embed_dim   # DINOv2 ViT-B/14 → 768
        self.dtype = dtype

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        """pixels: [B, T, 3, H, W] in [0, 1]."""
        B, T, C, H, W = pixels.shape
        x = pixels.reshape(B * T, C, H, W).to(self.dtype)
        if H != self.resize_to or W != self.resize_to:
            x = F.interpolate(x.float(), size=(self.resize_to, self.resize_to),
                              mode="bilinear", align_corners=False).to(self.dtype)
        mean = _IMAGENET_MEAN.to(x.device).to(x.dtype)
        std  = _IMAGENET_STD .to(x.device).to(x.dtype)
        x = (x - mean) / std
        feats = self.backbone(x)              # [B*T, 768]
        return feats


# ---------------------------------------------------------------------------
# Reference-stats I/O
# ---------------------------------------------------------------------------
def load_fd_stats(path: str):
    """Load precomputed (mu_ref, sigma_ref) from an .npz file produced by
    precompute_fd_stats.py."""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"FD reference stats not found at {path}.  Run "
            f"`python examples/wanvideo/model_training/precompute_fd_stats.py` first."
        )
    data = np.load(path)
    mu_ref    = torch.from_numpy(data["mu"]).to(torch.float64)
    sigma_ref = torch.from_numpy(data["sigma"]).to(torch.float64)
    return mu_ref, sigma_ref


def save_fd_stats(path: str, mu, sigma, num_samples: int):
    """Save (mu, sigma) + provenance to .npz."""
    Path(os.path.dirname(path) or ".").mkdir(parents=True, exist_ok=True)
    np.savez(path,
             mu=np.asarray(mu, dtype=np.float64),
             sigma=np.asarray(sigma, dtype=np.float64),
             num_samples=int(num_samples))
