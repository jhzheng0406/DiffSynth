"""
Offline pre-compute of (mu_ref, sigma_ref) for FD-loss training.

Iterates over the cartoon training set, VAE-encodes + decodes each chunk's
frames (so reference features sit in the SAME domain as the student's
VAE-decoded outputs), passes them through DINOv2, accumulates features,
and finally computes mean + covariance → saves to .npz.

Usage:
    cd /mnt/vita/scratch/vita-students/users/jinghao/code/DiffSynth-Studio
    python examples/wanvideo/model_training/precompute_fd_stats.py \
        --metadata_path ./data/cartoon_15s/metadata.csv \
        --output ./data/cartoon_15s/fd_stats_dinov2.npz \
        --max_frames 8000
"""
import argparse, os, sys
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from fd_loss import VideoFeatureExtractor, save_fd_stats
from train_chunk_aware import ChunkAwareDataset


WAN_MODEL_CONFIGS = [
    ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-1.3B-Control",
                origin_file_pattern="Wan2.1_VAE.pth"),
]


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metadata_path", default="./data/cartoon_15s/metadata.csv")
    p.add_argument("--height", type=int, default=832)
    p.add_argument("--width",  type=int, default=480)
    p.add_argument("--num_frames", type=int, default=49)
    p.add_argument("--output", default="./data/cartoon_15s/fd_stats_dinov2.npz")
    p.add_argument("--max_frames", type=int, default=8000,
                   help="Stop after collecting this many frame-features.")
    p.add_argument("--feature_model", default="dinov2_vitb14")
    p.add_argument("--vae_roundtrip", action="store_true", default=True,
                   help="VAE encode+decode the real frames so reference features "
                        "live in the same domain as student outputs.")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = args.device
    dtype = torch.bfloat16

    print(f"[setup] loading DINOv2 ({args.feature_model}) ...")
    extractor = VideoFeatureExtractor(args.feature_model, dtype=dtype).to(device)

    # Lightweight pipe — only VAE needed
    print(f"[setup] loading Wan VAE ...")
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=dtype, device=device,
        model_configs=WAN_MODEL_CONFIGS,
    )
    pipe.vae.to(device)

    print(f"[setup] loading dataset {args.metadata_path}")
    dataset = ChunkAwareDataset(
        csv_path=args.metadata_path,
        height=args.height, width=args.width,
        chunk_frames=args.num_frames,
        recent_aug_strength=0.0,    # clean, no aug, for reference
        dataset_repeat=1,
    )
    print(f"[setup] dataset has {len(dataset)} chunks → up to "
          f"{len(dataset) * args.num_frames} frames total")

    # Online mean/cov accumulation (Welford for mean; sum of outer products for cov)
    feat_dim = extractor.feat_dim
    print(f"[setup] feature dim = {feat_dim}")
    sum_x   = torch.zeros(feat_dim, dtype=torch.float64, device=device)
    sum_xxT = torch.zeros(feat_dim, feat_dim, dtype=torch.float64, device=device)
    n_total = 0

    pbar = tqdm(total=args.max_frames, desc="frames")
    for idx in range(len(dataset)):
        if n_total >= args.max_frames:
            break
        batch = dataset[idx]
        # Real video frames (49 frames per chunk)
        video_pil = batch["video"]                              # list of PIL
        # → tensor [1, C, T, H, W] in pipe's normalization
        video = pipe.preprocess_video(video_pil).to(dtype=dtype, device=device)

        if args.vae_roundtrip:
            # encode → decode  (≈ student output domain)
            latents = pipe.vae.encode(video, device=device).to(dtype=dtype)
            decoded = pipe.vae.decode(latents, device=device).to(dtype=dtype)
            # pipe.vae.decode returns frames in [-1, 1]; rescale to [0, 1]
            pixels = (decoded.clamp(-1, 1) + 1) / 2
        else:
            pixels = (video.clamp(-1, 1) + 1) / 2

        # [1, 3, T, H, W] → [1, T, 3, H, W] for extractor
        pixels = pixels.permute(0, 2, 1, 3, 4)

        feats = extractor(pixels).to(torch.float64)              # [T, D]

        n = feats.shape[0]
        sum_x   += feats.sum(dim=0)
        sum_xxT += feats.T @ feats
        n_total += n
        pbar.update(n)
    pbar.close()

    if n_total < 2:
        raise RuntimeError(f"Only collected {n_total} features — need ≥ 2")

    mu = sum_x / n_total
    # Sample covariance (unbiased)  E[xxᵀ] − μμᵀ, rescaled (n / (n-1))
    sigma = (sum_xxT - n_total * torch.outer(mu, mu)) / (n_total - 1)
    sigma = 0.5 * (sigma + sigma.T)         # enforce symmetry

    print(f"[done] collected {n_total} features.  "
          f"|mu|={mu.norm().item():.4f}  tr(sigma)={sigma.trace().item():.4f}")

    save_fd_stats(args.output, mu.cpu().numpy(), sigma.cpu().numpy(), n_total)
    print(f"[save] {args.output}")


if __name__ == "__main__":
    main()
