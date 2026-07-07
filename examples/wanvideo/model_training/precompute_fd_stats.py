"""
Offline pre-compute of (mu_ref, sigma_ref) for FD-loss training.

Iterates over the cartoon training set, VAE-encodes + decodes each chunk's
frames (so reference features sit in the SAME domain as the student's
VAE-decoded outputs), passes them through DINOv2, accumulates features,
and finally computes mean + covariance → saves to .npz.

v2 changes (diversity fixes — the reference stats are the training signal's
real-data anchor, so Σ_ref must reflect cross-video variance, not the
within-video variance of a few consecutive clips):
  - chunks visited in SHUFFLED order (--shuffle_chunks, default on) instead
    of sequentially → the max_frames budget spans many videos;
  - within each chunk only every --frame_stride-th decoded frame contributes
    a feature (default auto: smallest stride that spans the whole dataset
    within the budget; 1 for small datasets). The VAE roundtrip is still
    FULL-chunk (causal VAE; matches the training-side decode protocol of
    train_dmd_fd_v4.py) — the stride applies in pixel space after decode;
  - default budget raised 8000 → 20000 frame-features.

Usage (v2 stats for train_dmd_fd_v4.py):
    cd /mnt/vita/scratch/vita-students/users/jinghao/code/DiffSynth-Studio
    python examples/wanvideo/model_training/precompute_fd_stats.py \
        --metadata_path ./data/cartoon_15s/metadata.csv \
        --output ./data/cartoon_15s/fd_stats_dinov2_v2.npz \
        --max_frames 20000

Multi-GPU (same stats, ~N× faster — chunks are sharded, fp64 accumulators
all-reduced at the end, rank 0 saves):
    torchrun --nproc_per_node=2 \
        examples/wanvideo/model_training/precompute_fd_stats.py \
        --metadata_path ./data/cartoon_15s/metadata.csv \
        --output ./data/cartoon_15s/fd_stats_dinov2_v2.npz \
        --max_frames 20000

Legacy (v1-v3 file, sequential + every frame):
    ... --output ./data/cartoon_15s/fd_stats_dinov2.npz \
        --max_frames 8000 --frame_stride 1 --no_shuffle_chunks
"""
import argparse, os, random, sys
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from fd_loss_v2 import (VideoFeatureExtractor, save_fd_stats,
                        estimate_rbf_bandwidth, sample_crop_offsets, crop_frames)
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
    p.add_argument("--output", default="./data/cartoon_15s/fd_stats_dinov2_v2.npz")
    p.add_argument("--max_frames", type=int, default=20000,
                   help="Stop after collecting this many frame-features.")
    p.add_argument("--frame_stride", type=int, default=0,
                   help="Keep every Nth decoded frame as a feature sample "
                        "(pixel space, AFTER the full-chunk VAE roundtrip). "
                        ">1 spends the max_frames budget across more chunks "
                        "→ more diverse Σ_ref. 0 (default) = AUTO: smallest "
                        "stride that still lets the budget span the WHOLE "
                        "dataset (stride 1 when the dataset is smaller than "
                        "the budget — never discard frames without a "
                        "diversity gain).")
    p.add_argument("--shuffle_chunks", dest="shuffle_chunks",
                   action="store_true", default=True,
                   help="Visit dataset chunks in shuffled order so the budget "
                        "spans many videos (default on).")
    p.add_argument("--no_shuffle_chunks", dest="shuffle_chunks",
                   action="store_false")
    p.add_argument("--seed", type=int, default=0,
                   help="Shuffle seed (reproducible stats).")
    p.add_argument("--crops_per_frame", type=int, default=0,
                   help="If >0, feed K random NATIVE-RESOLUTION crops per kept "
                        "frame to the encoder instead of the squash-resized "
                        "whole frame. The resize path low-passes 832x480 → "
                        "224² (3.7×), so its features are nearly blind to "
                        "blur; crops keep high frequencies visible. Must "
                        "match the training-side --fd_crops_per_frame "
                        "protocol (crop stats file).")
    p.add_argument("--crop_size", type=int, default=224)
    p.add_argument("--feature_model", default="dinov2_vitb14")
    p.add_argument("--save_features", dest="save_features",
                   action="store_true", default=True,
                   help="Also store a frozen real feature bank for --fd_objective mmd.")
    p.add_argument("--no_save_features", dest="save_features",
                   action="store_false")
    p.add_argument("--max_ref_features", type=int, default=20000,
                   help="Maximum number of reference features saved for "
                        "MMD/RDM-lite. Must be >= 2 (bandwidth needs pairs).")
    p.add_argument("--vae_roundtrip", action="store_true", default=True,
                   help="VAE encode+decode the real frames so reference features "
                        "live in the same domain as student outputs.")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if args.save_features and args.max_ref_features < 2:
        raise ValueError("--max_ref_features must be >= 2 (the RBF bandwidth "
                         "is a median over feature PAIRS); got "
                         f"{args.max_ref_features}")

    # Multi-GPU via torchrun: each rank takes an interleaved shard of the
    # identically-shuffled chunk order; per-chunk frame RNG is seeded by the
    # chunk index, so the collected feature SET is independent of world size.
    rank, world = 0, 1
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        import torch.distributed as dist
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        args.device = f"cuda:{local_rank}"
    is_main = rank == 0

    device = args.device
    dtype = torch.bfloat16

    if is_main:
        print(f"[setup] world_size={world}")
        print(f"[setup] loading DINOv2 ({args.feature_model}) ...")
    # torch.hub download is NOT multi-process safe (concurrent zip extract +
    # rmtree in one cache dir → 'Directory not empty'). Rank 0 downloads
    # first; the rest wait at the barrier and then load from the warm cache.
    if world > 1 and not is_main:
        dist.barrier()
    extractor = VideoFeatureExtractor(args.feature_model, dtype=dtype).to(device)
    if world > 1 and is_main:
        dist.barrier()

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
    ref_feature_chunks = []

    if args.frame_stride <= 0:
        # Auto: smallest stride whose per-chunk yield lets the budget cover
        # every chunk. Diversity is capped by the number of clips, so with a
        # small dataset (chunks*frames <= budget) stride must be 1. With
        # crops, each kept frame yields K features — account for that.
        feats_per_frame = max(1, args.crops_per_frame)
        total_feats = len(dataset) * args.num_frames * feats_per_frame
        args.frame_stride = max(1, -(-total_feats // args.max_frames))  # ceil
        print(f"[setup] auto frame_stride={args.frame_stride} "
              f"({len(dataset)} chunks × {args.num_frames} frames × "
              f"{feats_per_frame} feats/frame = {total_feats} vs budget "
              f"{args.max_frames})")

    order = list(range(len(dataset)))
    if args.shuffle_chunks:
        random.Random(args.seed).shuffle(order)

    local_budget = -(-args.max_frames // world)
    order = order[rank::world]
    pbar = tqdm(total=local_budget, desc=f"frames/rank{rank}", disable=not is_main)
    for idx in order:
        if n_total >= local_budget:
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
        if args.frame_stride > 1:
            # Thin in PIXEL space (roundtrip above stays full-chunk so the
            # decode domain matches training exactly). RANDOM subset, not a
            # fixed-phase strided slice: stride 4 equals the VAE's 4x temporal
            # block, so pixels[:, ::4] would always sample the same intra-
            # block position (position-dependent reconstruction artifacts →
            # phase-biased mu_ref/sigma_ref), and the training side draws
            # random frames (train_dmd_fd_v4.py randperm). Seeded per chunk
            # for reproducible stats.
            T = pixels.shape[1]
            n_keep = -(-T // args.frame_stride)          # ceil
            g = torch.Generator().manual_seed(args.seed * 1_000_003 + idx)
            fidx = torch.randperm(T, generator=g)[:n_keep].to(pixels.device)
            pixels = pixels.index_select(1, fidx)

        if args.crops_per_frame > 0:
            # Native-resolution random crops (seeded per chunk); 224px crops
            # skip the extractor's resize entirely → blur stays visible.
            gc_ = torch.Generator().manual_seed(args.seed * 7_000_003 + idx)
            offs = sample_crop_offsets(
                pixels.shape[0], pixels.shape[1], args.crops_per_frame,
                pixels.shape[3], pixels.shape[4], args.crop_size, generator=gc_,
            )
            pixels = crop_frames(pixels, offs, args.crop_size)   # [1, T'*K, 3, c, c]

        feats = extractor(pixels).to(torch.float64)              # [T'(*K), D]

        n = feats.shape[0]
        sum_x   += feats.sum(dim=0)
        sum_xxT += feats.T @ feats
        if args.save_features:
            # Collect EVERYTHING within the budget; the bank is subsampled
            # uniformly (seeded randperm) at save time. Capping during
            # collection would fill the bank from whichever chunks (or, in
            # multi-GPU, whichever ranks) come first — a biased bank.
            ref_feature_chunks.append(feats.float().cpu())
        n_total += n
        pbar.update(n)
    pbar.close()

    n_total_t = torch.tensor([n_total], dtype=torch.long, device=device)
    if world > 1:
        import torch.distributed as dist
        dist.all_reduce(sum_x, op=dist.ReduceOp.SUM)
        dist.all_reduce(sum_xxT, op=dist.ReduceOp.SUM)
        dist.all_reduce(n_total_t, op=dist.ReduceOp.SUM)
    n_total = int(n_total_t.item())

    if n_total < 2:
        raise RuntimeError(f"Only collected {n_total} features — need ≥ 2")

    mu = sum_x / n_total
    # Sample covariance (unbiased)  E[xxᵀ] − μμᵀ, rescaled (n / (n-1))
    sigma = (sum_xxT - n_total * torch.outer(mu, mu)) / (n_total - 1)
    sigma = 0.5 * (sigma + sigma.T)         # enforce symmetry

    ref_features = None
    mmd_bandwidth = None
    if args.save_features:
        local_ref = (torch.cat(ref_feature_chunks, dim=0)
                     if ref_feature_chunks else torch.empty(0, feat_dim))
        if world > 1:
            import torch.distributed as dist
            local_n = torch.tensor([local_ref.shape[0]], dtype=torch.long, device=device)
            sizes = [torch.zeros_like(local_n) for _ in range(world)]
            dist.all_gather(sizes, local_n)
            max_n = max(int(x.item()) for x in sizes)
            send = torch.zeros(max_n, feat_dim, dtype=torch.float32, device=device)
            if local_ref.numel() > 0:
                send[:local_ref.shape[0]] = local_ref.to(device)
            gathered = [torch.zeros_like(send) for _ in range(world)]
            dist.all_gather(gathered, send)
            if is_main:
                ref_features = torch.cat(
                    [g[:int(sz.item())].cpu() for g, sz in zip(gathered, sizes)], dim=0
                )
        else:
            ref_features = local_ref
        if is_main and ref_features is not None:
            if ref_features.shape[0] > args.max_ref_features:
                # Uniform seeded subsample over the FULL pooled set — a plain
                # [:max] slice would keep whole rank-shards / early chunks and
                # bias the bank.
                g = torch.Generator().manual_seed(args.seed + 987_654_321)
                sel = torch.randperm(ref_features.shape[0], generator=g)[:args.max_ref_features]
                ref_features = ref_features[sel]
            if ref_features.shape[0] >= 2:
                mmd_bandwidth = float(estimate_rbf_bandwidth(ref_features).item())

    if is_main:
        print(f"[done] collected {n_total} features.  "
              f"|mu|={mu.norm().item():.4f}  tr(sigma)={sigma.trace().item():.4f}")
        if ref_features is not None:
            bw_str = "n/a" if mmd_bandwidth is None else f"{mmd_bandwidth:.4f}"
            print(f"[done] saved feature bank: {tuple(ref_features.shape)}, "
                  f"mmd_bandwidth={bw_str}")
        # Protocol stamp: the training side validates this so a whole-frame
        # bank can never be silently matched against crop features (or with a
        # different crop size / encoder).
        protocol = {
            "crops_per_frame": int(args.crops_per_frame),
            "crop_size": int(args.crop_size),
            "feature_model": args.feature_model,
            "vae_roundtrip": bool(args.vae_roundtrip),
            "height": int(args.height), "width": int(args.width),
            "frame_stride": int(args.frame_stride), "seed": int(args.seed),
        }
        save_fd_stats(args.output, mu.cpu().numpy(), sigma.cpu().numpy(), n_total,
                      features=None if ref_features is None else ref_features.numpy(),
                      mmd_bandwidth=mmd_bandwidth, protocol=protocol)
        print(f"[save] {args.output}  protocol={protocol}")
    if world > 1:
        import torch.distributed as dist
        dist.barrier()


if __name__ == "__main__":
    main()
