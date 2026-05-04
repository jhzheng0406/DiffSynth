"""
Full Helios implementation for DiffSynth-Studio WanModel.

Three key contributions ported from the Helios paper (arXiv:2603.04379):

1. Unified History Injection (UHI)
   - Three separate Conv3d (patch_helios_{short,mid,long}) with different
     kernel/stride sizes per tier, initialised from the existing patch_embedding.
   - History tokens are PREPENDED to the current token sequence and pass
     through transformer blocks; text cross-attention can be restricted to
     current tokens to match Helios guidance_cross_attn.
   - 3D RoPE is applied to history tokens using their ACTUAL absolute temporal
     frame indices, and spatially-strided freq entries for compressed tiers.
   - After the block loop, history tokens are stripped; only current tokens
     reach the prediction head.

2. Easy Anti-Drifting (EAD)  [training time]
   - corrupt_history_latents():  Gaussian noise or downsample/upsample
     corruption applied independently per temporal-window, simulating the
     appearance drift that occurs at inference.
   - add_saturation_to_history_latents():  Scales latents around their
     channel-mean by a random factor, simulating colour/brightness drift.

3. History latent preparation
   - At inference, accumulated chunk latents are concatenated and the last
     sum(history_sizes) frames are sliced into [long, mid, short] tiers.
   - Frame IDs are tracked for correct RoPE.

Usage (inference):
    patch_wan_model_for_helios(pipe.dit)
    # before each chunk k > 0:
    hist = prepare_helios_history(accumulated_latents, history_sizes=[16,2,1])
    pipe.dit.set_helios_history(*hist)
    video = pipe(...)
    pipe.dit.clear_helios_history()

Usage (training):
    patch_wan_model_for_helios(pipe.dit, freeze_history_scale=False)
    lat_s, lat_m, lat_l = split_training_video_into_history_tiers(video_latents)
    lat_s, lat_m, lat_l = corrupt_history_latents(lat_s, lat_m, lat_l)
    lat_s, lat_m, lat_l = add_saturation_to_history_latents(lat_s, lat_m, lat_l)
    fids_s, fids_m, fids_l = build_frame_ids(...)
    pipe.dit.set_helios_history(lat_l, lat_m, lat_s, fids_l, fids_m, fids_s)
    loss = FlowMatchSFTLoss(pipe, ...)
    pipe.dit.clear_helios_history()
"""

import random
import types
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
from einops import rearrange, repeat


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pad_for_conv3d(x: torch.Tensor, stride: Tuple[int, int, int]) -> torch.Tensor:
    """Pad x so T, H, W are each divisible by the corresponding stride."""
    sT, sH, sW = stride
    _, _, T, H, W = x.shape
    pT = (sT - T % sT) % sT
    pH = (sH - H % sH) % sH
    pW = (sW - W % sW) % sW
    if pT > 0 or pH > 0 or pW > 0:
        x = F.pad(x, (0, pW, 0, pH, 0, pT), mode="replicate")
    return x


def _compute_history_freqs(
    dit,
    f_out: int, h_out: int, w_out: int,
    frame_ids: torch.Tensor,   # [f_out] int — absolute latent frame positions
    spatial_stride: int = 1,   # 1 for short, 2 for mid, 4 for long
) -> torch.Tensor:
    """
    Build 3D RoPE freqs for history tokens at a given tier.

    spatial_stride reflects how many current-token spatial positions one
    history token covers:
      short tier → 1  (same resolution as current)
      mid   tier → 2  (2× downsampled)
      long  tier → 4  (4× downsampled)

    Returns freqs of shape [f_out * h_out * w_out, 1, head_dim_half].
    """
    freqs_t = dit.freqs[0]
    freqs_h = dit.freqs[1]
    freqs_w = dit.freqs[2]

    # Temporal: use actual frame positions
    f_ids = frame_ids.long().clamp(0, len(freqs_t) - 1).to(freqs_t.device)
    temp = freqs_t[f_ids].view(f_out, 1, 1, -1).expand(f_out, h_out, w_out, -1)

    # Spatial: every `spatial_stride`-th position
    h_idx = torch.arange(0, h_out * spatial_stride, spatial_stride, device=freqs_h.device)
    w_idx = torch.arange(0, w_out * spatial_stride, spatial_stride, device=freqs_w.device)
    h_idx = h_idx[:h_out].clamp(max=len(freqs_h) - 1)
    w_idx = w_idx[:w_out].clamp(max=len(freqs_w) - 1)

    hf = freqs_h[h_idx].view(1, h_out, 1, -1).expand(f_out, h_out, w_out, -1)
    wf = freqs_w[w_idx].view(1, 1, w_out, -1).expand(f_out, h_out, w_out, -1)

    freqs = torch.cat([temp, hf, wf], dim=-1)          # [f,h,w, head_dim_half]
    return freqs.reshape(f_out * h_out * w_out, 1, -1)


def _build_helios_token_sequence(
    dit,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Build Helios history token sequence and corresponding RoPE freqs for
    injection into model_fn_wan_video (which never calls dit.forward).

    Returns:
        h_tokens : [B, N_hist, dim] concatenated history tokens, or None
        h_freqs  : [N_hist, 1, head_dim_half] RoPE freqs,          or None
    """
    if not (hasattr(dit, '_helios_history') and dit._helios_history is not None):
        return None, None

    lat_long, lat_mid, lat_short = dit._helios_history
    fids_long, fids_mid, fids_short = dit._helios_frame_ids
    h_tokens: list = []
    h_freqs:  list = []

    # long tier — patch_helios_long, stride (4,8,8)
    if lat_long is not None and lat_long.shape[2] > 0:
        lat = _pad_for_conv3d(lat_long.to(dtype=dtype, device=device), (4, 8, 8))
        tok = dit.patch_helios_long(lat)          # [B, dim, fl, hl, wl]
        _, _, fl, hl, wl = tok.shape
        tok = tok.flatten(2).transpose(1, 2)      # [B, fl*hl*wl, dim]
        t_ids = fids_long[::4][:fl].to(device)
        fr = _compute_history_freqs(dit, fl, hl, wl, t_ids, spatial_stride=4).to(device)
        h_tokens.append(tok)
        h_freqs.append(fr)

    # mid tier — patch_helios_mid, stride (2,4,4)
    if lat_mid is not None and lat_mid.shape[2] > 0:
        lat = _pad_for_conv3d(lat_mid.to(dtype=dtype, device=device), (2, 4, 4))
        tok = dit.patch_helios_mid(lat)
        _, _, fm, hm, wm = tok.shape
        tok = tok.flatten(2).transpose(1, 2)
        t_ids = fids_mid[::2][:fm].to(device)
        fr = _compute_history_freqs(dit, fm, hm, wm, t_ids, spatial_stride=2).to(device)
        h_tokens.append(tok)
        h_freqs.append(fr)

    # short tier — patch_helios_short, stride (1,2,2)
    if lat_short is not None and lat_short.shape[2] > 0:
        lat = lat_short.to(dtype=dtype, device=device)
        tok = dit.patch_helios_short(lat)
        _, _, fs, hs, ws = tok.shape
        tok = tok.flatten(2).transpose(1, 2)
        t_ids = fids_short[:fs].to(device)
        fr = _compute_history_freqs(dit, fs, hs, ws, t_ids, spatial_stride=1).to(device)
        h_tokens.append(tok)
        h_freqs.append(fr)

    if not h_tokens:
        return None, None
    return torch.cat(h_tokens, dim=1), torch.cat(h_freqs, dim=0)


class HeliosSelfAttention(nn.Module):
    """
    Wan self-attention patched with Helios history-key amplification.

    The original Wan attention weights are reused. We only add a learnable
    per-head `history_key_scale`, which scales the prepended history keys.
    """

    def __init__(
        self,
        base_attn: nn.Module,
        parent_dit: nn.Module,
        is_amplify_history: bool = True,
        init_scale_logit: float = -3.0,
        max_history_scale: float = 10.0,
        freeze_history_scale: bool = True,
    ):
        super().__init__()
        self.dim = base_attn.dim
        self.num_heads = base_attn.num_heads
        self.head_dim = base_attn.head_dim

        self.q = base_attn.q
        self.k = base_attn.k
        self.v = base_attn.v
        self.o = base_attn.o
        self.norm_q = base_attn.norm_q
        self.norm_k = base_attn.norm_k
        self.attn = base_attn.attn

        # Use object.__setattr__ to avoid registering parent_dit as a submodule
        # (which would create a circular reference: dit → block → _helios_parent → dit)
        object.__setattr__(self, '_helios_parent', parent_dit)
        self.is_amplify_history = is_amplify_history
        self.max_history_scale = max_history_scale
        self.register_buffer("_scale_cache", None)

        if is_amplify_history:
            # float32 regardless of model dtype: bfloat16 ULP at init_scale_logit≈-3
            # is ~0.016, far above the 1e-5 AdamW update, so the parameter would
            # never move if stored in bfloat16.
            self.history_key_scale = nn.Parameter(
                torch.full(
                    (self.num_heads,),
                    init_scale_logit,
                    device=self.q.weight.device,
                    dtype=torch.float32,
                )
            )
            self.history_key_scale.requires_grad_(not freeze_history_scale)
        else:
            self.register_parameter("history_key_scale", None)

    def get_scale_key(self) -> Optional[torch.Tensor]:
        if self.history_key_scale is None:
            return None
        if self.history_key_scale.requires_grad:
            return 1.0 + torch.sigmoid(self.history_key_scale) * (self.max_history_scale - 1.0)
        if self._scale_cache is None:
            self._scale_cache = 1.0 + torch.sigmoid(self.history_key_scale) * (self.max_history_scale - 1.0)
        return self._scale_cache

    def forward(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        from .wan_video_dit import rope_apply

        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)

        history_len = getattr(self._helios_parent, "_helios_current_history_len", 0)
        if history_len > 0 and self.is_amplify_history:
            scale_key = self.get_scale_key()
            if scale_key is not None:
                s = scale_key.to(device=k.device, dtype=k.dtype)   # [num_heads]
                k_hist = rearrange(k[:, :history_len], "b s (n d) -> b s n d", n=self.num_heads)
                k_hist = k_hist * s.view(1, 1, -1, 1)              # out-of-place: new tensor
                k = torch.cat([
                    rearrange(k_hist, "b s n d -> b s (n d)"),
                    k[:, history_len:],
                ], dim=1)

        x = self.attn(q, k, v)
        return self.o(x)


# ---------------------------------------------------------------------------
# Patching
# ---------------------------------------------------------------------------

def patch_wan_model_for_helios(
    dit,
    history_sizes: List[int] = (16, 2, 1),   # [long, mid, short] latent frames
    is_amplify_history: bool = True,
    init_scale_logit: float = -3.0,
    max_history_scale: float = 10.0,
    freeze_history_scale: bool = True,
    use_first_frame_anchor: bool = True,
) -> None:
    """
    Mutates a WanModel `dit` in-place to support Helios history injection.

    Steps:
      1. Adds patch_helios_{short,mid,long} Conv3d modules with weights
         initialised by tile-repeating patch_embedding's weights.
      2. Adds an optional per-head learnable history_key_scale parameter.
      3. Replaces dit.forward() with a history-aware version that prepends
         history tokens (with correct 3D RoPE) and strips them after blocks.
      4. Adds set_helios_history() / clear_helios_history() helpers.
    """
    from .wan_video_dit import sinusoidal_embedding_1d
    from ..core.gradient import gradient_checkpoint_forward

    if getattr(dit, "_helios_patch_applied", False):
        print("[Helios] Patch already applied, skip.")
        return

    dim = dit.dim
    # History latents are raw VAE latents regardless of model type.
    # dit.in_dim may be larger (e.g. 48 for Fun-Control = noise+control+image),
    # but the VAE latent channel count is out_dim (typically 16).
    # WanModel doesn't store out_dim directly; derive it from the head linear layer:
    #   head.head: Linear(dim, out_dim * prod(patch_size))
    lat_dim = dit.head.head.out_features // math.prod(dit.patch_size)

    # ── 1. Add tier-specific patch Conv3d modules ─────────────────────
    dit.patch_helios_short = nn.Conv3d(lat_dim, dim, kernel_size=(1, 2, 2), stride=(1, 2, 2))
    dit.patch_helios_mid   = nn.Conv3d(lat_dim, dim, kernel_size=(2, 4, 4), stride=(2, 4, 4))
    dit.patch_helios_long  = nn.Conv3d(lat_dim, dim, kernel_size=(4, 8, 8), stride=(4, 8, 8))

    # Move to same device/dtype as patch_embedding
    ref = dit.patch_embedding
    for name in ('patch_helios_short', 'patch_helios_mid', 'patch_helios_long'):
        getattr(dit, name).to(device=ref.weight.device, dtype=ref.weight.dtype)

    # Initialise from the first lat_dim channels of patch_embedding.
    # patch_embedding weight shape: [dim, dit.in_dim, 1, 2, 2].
    # We use only the first lat_dim input channels (noise channels) as
    # the initialization seed; this gives a reasonable starting point
    # since those channels process the same VAE latent domain.
    with torch.no_grad():
        w_full = dit.patch_embedding.weight.detach()   # [dim, dit.in_dim, 1, 2, 2]
        w = w_full[:, :lat_dim]                        # [dim, lat_dim, 1, 2, 2]
        b = dit.patch_embedding.bias.detach()          # [dim]

        dit.patch_helios_short.weight.copy_(w)
        dit.patch_helios_short.bias.copy_(b)

        # Tile kernel spatially/temporally then rescale to preserve activation scale
        dit.patch_helios_mid.weight.copy_(
            repeat(w, 'o c t h w -> o c (t tk) (h hk) (w wk)', tk=2, hk=2, wk=2) / 8.0
        )
        dit.patch_helios_mid.bias.copy_(b)

        dit.patch_helios_long.weight.copy_(
            repeat(w, 'o c t h w -> o c (t tk) (h hk) (w wk)', tk=4, hk=4, wk=4) / 64.0
        )
        dit.patch_helios_long.bias.copy_(b)

    # ── 2. Replace Wan self-attention with Helios-aware attention ─────
    for block in dit.blocks:
        block.self_attn = HeliosSelfAttention(
            block.self_attn,
            parent_dit=dit,
            is_amplify_history=is_amplify_history,
            init_scale_logit=init_scale_logit,
            max_history_scale=max_history_scale,
            freeze_history_scale=freeze_history_scale,
        )

    # ── 3. Side-channel attributes ────────────────────────────────────
    # Set by set_helios_history(), read by patched forward
    dit._helios_history      = None   # tuple (lat_long, lat_mid, lat_short) or None
    dit._helios_frame_ids    = None   # tuple (fids_long, fids_mid, fids_short)
    dit._helios_history_sizes = list(history_sizes)
    dit._helios_use_first_frame_anchor = bool(use_first_frame_anchor)
    # Number of RoPE positions consumed by history layout (== current chunk's
    # temporal RoPE offset). Anchor adds one extra position at index 0.
    dit._helios_total_history_positions = sum(history_sizes) + (1 if use_first_frame_anchor else 0)
    dit._helios_current_history_len = 0

    # ── 4. Patch forward ──────────────────────────────────────────────
    def _helios_forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        clip_feature=None,
        y=None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        **kwargs,
    ):
        # ── standard preamble (copied from WanModel.forward) ──────────
        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep).to(x.dtype))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        context = self.text_embedding(context)

        if self.has_image_input:
            x = torch.cat([x, y], dim=1)
            clip_embedding = self.img_emb(clip_feature)
            context = torch.cat([clip_embedding, context], dim=1)

        x, (f, h, w) = self.patchify(x)   # [B, f*h*w, dim]

        # Relative RoPE: when history is present, the current chunk's temporal
        # positions must land AFTER all history positions [0..total_hist_pos-1],
        # otherwise current tokens at temporal index k share a RoPE freq with
        # history tokens at the same index, breaking past/future separation.
        # When no history is set (e.g. chunk 0 in inference), fall back to [0..f-1].
        if self._helios_history is not None:
            t_offset = getattr(self, '_helios_total_history_positions', sum(self._helios_history_sizes))
        else:
            t_offset = 0

        # Freqs for current tokens
        freqs = torch.cat([
            self.freqs[0][t_offset : t_offset + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

        # ── Prepend history tokens ─────────────────────────────────────
        history_len = 0
        if self._helios_history is not None:
            lat_long, lat_mid, lat_short = self._helios_history
            fids_long, fids_mid, fids_short = self._helios_frame_ids

            h_tokens = []
            h_freqs  = []

            # long tier — patch_helios_long, stride (4,8,8)
            if lat_long is not None and lat_long.shape[2] > 0:
                lat = _pad_for_conv3d(
                    lat_long.to(dtype=x.dtype, device=x.device), (4, 8, 8))
                tok = self.patch_helios_long(lat)         # [B, dim, fl, hl, wl]
                _, _, fl, hl, wl = tok.shape
                tok = tok.flatten(2).transpose(1, 2)      # [B, fl*hl*wl, dim]
                # Temporal frame IDs: first of each stride-4 window
                t_ids = fids_long[::4][:fl].to(x.device)
                fr = _compute_history_freqs(self, fl, hl, wl, t_ids,
                                            spatial_stride=4).to(x.device)
                h_tokens.append(tok)
                h_freqs.append(fr)

            # mid tier — patch_helios_mid, stride (2,4,4)
            if lat_mid is not None and lat_mid.shape[2] > 0:
                lat = _pad_for_conv3d(
                    lat_mid.to(dtype=x.dtype, device=x.device), (2, 4, 4))
                tok = self.patch_helios_mid(lat)          # [B, dim, fm, hm, wm]
                _, _, fm, hm, wm = tok.shape
                tok = tok.flatten(2).transpose(1, 2)
                t_ids = fids_mid[::2][:fm].to(x.device)
                fr = _compute_history_freqs(self, fm, hm, wm, t_ids,
                                            spatial_stride=2).to(x.device)
                h_tokens.append(tok)
                h_freqs.append(fr)

            # short tier — patch_helios_short, stride (1,2,2)
            if lat_short is not None and lat_short.shape[2] > 0:
                lat = lat_short.to(dtype=x.dtype, device=x.device)
                tok = self.patch_helios_short(lat)        # [B, dim, fs, hs, ws]
                _, _, fs, hs, ws = tok.shape
                tok = tok.flatten(2).transpose(1, 2)
                t_ids = fids_short[:fs].to(x.device)
                fr = _compute_history_freqs(self, fs, hs, ws, t_ids,
                                            spatial_stride=1).to(x.device)
                h_tokens.append(tok)
                h_freqs.append(fr)

            if h_tokens:
                history_len = sum(t.shape[1] for t in h_tokens)
                x     = torch.cat(h_tokens + [x],     dim=1)   # prepend
                freqs = torch.cat(h_freqs  + [freqs], dim=0)

        # ── Apply per-head history key scale inside self-attn ─────────
        # We inject it by hooking into the AttentionModule at block level.
        # Simpler: we mark the history length on self so blocks can read it.
        # (History tokens appear at positions [0:history_len] of x.)
        self._helios_current_history_len = history_len

        # ── Transformer blocks ────────────────────────────────────────
        for block in self.blocks:
            if self.training:
                x = gradient_checkpoint_forward(
                    block, use_gradient_checkpointing,
                    use_gradient_checkpointing_offload,
                    x, context, t_mod, freqs,
                )
            else:
                x = block(x, context, t_mod, freqs)

        # ── Strip history tokens ──────────────────────────────────────
        if history_len > 0:
            x = x[:, history_len:]

        x = self.head(x, t)
        x = self.unpatchify(x, (f, h, w))
        return x

    dit.forward = types.MethodType(_helios_forward, dit)

    # ── 5. Helpers ────────────────────────────────────────────────────
    def set_helios_history(self_dit, *history) -> None:
        """Store history latents in the side-channel."""
        if len(history) == 1 and isinstance(history[0], (tuple, list)):
            history = tuple(history[0])

        if len(history) == 3:
            lat_long, lat_mid, lat_short = history
            fids_long = torch.arange(lat_long.shape[2], dtype=torch.long) if lat_long is not None else None
            fids_mid = torch.arange(lat_mid.shape[2], dtype=torch.long) if lat_mid is not None else None
            fids_short = torch.arange(lat_short.shape[2], dtype=torch.long) if lat_short is not None else None
        elif len(history) == 6:
            lat_long, lat_mid, lat_short, fids_long, fids_mid, fids_short = history
        else:
            raise ValueError(
                "set_helios_history expects either "
                "(lat_long, lat_mid, lat_short), "
                "(lat_long, lat_mid, lat_short, fids_long, fids_mid, fids_short), "
                "or a single tuple/list containing one of those forms."
            )

        self_dit._helios_history   = (lat_long, lat_mid, lat_short)
        self_dit._helios_frame_ids = (
            fids_long  if fids_long  is not None else torch.zeros(0, dtype=torch.long),
            fids_mid   if fids_mid   is not None else torch.zeros(0, dtype=torch.long),
            fids_short if fids_short is not None else torch.zeros(0, dtype=torch.long),
        )

    def clear_helios_history(self_dit) -> None:
        self_dit._helios_history   = None
        self_dit._helios_frame_ids = None
        # NOTE: _helios_current_history_len is intentionally NOT reset here.
        # Gradient checkpointing recomputes block forwards AFTER clear_helios_history()
        # is called (in the loss finally-block). HeliosAttention reads this value
        # to decide whether to apply history key-scaling. If we zero it here, the
        # recomputed path diverges from the original forward (different tensor count
        # → CheckpointError). The value will be overwritten at the start of the
        # next forward pass by model_fn_wan_video.

    dit.set_helios_history   = types.MethodType(set_helios_history,   dit)
    dit.clear_helios_history = types.MethodType(clear_helios_history, dit)
    dit._helios_patch_applied = True

    print(f"[Helios] Patch applied: patch_helios_{{short,mid,long}} added, "
          f"history_key_scale={'trainable' if not freeze_history_scale else 'frozen'} in each block, "
          f"history_sizes={list(history_sizes)}")


# ---------------------------------------------------------------------------
# Inference: build history from accumulated chunk latents
# ---------------------------------------------------------------------------

def prepare_helios_history(
    accumulated_latents: List[torch.Tensor],  # each [1, C, F_lat, H_lat, W_lat]
    history_sizes: List[int] = (16, 2, 1),    # [long, mid, short] frames
    pool_factors: Optional[List[int]] = None, # ignored (pooling done by patch convolutions)
    noise_aug_strength: float = 0.0,          # optional noise augmentation on history latents
    frame_offset: int = 0,                    # global latent frame index of first history frame
    device: str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    use_first_frame_anchor: bool = True,
    first_frame_latent: Optional[torch.Tensor] = None,
) -> Tuple[
    Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor],
    torch.Tensor, torch.Tensor, torch.Tensor,
]:
    """
    Slice the accumulated chunk latents into Helios history tiers.

    With use_first_frame_anchor=True (Helios upstream behavior):
      - Position 0 holds the very first VAE latent of the video (anchor).
      - Positions [1..sum(sizes)] are the sliding window of recent history.
      - lat_short returns 2 frames: [anchor, most_recent_short] with
        frame ids [0, 1+sizes[0]+sizes[1]] (non-contiguous).
      - This anchor never falls out of context, providing a stable color
        reference that combats colour drift across long generations.

    With use_first_frame_anchor=False (legacy port behavior):
      - All sum(sizes) slots are the most-recent sliding window.
      - lat_short returns 1 frame with id sizes[0]+sizes[1].

    Returns:
        lat_long, lat_mid, lat_short   — raw latent tensors (on `device`)
        fids_long, fids_mid, fids_short — absolute latent frame indices
    """
    sizes = list(history_sizes)     # [long, mid, short]
    total = sum(sizes)

    # Concatenate all chunks (preserve their original device — usually CPU
    # in the existing inference loop, but GPU when chunk-0 zero-priming is used).
    all_lat = torch.cat([l.to(dtype=dtype) for l in accumulated_latents], dim=2)
    F_total = all_lat.shape[2]

    # Keep last `total` frames (pad with zeros if not enough history yet)
    if F_total >= total:
        hist = all_lat[:, :, F_total - total:]
    else:
        pad = torch.zeros(
            all_lat.shape[0], all_lat.shape[1], total - F_total,
            all_lat.shape[3], all_lat.shape[4],
            dtype=dtype, device=all_lat.device)
        hist = torch.cat([pad, all_lat], dim=2)

    # Split into tiers: [long (oldest) | mid | short (newest)]
    lat_long, lat_mid, lat_short = hist.split(sizes, dim=2)

    if use_first_frame_anchor:
        if first_frame_latent is None:
            anchor = accumulated_latents[0][:, :, :1].to(dtype=dtype)
        else:
            anchor = first_frame_latent.to(dtype=dtype)
        # Frame ids match upstream layout
        # [prefix=0 | long=1..sizes[0] | mid=...+sizes[1] | recent=...+1]
        fids_long  = torch.arange(1, 1 + sizes[0], dtype=torch.long)
        fids_mid   = torch.arange(1 + sizes[0], 1 + sizes[0] + sizes[1], dtype=torch.long)
        fids_short = torch.tensor([0, 1 + sizes[0] + sizes[1]], dtype=torch.long)
        # short tier carries [anchor, recent] — patch_helios_short stride is (1,2,2)
        # so they remain two distinct tokens with separate RoPE positions.
        lat_short = torch.cat([anchor.to(lat_short.device), lat_short], dim=2)
    else:
        fids_long  = torch.arange(0, sizes[0], dtype=torch.long)
        fids_mid   = torch.arange(sizes[0], sizes[0] + sizes[1], dtype=torch.long)
        fids_short = torch.arange(sizes[0] + sizes[1], total, dtype=torch.long)

    lat_long  = lat_long.to(device)
    lat_mid   = lat_mid.to(device)
    lat_short = lat_short.to(device)

    if noise_aug_strength > 0:
        lat_long  = lat_long  + noise_aug_strength * torch.randn_like(lat_long)
        lat_mid   = lat_mid   + noise_aug_strength * torch.randn_like(lat_mid)
        lat_short = lat_short + noise_aug_strength * torch.randn_like(lat_short)

    return (lat_long, lat_mid, lat_short, fids_long, fids_mid, fids_short)


# ---------------------------------------------------------------------------
# EAD: Easy Anti-Drifting corruption (training time)
# ---------------------------------------------------------------------------

def corrupt_history_latents(
    lat_short: torch.Tensor,   # [B, C, F_short, H, W]
    lat_mid:   torch.Tensor,   # [B, C, F_mid,   H, W]
    lat_long:  torch.Tensor,   # [B, C, F_long,  H, W]
    noise_ratio_short: float = 0.10,
    noise_ratio_mid:   float = 0.15,
    noise_ratio_long:  float = 0.20,
    corrupt_mode: str = "noise",        # "noise" | "downsample" | "random"
    is_keep_x0: bool = True,            # keep first (oldest) frame clean
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    EAD: Corrupt history latents to simulate the drift that occurs at
    inference time (model generates imperfect frames that accumulate error).

    Two modes:
      "noise"      — convex blend with random σ ∈ [0, ratio]: (1-σ)·x + σ·ε
      "downsample" — downsample then upsample to create compression artifacts
      "random"     — randomly choose one of the above per sample
    """
    def _noise_corrupt(x, ratio):
        # Flow-matching convex blend (matches Helios upstream); additive form
        # would inflate latent variance and bias the model.
        sigma = torch.rand(1, device=x.device).item() * ratio
        return (1.0 - sigma) * x + sigma * torch.randn_like(x)

    def _downsample_corrupt(x, min_ratio=0.85, max_ratio=1.0):
        B, C, T, H, W = x.shape
        # Random spatial scale factor close to 1.0 (subtle artifact)
        scale = random.uniform(min_ratio, max_ratio)
        h1 = max(1, int(H * scale))
        w1 = max(1, int(W * scale))
        x_flat = x.reshape(B * T, C, H, W)
        x_small = F.interpolate(x_flat, size=(h1, w1), mode='bilinear', align_corners=False)
        x_back  = F.interpolate(x_small, size=(H, W),  mode='bilinear', align_corners=False)
        return x_back.reshape(B, C, T, H, W)

    def _corrupt(x, ratio, mode):
        if mode == "random":
            mode = random.choice(["noise", "downsample"])
        if mode == "noise":
            return _noise_corrupt(x, ratio)
        else:
            return _downsample_corrupt(x)

    if corrupt_mode == "random":
        mode_short = random.choice(["noise", "downsample"])
        mode_mid   = random.choice(["noise", "downsample"])
        mode_long  = random.choice(["noise", "downsample"])
    else:
        mode_short = mode_mid = mode_long = corrupt_mode

    # Optionally keep the very first (oldest) long frame clean as an anchor
    if is_keep_x0 and lat_long.shape[2] > 1:
        x0         = lat_long[:, :, :1]
        rest_long  = _corrupt(lat_long[:, :, 1:], noise_ratio_long, mode_long)
        lat_long   = torch.cat([x0, rest_long], dim=2)
    else:
        lat_long   = _corrupt(lat_long, noise_ratio_long, mode_long)

    lat_mid   = _corrupt(lat_mid,   noise_ratio_mid,   mode_mid)
    lat_short = _corrupt(lat_short, noise_ratio_short, mode_short)

    return lat_short, lat_mid, lat_long


def add_saturation_to_history_latents(
    lat_short: torch.Tensor,
    lat_mid:   torch.Tensor,
    lat_long:  torch.Tensor,
    sat_min: float = 0.7,
    sat_max: float = 2.0,
    clean_prob: float = 0.2,     # probability of leaving a window unchanged
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    EAD: Simulate colour/brightness drift by randomly scaling latent values
    around their channel mean (per temporal window).

    sat_factor ∈ [sat_min, 1) for desaturation, (1, sat_max] for over-saturation.
    With probability `clean_prob` a window is left unchanged.
    """
    def _saturate_window(x):
        if random.random() < clean_prob:
            return x
        if random.random() < 0.5:
            factor = random.uniform(sat_min, 1.0 - 1e-3)
        else:
            factor = random.uniform(1.0 + 1e-3, sat_max)
        mean = x.mean(dim=1, keepdim=True)   # channel mean → [B, 1, T, H, W]
        return (x - mean) * factor + mean

    def _apply_per_frame(lat):
        """Apply independently per temporal frame."""
        frames = lat.unbind(dim=2)
        corrupted = [_saturate_window(f.unsqueeze(2)) for f in frames]
        return torch.cat(corrupted, dim=2)

    return (
        _apply_per_frame(lat_short),
        _apply_per_frame(lat_mid),
        _apply_per_frame(lat_long),
    )


# ---------------------------------------------------------------------------
# Training helper: split training clip into history tiers
# ---------------------------------------------------------------------------

def split_into_helios_history_and_target(
    video_latents: torch.Tensor,       # [B, C, F_total_lat, H, W]
    history_sizes: List[int] = (16, 2, 1),
    latent_window_size: int = 9,       # target chunk size in latent frames
    is_random_drop: bool = True,
    drop_t2v_ratio: float = 0.10,      # prob of zeroing all history (T2V sim.)
    drop_i2v_ratio: float = 0.05,      # prob of keeping only x0  (I2V sim.)
    use_first_frame_anchor: bool = True,
) -> Tuple[
    torch.Tensor, torch.Tensor,        # target_latents, history_latents (concat)
    torch.Tensor, torch.Tensor, torch.Tensor,   # fids_short, fids_mid, fids_long
]:
    """
    For training: split a video clip into a history portion and a target
    (denoising) portion, matching the inference setup.

    With use_first_frame_anchor=True (matches Helios upstream + new prepare_helios_history):
        clip layout: [anchor(1) | long(sizes[0]) | mid(sizes[1]) | recent(sizes[2]=1) | target(window)]
        anchor      = video_latents[:, :, :1]
        lat_short   = cat([anchor, recent]) — 2 frames at positions [0, 1+sizes[0]+sizes[1]]
        target      = video_latents[:, :, 1 + total_hist : 1 + total_hist + window]
        Required clip length: 1 + total_hist + window.

    With use_first_frame_anchor=False (legacy port behavior):
        clip layout: [long | mid | short | target]
        Required clip length: total_hist + window.

    Returns history tensors and frame IDs ready to pass to
    set_helios_history() after optional EAD corruption.
    """
    sizes  = list(history_sizes)
    total_hist = sum(sizes)
    F = video_latents.shape[2]

    if use_first_frame_anchor:
        required = 1 + total_hist + latent_window_size
        assert F >= required, (
            f"Video too short for first-frame-anchor mode: need {required} latent frames, got {F}")

        anchor      = video_latents[:, :, :1].clone()
        history_lat = video_latents[:, :, 1 : 1 + total_hist]
        target_lat  = video_latents[:, :, 1 + total_hist : 1 + total_hist + latent_window_size]

        lat_long, lat_mid, lat_short_recent = history_lat.split(sizes, dim=2)
        lat_long  = lat_long.clone()
        lat_mid   = lat_mid.clone()
        lat_short = torch.cat([anchor, lat_short_recent], dim=2)  # [B, C, 2, H, W]

        fids_long   = torch.arange(1, 1 + sizes[0], dtype=torch.long)
        fids_mid    = torch.arange(1 + sizes[0], 1 + sizes[0] + sizes[1], dtype=torch.long)
        fids_short  = torch.tensor([0, 1 + sizes[0] + sizes[1]], dtype=torch.long)
        fids_target = torch.arange(1 + total_hist, 1 + total_hist + latent_window_size, dtype=torch.long)

        if is_random_drop:
            if drop_t2v_ratio > 0 and torch.rand(1).item() < drop_t2v_ratio:
                # T2V: zero all history INCLUDING anchor (matches upstream random_drop_t2v branch)
                lat_long  = torch.zeros_like(lat_long)
                lat_mid   = torch.zeros_like(lat_mid)
                lat_short = torch.zeros_like(lat_short)
            elif drop_i2v_ratio > 0 and torch.rand(1).item() < drop_i2v_ratio:
                # I2V: keep only the anchor; zero everything else
                lat_long  = torch.zeros_like(lat_long)
                lat_mid   = torch.zeros_like(lat_mid)
                lat_short[:, :, 1:] = 0  # zero recent, keep anchor at index 0
    else:
        required = total_hist + latent_window_size
        assert F >= required, (
            f"Video too short: need {required} latent frames, got {F}")

        history_lat = video_latents[:, :, :total_hist]
        target_lat  = video_latents[:, :, total_hist : total_hist + latent_window_size]

        lat_long, lat_mid, lat_short = history_lat.split(sizes, dim=2)
        lat_long  = lat_long.clone()
        lat_mid   = lat_mid.clone()
        lat_short = lat_short.clone()

        fids_long   = torch.arange(0, sizes[0], dtype=torch.long)
        fids_mid    = torch.arange(sizes[0], sizes[0] + sizes[1], dtype=torch.long)
        fids_short  = torch.arange(sizes[0] + sizes[1], total_hist, dtype=torch.long)
        fids_target = torch.arange(total_hist, total_hist + latent_window_size, dtype=torch.long)

        if is_random_drop:
            if drop_t2v_ratio > 0 and torch.rand(1).item() < drop_t2v_ratio:
                lat_long  = torch.zeros_like(lat_long)
                lat_mid   = torch.zeros_like(lat_mid)
                lat_short = torch.zeros_like(lat_short)
            elif drop_i2v_ratio > 0 and torch.rand(1).item() < drop_i2v_ratio:
                lat_long[:, :, 1:] = 0
                lat_mid[:]  = 0
                lat_short[:] = 0

    return (
        target_lat,
        lat_long, lat_mid, lat_short,
        fids_long, fids_mid, fids_short,
        fids_target,
    )
