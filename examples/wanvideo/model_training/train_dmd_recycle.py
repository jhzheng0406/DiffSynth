"""
DMD2 + ONE-FORCING (cls_branch GAN) + SVI-STYLE SELF-CORRECTING ERROR RECYCLE.

Keeps DMD + cls_branch GAN intact, adds SVI's core mechanism:

  Per step:
    1. CLEAN rollout (no grad):
         x_pred_clean = student(noise, ref_clean)         ← no error injected
    2. INJECT error from buffer:
         ref_corrupt = ref_clean + α · sampled_error
    3. CORRUPT rollout (with grad):
         x_pred      = student(noise, ref_corrupt)
    4. Critic + GAN-D updates on x_pred.detach()           (unchanged)
    5. Generator update:
         dmd_g_loss + gan_g_loss + λ_sc · L1(x_pred, x_pred_clean.detach())
                                          ↑↑↑ SELF-CORRECTING LOSS
         student must produce same output regardless of error in recent_ref
    6. COLLECT error (drift direction):
         err = x_pred_clean[last] - target_latents[last]   ← (student - real)
         buffer.push(err)
       Sign matters: this is the direction student's autoregressive recent_ref
       drifts away from real at inference. Injecting this direction during
       training simulates what student will see at inference.

Key difference vs the naive recycle (replay only): the self-correcting L1
trains student to be INVARIANT to recent_ref perturbation. This is the
mechanism that makes SVI work — not just replaying errors, but training
the network to output the same answer under input corruption.

α schedule: 0 during buffer warmup, then linearly ramps to error_alpha.

Reference: SVI (Stable-Video-Infinity, vita-epfl). Adapted from base-model
finetuning to DMD distillation regime by anchoring the self-correcting
target to the clean-input rollout.
"""
import argparse, contextlib, os, random, sys, time
import accelerate
import decord
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.distributed as dist
from PIL import Image
from tqdm import tqdm

# Make sibling files importable regardless of cwd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from diffsynth.pipelines.wan_video import (
    WanVideoPipeline,
    ModelConfig,
    model_fn_wan_video,
)
from diffsynth.core import load_state_dict
from peft import LoraConfig, inject_adapter_in_model
from safetensors.torch import save_file

from dmd_utils import (
    DEFAULT_DENOISING_STEPS,
    NUM_TRAIN_TIMESTEPS,
    timestep_to_sigma,
    add_noise_flow,
    velocity_to_x0,
    sample_generator_timestep,
    sample_critic_timestep,
    compute_critic_loss,
    compute_dmd_gradient,
    compute_dmd_loss,
    cfg_real_x0,
)
from train_chunk_aware import ChunkAwareDataset
from cls_branch import ClsBranch, FeatureCapturer, gan_g_loss, gan_d_loss


# ===========================================================================
# Error replay buffer (FIFO ring buffer of latent residuals)
# ===========================================================================
class ErrorBuffer:
    """Per-rank FIFO buffer of single-latent-frame residuals [C, h, w].

    Default (freq_split=False): a single FIFO — IDENTICAL to the original
    behaviour. With freq_split=True it keeps TWO FIFOs, splitting each residual
    into low-/high-spatial-frequency bands (box low-pass at scale band_k) so the
    injection can weight them separately (α_low for background/stability drift,
    α_high for detail/clarity drift). Diagnostic justified this: residual energy
    is substantial + stable in both bands (k=2: 58/42, k=4: 33/67)."""
    def __init__(self, max_size: int = 500, freq_split: bool = False, band_k: int = 2):
        self.max_size = max_size
        self.freq_split = freq_split
        self.band_k = band_k
        self.buf = []          # flat mode
        self.buf_low = []      # freq mode
        self.buf_high = []

    @staticmethod
    def _split(err, k):
        """err [C,h,w] → (low, high). low = avg_pool(k)+nearest-upsample."""
        xb = err.unsqueeze(0)
        low = F.avg_pool2d(xb, kernel_size=k, stride=k, ceil_mode=True)
        low = F.interpolate(low, size=err.shape[-2:], mode="nearest").squeeze(0)
        return low, err - low

    def push(self, err: torch.Tensor):
        e = err.detach().to(torch.bfloat16).cpu().contiguous()
        if self.freq_split:
            lo, hi = self._split(e, self.band_k)
            self.buf_low.append(lo); self.buf_high.append(hi)
            if len(self.buf_low) > self.max_size:
                self.buf_low.pop(0); self.buf_high.pop(0)
        else:
            self.buf.append(e)
            if len(self.buf) > self.max_size:
                self.buf.pop(0)

    def sample(self, device=None, dtype=None) -> torch.Tensor:
        """Flat mode: a single error [C, h, w]."""
        assert len(self.buf) > 0, "ErrorBuffer is empty"
        idx = torch.randint(0, len(self.buf), (1,)).item()
        out = self.buf[idx]
        if device is not None: out = out.to(device)
        if dtype is not None:  out = out.to(dtype)
        return out

    def sample_bands(self, device=None, dtype=None):
        """Freq mode: (low, high), each sampled INDEPENDENTLY from its own bank
        (two reservoirs → maximal decoupling of the two drift modes)."""
        assert len(self.buf_low) > 0, "ErrorBuffer (freq) is empty"
        il = torch.randint(0, len(self.buf_low), (1,)).item()
        ih = torch.randint(0, len(self.buf_high), (1,)).item()
        lo, hi = self.buf_low[il], self.buf_high[ih]
        if device is not None: lo, hi = lo.to(device), hi.to(device)
        if dtype is not None:  lo, hi = lo.to(dtype), hi.to(dtype)
        return lo, hi

    def state_dict(self):
        """Persist BOTH modes' banks so resume works regardless of freq_split."""
        return {"freq_split": self.freq_split, "band_k": self.band_k,
                "buf": list(self.buf), "buf_low": list(self.buf_low),
                "buf_high": list(self.buf_high)}

    def load_state_dict(self, sd):
        """Tolerant restore. Accepts the new dict format OR a legacy flat list
        (old checkpoints saved error_buffer as just list(buf))."""
        def _trim(lst):
            return [t.detach().to(torch.bfloat16).cpu().contiguous()
                    for t in list(lst)[-self.max_size:]]
        if isinstance(sd, dict):
            self.buf = _trim(sd.get("buf", []))
            self.buf_low = _trim(sd.get("buf_low", []))
            self.buf_high = _trim(sd.get("buf_high", []))
        else:                                   # legacy: a flat list of residuals
            self.buf = _trim(sd)
            self.buf_low, self.buf_high = [], []

    def __len__(self):
        return len(self.buf_low) if self.freq_split else len(self.buf)


# ===========================================================================
# Pipeline construction
# ===========================================================================
# Default base = 1.3B Fun-Control. Override with --base_model_id (e.g.
# "PAI/Wan2.1-Fun-V1.1-14B-Control") for the 14B scale-generality run; the dit
# in_dim differs but y_dim is computed dynamically, so the rest adapts.
DEFAULT_BASE_MODEL_ID = "PAI/Wan2.1-Fun-V1.1-1.3B-Control"
TOKENIZER_CONFIG = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B",
                               origin_file_pattern="google/umt5-xxl/")


def wan_model_configs(model_id=DEFAULT_BASE_MODEL_ID):
    """The 4 component files (DiT/T5/VAE/CLIP) all live under the same Fun-Control
    model_id; only the tokenizer (umt5) is shared/separate."""
    return [
        ModelConfig(model_id=model_id, origin_file_pattern="diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id=model_id, origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(model_id=model_id, origin_file_pattern="Wan2.1_VAE.pth"),
        ModelConfig(model_id=model_id, origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
    ]


# Back-compat alias (some scripts import this name).
WAN_MODEL_CONFIGS = wan_model_configs()


def build_pipe(device, dtype=torch.bfloat16, model_id=DEFAULT_BASE_MODEL_ID):
    return WanVideoPipeline.from_pretrained(
        torch_dtype=dtype,
        device=device,
        model_configs=wan_model_configs(model_id),
        tokenizer_config=TOKENIZER_CONFIG,
    )


def fuse_sink_lora_into_pipe(pipe, sink_lora_path, alpha=1.0):
    """Load the sink LoRA and merge into the DiT. After this the DiT weights
    have moved by alpha * (lora_B @ lora_A). The DiT no longer carries LoRA
    modules — it's just a slightly different base."""
    pipe.load_lora(pipe.dit, lora_config=sink_lora_path, alpha=alpha)
    return pipe


def add_trainable_lora(dit, target_modules, rank=32, init_zero_B=True):
    """Add a fresh trainable LoRA on top of an already-fused DiT.
    With init_zero_B=True (peft default), the LoRA's initial effect is 0 so
    the model behaves exactly like its current (sink-fused) state at step 0."""
    lora_config = LoraConfig(
        r=rank, lora_alpha=rank,
        target_modules=target_modules,
    )
    dit = inject_adapter_in_model(lora_config, dit)
    return dit


def trainable_params(dit):
    return [p for p in dit.parameters() if p.requires_grad]


def all_reduce_grads(params, num_processes):
    """Manually sync gradients across DDP ranks. Cheap for LoRA (~80MB).
    Skips ranks==1 (single-GPU) automatically."""
    if num_processes <= 1:
        return
    for p in params:
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad.div_(num_processes)


class ResumableSampler:
    """Wraps a base sampler and, ONLY on epoch `skip_epoch`, drops the first
    `skip_first` indices — at the index level. The DataLoader never fetches the
    skipped samples, so resume does NOT decode their videos (the slow part).
    Just advancing integers, not 49-frame decords."""
    def __init__(self, base_sampler, skip_first=0, skip_epoch=0):
        self.base = base_sampler
        self.skip_first = skip_first
        self.skip_epoch = skip_epoch
        self._epoch = 0

    def set_epoch(self, epoch):
        self._epoch = epoch
        if hasattr(self.base, "set_epoch"):
            self.base.set_epoch(epoch)

    def __iter__(self):
        it = iter(self.base)
        if self._epoch == self.skip_epoch and self.skip_first > 0:
            for _ in range(self.skip_first):
                next(it, None)        # advance index only — no __getitem__/decode
        return it

    def __len__(self):
        if self._epoch == self.skip_epoch:
            return max(0, len(self.base) - self.skip_first)
        return len(self.base)


def save_lora_state_dict(dit, out_path, remove_prefix=None, adapter=None):
    """Save only the trainable LoRA params as a safetensors file. With `adapter`
    set (shared-base), save ONLY that named adapter's params (e.g. 'student')."""
    state_dict = {}
    for name, param in dit.state_dict().items():
        if "lora_A" in name or "lora_B" in name:
            if adapter is not None and f".{adapter}." not in name:
                continue
            if remove_prefix and name.startswith(remove_prefix):
                name = name[len(remove_prefix):]
            state_dict[name] = param.detach().cpu().contiguous()
    save_file(state_dict, out_path)


# ===========================================================================
# Shared-base: ONE heavy base+sink DiT carrying TWO named LoRA adapters
# (student, critic) so 14B fits — teacher/student/critic no longer each hold a
# full base copy. teacher = adapters disabled; student/critic = their adapter.
# ===========================================================================
SHARED_ADAPTERS = ("student", "critic")


def add_named_lora(dit, target_modules, rank, names=SHARED_ADAPTERS):
    """Inject MULTIPLE named LoRA adapters onto one (sink-fused) DiT."""
    cfg = LoraConfig(r=rank, lora_alpha=rank, target_modules=target_modules)
    for nm in names:
        inject_adapter_in_model(cfg, dit, adapter_name=nm)
    for n, p in dit.named_parameters():        # keep ALL adapters trainable
        if "lora_" in n:
            p.requires_grad_(True)
    return dit


def use_adapter(dit, name):
    """Activate ONE adapter for the next forward(s), or None = base+sink only
    (teacher). Uses peft's public set_adapter to switch, then RE-FORCES
    requires_grad=True on every LoRA param — because set_adapter freezes the
    non-active adapter, which would silently break the still-alive student
    rollout graph when we switch to 'critic' mid-step."""
    from peft.tuners.tuners_utils import BaseTunerLayer
    for m in dit.modules():
        if isinstance(m, BaseTunerLayer):
            if name is None:
                m.enable_adapters(False)
            else:
                m.enable_adapters(True)
                m.set_adapter(name)
    if name is not None:
        for n, p in dit.named_parameters():
            if "lora_" in n:
                p.requires_grad_(True)


def adapter_params(dit, name):
    return [p for n, p in dit.named_parameters() if "lora_" in n and f".{name}." in n]


def fsdp_wrap_blocks(dit):
    """Per-DiTBlock FULL_SHARD (use_orig_params=True for PEFT). Leaves the dit
    root intact → .dim/.in_dim/.blocks and FeatureCapturer hooks keep working.
    All LoRA (q/k/v/o/ffn) lives inside the blocks → sharded + FSDP grad-reduced;
    only cls_branch (outside) still needs manual all_reduce. Validated mechanics:
    notes/analysis/test_fsdp_peft.py."""
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
    dev = torch.cuda.current_device()
    for i in range(len(dit.blocks)):
        dit.blocks[i] = FSDP(dit.blocks[i], use_orig_params=True,
                             sharding_strategy=ShardingStrategy.FULL_SHARD, device_id=dev)
    return dit


def load_lora_into_trainable(dit, lora_state_dict):
    """Load a LoRA state dict into the trainable LoRA modules already injected
    via add_trainable_lora. Tolerant of missing keys (returns load_result)."""
    # peft uses 'lora_A.default.weight' naming, but the saved files may have
    # 'lora_A.weight'. Normalize.
    fixed = {}
    for k, v in lora_state_dict.items():
        if "lora_A.weight" in k:
            k = k.replace("lora_A.weight", "lora_A.default.weight")
        if "lora_B.weight" in k:
            k = k.replace("lora_B.weight", "lora_B.default.weight")
        fixed[k] = v
    return dit.load_state_dict(fixed, strict=False)


# ===========================================================================
# Conditioning encoder (shared across 3 models)
# ===========================================================================
@torch.no_grad()
def encode_prompt(pipe, prompt, dtype, device):
    """Replicates WanVideoUnit_PromptEmbedder.encode_prompt (wan_video.py:442)."""
    pipe.load_models_to_device(["text_encoder"])
    ids, mask = pipe.tokenizer(prompt, return_mask=True, add_special_tokens=True)
    ids = ids.to(device)
    mask = mask.to(device)
    seq_lens = mask.gt(0).sum(dim=1).long()
    prompt_emb = pipe.text_encoder(ids, mask).to(dtype=dtype)
    for i, v in enumerate(seq_lens):
        prompt_emb[:, v:] = 0
    return prompt_emb


LATENT_CACHE_VERSION = "v1"   # bump to invalidate all on-disk latent caches


@torch.no_grad()
def encode_batch(pipe, batch, dtype, device, no_recent=False, plp=False, cache_dir=None, aspect_crop=False, base_model_id="", vel_recent=False):
    """Use a pipe's encoders (T5/VAE/CLIP) to encode one batch from
    ChunkAwareDataset into the tensors model_fn_wan_video expects.

    Returns dict with: prompt_embed, target_latents, control_latents,
                       reference_latents (sink+recent on T-dim), clip_feature
    """
    # ---- latent disk cache: on hit, skip ALL encoders (deterministic under
    # --plp, where the recent ref is a video-mode latent, not a random aug).
    # First epoch populates; later epochs / reruns load tensors. Key folds in
    # everything that changes the cached conditioning: uid (video#pose#chunk),
    # prompt, plp, no_recent, H, W, and a manual LATENT_CACHE_VERSION. Bump the
    # version (or use a fresh --latent_cache dir per experiment) if the base
    # VAE/T5/CLIP, the aug recipe, or the encoding logic changes. ----
    _cache_path = None
    if cache_dir is not None and batch.get("uid") is not None:
        import hashlib
        _w, _h = batch["sink_reference_image"][0].size
        _phash = hashlib.md5(str(batch.get("prompt", "")).encode()).hexdigest()[:8]
        _btag = hashlib.md5(str(base_model_id).encode()).hexdigest()[:6]   # control y_pad depends on dit.in_dim
        _sig = f'{batch["uid"]}|p{_phash}|plp{int(plp)}|nr{int(no_recent)}|ac{int(aspect_crop)}|vr{int(vel_recent)}|b{_btag}|{LATENT_CACHE_VERSION}'
        _key = hashlib.md5(_sig.encode()).hexdigest()[:16]
        _cache_path = os.path.join(cache_dir, f"{_key}_{_w}x{_h}.pt")
        if os.path.isfile(_cache_path):
            d = torch.load(_cache_path, map_location="cpu")
            return {k: (v.to(device=device, dtype=dtype) if torch.is_tensor(v) else v)
                    for k, v in d.items()}

    # ---- T5 text ----
    prompt_emb = encode_prompt(pipe, batch["prompt"], dtype, device)

    # ---- VAE: target video → latent ----
    pipe.vae.to(device)
    target_video = pipe.preprocess_video(batch["video"]).to(dtype=dtype, device=device)
    target_latents = pipe.vae.encode(target_video, device=device).to(dtype=dtype)

    # ---- VAE: control video (pose) → latent ----
    # Wan-Fun-Control expects y to have (dit.in_dim - 2 * latent_channels) channels:
    # y = [control_latent(16) | zero_padding(16)] for V1.1 1.3B (in_dim=48).
    # See WanVideoUnit_FunControl.process (wan_video.py:540).
    control_video = pipe.preprocess_video(batch["control_video"]).to(dtype=dtype, device=device)
    control_latents = pipe.vae.encode(control_video, device=device).to(dtype=dtype)
    y_dim = pipe.dit.in_dim - control_latents.shape[1] - target_latents.shape[1]
    if y_dim > 0:
        y_pad = torch.zeros(
            (control_latents.shape[0], y_dim, control_latents.shape[2],
             control_latents.shape[3], control_latents.shape[4]),
            dtype=dtype, device=device,
        )
        control_latents = torch.cat([control_latents, y_pad], dim=1)   # [B, in_dim - 16, T, h, w]

    # ---- VAE: sink + recent ref images → ref latent ----
    # Build TWO reference latents: one with the augmented recent (for the
    # student, drift-robustness) and one with the clean recent (for the teacher
    # in asymmetric mode — a reliable, in-distribution target).
    sink_pil = batch["sink_reference_image"][0]
    recent_aug_pil = batch["reference_image"][0]                  # augmented
    recent_clean_pil = batch.get("reference_image_clean", [recent_aug_pil])[0]
    w, h = sink_pil.size

    # PLP: recent reference = VIDEO-mode latent of the previous chunk's last
    # frame (matches inference z_prev[:,:,-1]). When set, both aug and clean
    # refs use this same latent (pixel-space recent_aug is moot in latent space;
    # the student's drift perturbation comes from the recycle injection instead).
    # None on chunk 0 / non-plp → falls back to the image-mode single-frame encode.
    plp_recent = None
    if plp and batch.get("recent_window") is not None:
        rw = pipe.preprocess_video(batch["recent_window"]).to(dtype=dtype, device=device)
        # PLP-v2 (--vel_recent): keep the prev chunk's LAST TWO latent frames so
        # the 1-step student gets motion/velocity context, not just position
        # (directly attacks the motion-region blur that survives flat PLP). The
        # two frames land at ref T=1 (recent₋₂) and T=2 (recent₋₁); recycle error
        # is injected into the last one (recent₋₁). Default (-1:) = original PLP.
        _n = 2 if vel_recent else 1
        plp_recent = pipe.vae.encode(rw, device=device).to(dtype=dtype)[:, :, -_n:]

    def _ref_latent(recent_pil):
        sink_video = pipe.preprocess_video([sink_pil.resize((w, h))]).to(dtype=dtype, device=device)
        sink_latent = pipe.vae.encode(sink_video, device=device).to(dtype=dtype)
        if no_recent:
            return sink_latent           # sink-only (1 frame) for sinkonly teacher
        if plp_recent is not None:
            recent_latent = plp_recent
        else:
            recent_video = pipe.preprocess_video([recent_pil.resize((w, h))]).to(dtype=dtype, device=device)
            recent_latent = pipe.vae.encode(recent_video, device=device).to(dtype=dtype)
        # Concat sink at T=0, recent at T=1 (matches our patched FunReference)
        return torch.cat([sink_latent, recent_latent], dim=2)

    reference_latents_aug   = _ref_latent(recent_aug_pil)
    reference_latents_clean = _ref_latent(recent_clean_pil)

    # ---- CLIP: two versions so the loop can route per recent_augment_mode ----
    # no_recent → use sink for clip (matches WanVideoUnit_FunReference fallback).
    # Otherwise: one from aug recent (student side), one from clean (teacher).
    clip_feature_aug = None
    clip_feature_clean = None
    if pipe.image_encoder is not None:
        pipe.image_encoder.to(device)
        def _clip(pil):
            return pipe.image_encoder.encode_image([pipe.preprocess_image(pil)]).to(dtype=dtype)
        if no_recent:
            clip_feature_aug = clip_feature_clean = _clip(sink_pil)
        else:
            clip_feature_aug = _clip(recent_aug_pil)
            clip_feature_clean = (clip_feature_aug if recent_aug_pil is recent_clean_pil
                                  else _clip(recent_clean_pil))

    result = dict(
        prompt_embed=prompt_emb,
        target_latents=target_latents,
        control_latents=control_latents,
        reference_latents_aug=reference_latents_aug,
        reference_latents_clean=reference_latents_clean,
        clip_feature_aug=clip_feature_aug,
        clip_feature_clean=clip_feature_clean,
    )
    # ---- populate cache (atomic write: tmp + rename, multi-rank safe) ----
    if _cache_path is not None and not os.path.isfile(_cache_path):
        os.makedirs(cache_dir, exist_ok=True)
        tmp = f"{_cache_path}.tmp{os.getpid()}"
        torch.save({k: (v.detach().cpu() if torch.is_tensor(v) else v)
                    for k, v in result.items()}, tmp)
        os.replace(tmp, _cache_path)
    return result


# ===========================================================================
# Single DiT forward — wraps model_fn_wan_video to expose velocity output
# ===========================================================================
def dit_forward(dit, x_t, t_int, prompt_embed, control_latents,
                reference_latents, clip_feature, device, dtype,
                use_gradient_checkpointing=False):
    """Run a single dit forward with all our conditioning.

    Args:
        dit:               the WanModel (student / teacher / critic)
        x_t:               noisy target latent [B, 16, T, H, W]
        t_int:             integer timestep (0..1000)
        prompt_embed:      T5 text embedding
        control_latents:   pose latent [B, 16, T, H, W]  → passed as `y`
        reference_latents: sink+recent latent [B, 16, 2, h, w]
        clip_feature:      CLIP image feature
        use_gradient_checkpointing: enable activation checkpointing for the
            transformer blocks. Critical for fitting 3 DiTs in memory.

    Returns the velocity prediction (same shape as x_t).
    """
    # bf16 to match dit dtype (model_fn_wan_video calls dit.time_embedding on
    # sinusoidal_embedding_1d(timestep) without a dtype cast).
    timestep = torch.tensor([t_int], dtype=dtype, device=device)
    velocity = model_fn_wan_video(
        dit=dit,
        latents=x_t.to(dtype=dtype, device=device),
        timestep=timestep,
        context=prompt_embed.to(dtype=dtype, device=device),
        clip_feature=clip_feature,
        y=control_latents.to(dtype=dtype, device=device),
        reference_latents=reference_latents.to(dtype=dtype, device=device),
        use_gradient_checkpointing=use_gradient_checkpointing,
    )
    return velocity


# ===========================================================================
# Self-forcing rollout (Causal Forcing++ Stage-3 style)
# ===========================================================================
def rollout_student(
    dit, denoising_step_list, target_shape,
    prompt_embed, control_latents, reference_latents, clip_feature,
    device, dtype, use_gradient_checkpointing=False,
    initial_noisy=None, exit_idx=None, renoise_noises=None,
):
    """Roll the student through the full few-step denoising schedule, starting
    from PURE NOISE. Random exit step → only one forward keeps grad; everything
    earlier runs under no_grad with re-noising between steps.

    Args (new, for SVI self-correcting loss — shared randomness across two rollouts):
        initial_noisy : pre-generated noise tensor [target_shape]. If None, a new
                        one is sampled. Pass the same tensor to both clean+corrupt
                        rollouts so the only difference is the reference.
        exit_idx      : int in [0, N), the step at which to take the grad forward.
                        If None, sampled uniformly. Pass same value to both rollouts.
        renoise_noises: list of fresh noise tensors used for re-noising between
                        no-grad steps. Length = exit_idx. If None, sampled fresh.
    """
    N = len(denoising_step_list)
    if exit_idx is None:
        exit_idx = random.randrange(N)        # uniform over 0..N-1
    if initial_noisy is None:
        initial_noisy = torch.randn(target_shape, dtype=dtype, device=device)
    noisy = initial_noisy
    renoise_idx = 0

    for idx, t in enumerate(denoising_step_list):
        sigma_t = float(t) / NUM_TRAIN_TIMESTEPS
        if idx == exit_idx:
            # Grad step: one forward with autograd, then we stop the loop.
            v = dit_forward(
                dit, noisy, t, prompt_embed,
                control_latents, reference_latents, clip_feature, device, dtype,
                use_gradient_checkpointing=use_gradient_checkpointing,
            )
            x_pred = velocity_to_x0(v, noisy, sigma_t)
            return x_pred, t
        # No-grad step: forward → x0 → re-noise to NEXT timestep.
        with torch.no_grad():
            v = dit_forward(
                dit, noisy, t, prompt_embed,
                control_latents, reference_latents, clip_feature, device, dtype,
            )
            x0 = velocity_to_x0(v, noisy, sigma_t)
            next_t = denoising_step_list[idx + 1]
            sigma_next = float(next_t) / NUM_TRAIN_TIMESTEPS
            if renoise_noises is not None and renoise_idx < len(renoise_noises):
                fresh = renoise_noises[renoise_idx]
            else:
                fresh = torch.randn_like(x0)
            renoise_idx += 1
            # Re-noise the clean estimate to the next timestep (consistency-
            # sampler style, matches CF++ scheduler.add_noise).
            # NOTE: This is a LINEAR INTERPOLATION re-noise. The actual inference
            # pipeline uses the scheduler's Euler step, not this. For 1-step
            # rollout this branch is dead (exit_idx is always the only step).
            # If/when extending to 2/4-step rollouts, this train-vs-inference
            # mismatch becomes real — consider switching to scheduler.add_noise.
            noisy = sigma_next * fresh + (1 - sigma_next) * x0

    raise RuntimeError("rollout_student: exit step not consumed")


# ===========================================================================
# Novel-2: motion-targeted recycle. A per-pixel spatial weight from the pose-
# control video's frame-to-frame change (the subject-motion signal, free — no
# optical flow). Used to up-weight the recycled error injection in high-motion
# regions, where 1-step blur accumulates. weight = 1 + beta * motion01, shape
# [B,1,h,w] so it broadcasts onto an error tensor [C,h,w] or [B,C,h,w].
# ===========================================================================
def motion_weight_map(control_latents, beta):
    cl = control_latents[:, :16].float()                              # pose channels (drop y_pad)
    if cl.shape[2] < 2:
        return torch.ones(cl.shape[0], 1, cl.shape[-2], cl.shape[-1],
                          device=control_latents.device, dtype=control_latents.dtype)
    m = (cl[:, :, 1:] - cl[:, :, :-1]).abs().mean(dim=(1, 2))         # [B,h,w]
    m01 = m / (m.amax(dim=(1, 2), keepdim=True) + 1e-6)              # per-sample [0,1]
    return (1.0 + beta * m01).unsqueeze(1).to(control_latents.dtype)  # [B,1,h,w]


# ===========================================================================
# Argument parser
# ===========================================================================
def get_parser():
    p = argparse.ArgumentParser()
    # data
    p.add_argument("--dataset_metadata_path", required=True)
    p.add_argument("--height", type=int, default=832)
    p.add_argument("--width", type=int, default=480)
    p.add_argument("--num_frames", type=int, default=49)
    p.add_argument("--dataset_repeat", type=int, default=1)
    p.add_argument("--recent_aug_strength", type=float, default=0.5)
    # model
    p.add_argument("--teacher_lora_path", default="",
                   help="Sink LoRA fused into all 3 models' base. Empty / 'none' "
                        "→ vanilla base teacher (student must learn sink itself; "
                        "without regression it can't — experimental).")
    p.add_argument("--lora_rank", type=int, default=32)
    p.add_argument("--base_model_id", default=DEFAULT_BASE_MODEL_ID,
                   help="Base Fun-Control model_id for teacher/student/critic pipes. "
                        "Default 1.3B; set 'PAI/Wan2.1-Fun-V1.1-14B-Control' for the 14B run.")
    p.add_argument("--shared_base", action="store_true",
                   help="Memory: hold ONE base+sink DiT carrying student+critic LoRA "
                        "adapters (teacher=adapters off) instead of 3 full copies. "
                        "Required to fit 14B on 80GB. Default off = original 3 pipes.")
    p.add_argument("--fsdp", action="store_true",
                   help="Shard the shared-base DiT's transformer blocks across GPUs "
                        "(FSDP FULL_SHARD, per-DiTBlock) → ~1/N weights per GPU, lets "
                        "you drop the GAN CPU offload. 14B-only; requires --shared_base. "
                        "Default off — does NOT touch the 1.3B / non-fsdp path.")
    p.add_argument("--lora_target_modules", default="q,k,v,o,ffn.0,ffn.2")
    p.add_argument("--no_recent", action="store_true",
                   help="Sink-only conditioning (no recent frame). Use when the "
                        "teacher is a sinkonly model trained without recent.")
    # recent-frame augmentation routing (sink/drift experiments)
    p.add_argument("--recent_augment_mode", choices=["off", "symmetric", "asymmetric"],
                   default="symmetric",
                   help="off: clean recent everywhere. symmetric: augmented recent "
                        "for all (needs augment-trained teacher). asymmetric: clean "
                        "recent → teacher, augmented → student/critic (teaches drift "
                        "robustness via DMD, no regression needed).")
    # DMD specifics
    p.add_argument("--num_inference_steps", type=int, default=4,
                   help="Student's few-step count. The denoising timesteps are "
                        "derived from the scheduler (shift=flow_shift) so they "
                        "match inference EXACTLY.")
    p.add_argument("--denoising_step_list", default="",
                   help="Optional manual override (comma-sep). Leave empty to "
                        "derive from the scheduler (recommended).")
    p.add_argument("--dfake_gen_update_ratio", type=int, default=5)
    p.add_argument("--flow_shift", type=float, default=5.0)
    p.add_argument("--dmd_normalize_grad", action="store_true", default=True)
    p.add_argument("--real_guidance_scale", type=float, default=3.0,
                   help="CFG scale on the teacher (real score). Helios default 3.0. "
                        "0 disables CFG (matches unguided teacher → blurrier).")
    p.add_argument("--negative_prompt", default=(
        "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
        "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，"
        "畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"),
        help="Used as the CFG uncond branch for the teacher (real score).")
    # ─── GAN (One-Forcing) ───
    p.add_argument("--gan_g_weight", type=float, default=0.03,
                   help="Weight of gan_g_loss added to generator's total loss.")
    p.add_argument("--gan_d_weight", type=float, default=0.03,
                   help="Weight on the discriminator (cls_branch) update.")
    p.add_argument("--gan_feature_layers", default="13,21,29",
                   help="WanModel block indices whose outputs feed the cls_branch.")
    p.add_argument("--gan_ffn_dim", type=int, default=4096,
                   help="FFN hidden dim inside each GanCrossAttnBlock (One-Forcing uses 8192).")
    p.add_argument("--gan_num_heads", type=int, default=12)
    p.add_argument("--no_gan_offload", action="store_true",
                   help="Skip CPU activation offload in the GAN-D/GAN-G forwards "
                        "(faster, but uses more GPU memory). Safe when shared-base "
                        "freed enough room; OOM risk otherwise.")
    p.add_argument("--grad_clip", type=float, default=0.0,
                   help="Max grad norm (clip_grad_norm_) on student/critic before each "
                        "optimizer step. 0=off. Set ~1.0 to stop the recycle-feedback "
                        "dmd_g runaway (err_norm→∞) that hits harder data at 14B.")
    # ─── SVI-style error recycle + self-correcting loss ───
    p.add_argument("--error_alpha", type=float, default=0.5,
                   help="Final scale on injected error (after warmup ramp). 0=off (sc loss still active).")
    p.add_argument("--error_buffer_size", type=int, default=500,
                   help="Max entries in error replay buffer (per rank, FIFO). SVI uses 500.")
    p.add_argument("--error_warmup_count", type=int, default=50,
                   help="Wait until buffer has >= this many entries before injecting (and ramping α).")
    p.add_argument("--error_alpha_ramp_steps", type=int, default=200,
                   help="Linearly ramp α from 0 to --error_alpha over this many steps after warmup. "
                        "0 = apply --error_alpha immediately.")
    p.add_argument("--sc_weight", type=float, default=0.5,
                   help="Weight on self-correcting L1 loss between corrupted x_pred and clean x_pred.")
    p.add_argument("--error_inject_prob", type=float, default=0.8,
                   help="Probability of injecting error on each step (after warmup). "
                        "1.0 = always inject, 0.8 = 80% inject + 20% clean (SVI-style). "
                        "When skipped, alpha_eff=0 and student sees clean ref.")
    p.add_argument("--error_collect_start_step", type=int, default=0,
                   help="Global step before which error collection is skipped. "
                        "Use 100-300 if you want DMD to stabilize before collecting "
                        "(early residuals may include untrained-LoRA noise, not just drift).")
    # ─── EMA (One-Forcing default) ───
    p.add_argument("--ema_decay", type=float, default=0.99)
    p.add_argument("--ema_start_step", type=int, default=200,
                   help="Start EMA tracking after this many global steps.")
    # training
    p.add_argument("--learning_rate_student", type=float, default=5e-6)
    p.add_argument("--learning_rate_critic",  type=float, default=5e-6)
    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--global_step_offset", type=int, default=0)
    # checkpointing
    p.add_argument("--output_path", required=True)
    p.add_argument("--resume_student_from", default=None)
    p.add_argument("--resume_critic_from", default=None)
    p.add_argument("--use_gradient_checkpointing", action="store_true", default=True)
    p.add_argument("--plp", action="store_true",
                   help="Persistent Latent Propagation: recent reference = previous "
                        "chunk's VIDEO-mode latent (last latent frame) instead of "
                        "an image-mode single-frame encode. Matches PLP inference. "
                        "Needs recent_window from ChunkAwareDataset (auto when set).")
    p.add_argument("--aspect_crop", action="store_true",
                   help="Resize-to-cover + center-crop instead of stretch (stops 3:4 "
                        "sources being squished). Must match the teacher's setting.")
    p.add_argument("--motion_weight_beta", type=float, default=0.0,
                   help="Novel-2 (motion-targeted recycle): scale the recycled error "
                        "injection by a per-pixel motion map derived FOR FREE from the "
                        "pose-control video (|Δpose| over time), so self-correction "
                        "focuses on high-motion regions where blur accumulates. "
                        "weight = 1 + beta*motion01 (beta=0 → off, uniform injection; "
                        "try 1.0-2.0). Prior art: motion-weighted LOSS (MimicMotion/MoCo); "
                        "our twist applies it to the recycle INJECTION + uses pose, not flow.")
    p.add_argument("--vel_recent", action="store_true",
                   help="PLP-v2 velocity-aware recent: carry the prev chunk's LAST TWO "
                        "latent frames (recent₋₂ @ ref T=1, recent₋₁ @ T=2) so the 1-step "
                        "student has motion context, not just position. Targets the "
                        "motion-region blur that survives flat PLP. Requires --plp. "
                        "Recycle error injects into the last recent frame. Must match "
                        "--vel_recent at inference.")
    # --- frequency-band recycle (#2; default OFF = flat single-bank FIFO) ---
    p.add_argument("--recycle_freq_split", action="store_true",
                   help="Split recycle residuals into low/high spatial-freq banks "
                        "and inject α_low·low + α_high·high (decouples background/"
                        "stability drift from detail/clarity drift). OFF = original "
                        "flat FIFO (bit-identical to baseline).")
    p.add_argument("--recycle_band_k", type=int, default=2,
                   help="Low-pass pool kernel = band cutoff scale (2 → 58/42 split, "
                        "4 → 33/67). Only used with --recycle_freq_split.")
    p.add_argument("--error_alpha_low", type=float, default=None,
                   help="Injection scale for the LOW band (stability). Defaults to "
                        "--error_alpha when unset.")
    p.add_argument("--error_alpha_high", type=float, default=None,
                   help="Injection scale for the HIGH band (clarity). Defaults to "
                        "--error_alpha when unset.")
    p.add_argument("--latent_cache", default=None,
                   help="Dir for on-disk per-chunk latent cache. First epoch encodes "
                        "+ saves; later epochs/reruns load (skips decode + VAE/T5/CLIP). "
                        "Deterministic under --plp; do NOT use without --plp (the "
                        "random recent aug would be frozen). Clear the dir if the base "
                        "model changes.")
    return p


# ===========================================================================
# Main
# ===========================================================================
def main():
    args = get_parser().parse_args()
    accelerator = accelerate.Accelerator()
    device = accelerator.device
    dtype = torch.bfloat16
    is_main = accelerator.is_main_process
    num_proc = accelerator.num_processes
    target_modules = args.lora_target_modules.split(",")
    if is_main:
        os.makedirs(args.output_path, exist_ok=True)
    accelerator.wait_for_everyone()

    rank = accelerator.process_index
    if is_main:
        print(f"[accel] num_processes={num_proc} (rank {rank}, device {device})")

    # Shared seed BEFORE model building so peft's LoRA kaiming_uniform_ init
    # is bit-identical across ranks. Per-rank seed gets set AFTER model build.
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Empty / "none" → vanilla base teacher (no sink fused). Student then has to
    # learn sink itself, which pure DMD can't do without a regression loss —
    # this is the experimental "one-shot from base" config.
    use_sink = bool(args.teacher_lora_path) and args.teacher_lora_path.lower() != "none"
    if is_main:
        print(f"[setup] sink LoRA fused into base: {use_sink} "
              f"({'teacher=base+sink' if use_sink else 'teacher=VANILLA base'})")

    if args.shared_base:
        # ---- SHARED-BASE: ONE base+sink DiT + student/critic adapters (fits 14B) ----
        if is_main: print("[setup] SHARED-BASE: 1 base+sink DiT + 2 LoRA adapters (student/critic) ...")
        _pipe = build_pipe(device, dtype, model_id=args.base_model_id)
        if use_sink:
            fuse_sink_lora_into_pipe(_pipe, args.teacher_lora_path)
        _pipe.dit.requires_grad_(False)                       # base+sink frozen
        add_named_lora(_pipe.dit, target_modules, args.lora_rank)   # student + critic adapters
        _pipe.dit.train()
        if args.resume_student_from:
            load_lora_into_trainable(_pipe.dit, load_state_dict(args.resume_student_from))
            if is_main: print(f"[resume] student LoRA from {args.resume_student_from}")
        if args.resume_critic_from:
            load_lora_into_trainable(_pipe.dit, load_state_dict(args.resume_critic_from))
            if is_main: print(f"[resume] critic LoRA from {args.resume_critic_from}")
        if args.fsdp:
            # shard each DiTBlock across GPUs → ~1/N weights/GPU (lets us drop GAN offload)
            fsdp_wrap_blocks(_pipe.dit)
            if is_main: print(f"[setup] FSDP: per-DiTBlock FULL_SHARD across {num_proc} GPUs")
        teacher_pipe = student_pipe = critic_pipe = _pipe     # all share the one DiT
    else:
        # ---------------- Build 3 pipes (each rank gets its own 3 DiTs) ----------------
        if is_main: print("[setup] building teacher (frozen) ...")
        teacher_pipe = build_pipe(device, dtype, model_id=args.base_model_id)
        if use_sink:
            fuse_sink_lora_into_pipe(teacher_pipe, args.teacher_lora_path)
        teacher_pipe.dit.requires_grad_(False)
        teacher_pipe.dit.eval()

        if is_main: print("[setup] building student (trainable LoRA) ...")
        student_pipe = build_pipe(device, dtype, model_id=args.base_model_id)
        if use_sink:
            fuse_sink_lora_into_pipe(student_pipe, args.teacher_lora_path)
        student_pipe.dit = add_trainable_lora(student_pipe.dit, target_modules, args.lora_rank)
        student_pipe.dit.train()   # gradient_checkpoint_forward checks self.training
        if args.resume_student_from:
            load_lora_into_trainable(student_pipe.dit, load_state_dict(args.resume_student_from))
            if is_main: print(f"[resume] student LoRA from {args.resume_student_from}")

        if is_main: print("[setup] building critic (trainable LoRA) ...")
        critic_pipe = build_pipe(device, dtype, model_id=args.base_model_id)
        if use_sink:
            fuse_sink_lora_into_pipe(critic_pipe, args.teacher_lora_path)
        critic_pipe.dit = add_trainable_lora(critic_pipe.dit, target_modules, args.lora_rank)
        critic_pipe.dit.train()
        if args.resume_critic_from:
            load_lora_into_trainable(critic_pipe.dit, load_state_dict(args.resume_critic_from))
            if is_main: print(f"[resume] critic LoRA from {args.resume_critic_from}")

    # ---------------- ClsBranch discriminator (hangs off the critic) ----------------
    gan_layers = [int(x) for x in args.gan_feature_layers.split(",")]
    dit_dim = critic_pipe.dit.dim
    cls_branch = ClsBranch(
        num_layers=len(gan_layers),
        dim=dit_dim,
        ffn_dim=args.gan_ffn_dim,
        num_heads=args.gan_num_heads,
        num_class=1,
    ).to(device=device, dtype=dtype)
    cls_branch.train()
    if is_main:
        n_cls = sum(p.numel() for p in cls_branch.parameters())
        print(f"[setup] cls_branch: {n_cls/1e6:.1f}M params, layers={gan_layers}")

    # ---------------- Optimizers ----------------
    if args.shared_base:
        # student_pipe.dit IS critic_pipe.dit → split by adapter name.
        student_params = adapter_params(student_pipe.dit, "student")
        critic_params  = adapter_params(critic_pipe.dit, "critic")
    else:
        student_params = trainable_params(student_pipe.dit)
        critic_params  = trainable_params(critic_pipe.dit)
    cls_params     = list(cls_branch.parameters())
    if is_main:
        print(f"[setup] student trainable params: {sum(p.numel() for p in student_params)/1e6:.1f}M")
        print(f"[setup] critic  trainable params: {sum(p.numel() for p in critic_params )/1e6:.1f}M")

    student_optimizer = torch.optim.AdamW(student_params, lr=args.learning_rate_student, weight_decay=1e-2)
    # critic_optimizer drives BOTH the denoising critic AND the cls_branch — they
    # are updated together by the GAN-D step and by critic_loss.
    critic_optimizer  = torch.optim.AdamW(critic_params + cls_params,
                                          lr=args.learning_rate_critic, weight_decay=1e-2)

    # ---------------- Resume optimizer state (AdamW moments) ----------------
    # The state sidecar sits next to the student LoRA: step-N.safetensors →
    # step-N_state.pt. Restoring it avoids the cold-start momentum re-warm.
    if args.resume_student_from:
        # cls_branch state (also next to student LoRA: ..._cls.pt)
        cls_path = args.resume_student_from.replace(".safetensors", "_cls.pt")
        if os.path.isfile(cls_path):
            cls_branch.load_state_dict(torch.load(cls_path, map_location=device))
            if is_main: print(f"[resume] cls_branch state from {cls_path}")
        elif is_main:
            print(f"[resume] cls_branch NOT loaded (no {cls_path}) — fresh init.")

        # EMA state (saved as LoRA-format safetensors)
        ema_resume_path = args.resume_student_from.replace(".safetensors", "_ema.safetensors")
        if os.path.isfile(ema_resume_path):
            from diffsynth.core import load_state_dict as _lsd
            ema_sd = _lsd(ema_resume_path)
            # Match against current student LoRA param names
            main._ema_state = {}
            for n, p in student_pipe.dit.named_parameters():
                if p.requires_grad and n in ema_sd:
                    main._ema_state[n] = ema_sd[n].to(device=p.device, dtype=p.dtype)
            if is_main:
                print(f"[resume] EMA state from {ema_resume_path} ({len(main._ema_state)} tensors)")
        elif is_main:
            print(f"[resume] EMA NOT loaded (no {ema_resume_path}) — will init when started.")

        # Optimizer states (+ SVI error buffer state)
        state_path = args.resume_student_from.replace(".safetensors", "_state.pt")
        if os.path.isfile(state_path):
            # Load to CPU: the optimizer state (full AdamW moments, ~3.4GB at
            # 14B) must NOT land on the GPU — under FSDP it mismatches the
            # sharded params and is discarded anyway, so putting it on-device
            # just burns/fragments VRAM at the tight 80GB boundary. AdamW's
            # load_state_dict moves the adopted tensors onto the param devices
            # itself; on mismatch we cold-start and the CPU copy is freed.
            ckpt = torch.load(state_path, map_location="cpu")
            try:
                student_optimizer.load_state_dict(ckpt["student_optimizer"])
                if is_main:
                    print(f"[resume] student optimizer state from {state_path} "
                          f"(saved at step {ckpt.get('global_step', '?')})")
            except (ValueError, KeyError, RuntimeError) as e:
                if is_main:
                    print(f"[resume] student_optimizer state mismatch ({e}) — "
                          f"cold start student moments (FSDP transition?).")
            try:
                critic_optimizer.load_state_dict(ckpt["critic_optimizer"])
            except (ValueError, KeyError, RuntimeError) as e:
                if is_main:
                    print(f"[resume] critic_optimizer state mismatch ({e}) — cold start critic moments.")
            # SVI buffer state (may not exist in old checkpoints — falls back to empty)
            resume_buffer = ckpt.get("error_buffer", None)
            resume_warmup_done_step = ckpt.get("error_warmup_done_step", None)
            if is_main and resume_buffer is not None:
                print(f"[resume] error buffer with {len(resume_buffer)} entries, "
                      f"warmup_done_step={resume_warmup_done_step}")
        else:
            resume_buffer = None
            resume_warmup_done_step = None
            if is_main:
                print(f"[resume] WARNING: no optimizer state at {state_path} — "
                      f"AdamW moments + error buffer start cold.")
    else:
        resume_buffer = None
        resume_warmup_done_step = None

    # ---------------- Derive denoising timesteps from the scheduler ----------------
    # Guarantees train timesteps == inference timesteps (sigma = t/1000).
    if args.denoising_step_list.strip():
        denoising_step_list = [float(x) for x in args.denoising_step_list.split(",")]
    else:
        student_pipe.scheduler.set_timesteps(
            num_inference_steps=args.num_inference_steps, shift=args.flow_shift,
        )
        denoising_step_list = [float(t) for t in student_pipe.scheduler.timesteps]
    if is_main:
        print(f"[setup] denoising_step_list (shift={args.flow_shift}): "
              f"{[round(t, 1) for t in denoising_step_list]}")

    # Negative prompt embedding for the teacher's CFG uncond branch. Fixed text
    # → encode ONCE and reuse every step.
    neg_prompt_embed = None
    if args.real_guidance_scale != 0:
        neg_prompt_embed = encode_prompt(teacher_pipe, args.negative_prompt, dtype, device)
        if is_main:
            print(f"[setup] teacher CFG enabled (real_guidance_scale={args.real_guidance_scale})")

    # Models are built with identical weights across ranks. Now switch to a
    # per-rank seed so noise / timestep / aug sampling differs per rank — more
    # diverse training signal at the same wall-clock step. The +global_step_offset
    # term advances the RNG on resume so we don't replay the same noise/timestep
    # stream we already trained on.
    seed = 42 + rank * 10000 + args.global_step_offset
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ---------------- Dataset ----------------
    dataset = ChunkAwareDataset(
        csv_path=args.dataset_metadata_path,
        height=args.height,
        width=args.width,
        chunk_frames=args.num_frames,
        recent_aug_strength=args.recent_aug_strength,
        dataset_repeat=args.dataset_repeat,
        plp=args.plp,
        aspect_crop=args.aspect_crop,
    )
    # Resume position: map global_step_offset back to (epoch, batch-in-epoch)
    # so we CONTINUE the stream instead of replaying epoch 0 from the top.
    steps_per_epoch = (len(dataset) // num_proc) if num_proc > 1 else len(dataset)
    steps_per_epoch = max(steps_per_epoch, 1)
    start_epoch  = args.global_step_offset // steps_per_epoch
    skip_batches = args.global_step_offset %  steps_per_epoch

    # Base sampler: DistributedSampler (multi-GPU) or RandomSampler (single).
    if num_proc > 1:
        base_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=num_proc, rank=rank, shuffle=True, drop_last=True,
        )
    else:
        base_sampler = torch.utils.data.RandomSampler(dataset)
    # Wrap so the skipped batches are dropped at the index level (no decode).
    sampler = ResumableSampler(base_sampler, skip_first=skip_batches, skip_epoch=start_epoch)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        shuffle=False,            # sampler handles shuffling
        batch_size=None,
        num_workers=4,
        persistent_workers=True,
        prefetch_factor=2,
    )

    # ---------------- Training log (main process only) ----------------
    log_file = None
    if is_main:
        log_file = open(os.path.join(args.output_path, "dmd_train.log"), "a", buffering=1)
        log_file.write(f"=== run start {time.strftime('%Y-%m-%d %H:%M:%S')} | "
                       f"world_size={num_proc} "
                       f"lr_s={args.learning_rate_student} lr_c={args.learning_rate_critic} "
                       f"ratio={args.dfake_gen_update_ratio} steps={denoising_step_list} "
                       f"offset={args.global_step_offset} ===\n")

    if is_main and args.global_step_offset:
        print(f"[resume] offset={args.global_step_offset} → start at epoch "
              f"{start_epoch}, skip {skip_batches}/{steps_per_epoch} batches "
              f"(index-level, no decode)")

    # ---------------- SVI Error Buffer + state ----------------
    error_buffer = ErrorBuffer(max_size=args.error_buffer_size,
                               freq_split=args.recycle_freq_split,
                               band_k=args.recycle_band_k)
    error_warmup_done_step = None
    # Restore from resume checkpoint if available
    if resume_buffer is not None:
        # Restores flat buf AND freq buf_low/buf_high (new dict format), or a
        # legacy flat list — load_state_dict forces CPU bf16 + trims to max_size.
        error_buffer.load_state_dict(resume_buffer)
    if resume_warmup_done_step is not None:
        error_warmup_done_step = resume_warmup_done_step
    # Guard: if the ACTIVE buffer (mode-specific) is under-filled — e.g. resuming
    # a flat checkpoint into --recycle_freq_split, where buf_low/high are empty —
    # the restored warmup_done_step would make α ramp jump as the buffer refills.
    # Force a fresh warmup in that case.
    if error_warmup_done_step is not None and len(error_buffer) < args.error_warmup_count:
        if is_main:
            print(f"[resume] active buffer under-filled ({len(error_buffer)} < "
                  f"{args.error_warmup_count}; mode switch?) → reset warmup, re-fill before ramp")
        error_warmup_done_step = None
    if is_main:
        print(f"[setup] error recycle: alpha={args.error_alpha}, buffer={args.error_buffer_size}, "
              f"warmup_count={args.error_warmup_count}, ramp_steps={args.error_alpha_ramp_steps}, "
              f"sc_weight={args.sc_weight}")
        print(f"[setup] error_buffer initial size: {len(error_buffer)}, "
              f"warmup_done_step: {error_warmup_done_step}")

    # Per-band injection scales (freq mode). Band α's fall back to --error_alpha.
    # inject_enabled gates the whole recycle: in freq mode a nonzero band α is
    # enough even when --error_alpha is 0 (so you can drive only one band).
    base_alpha_lo = args.error_alpha if args.error_alpha_low  is None else args.error_alpha_low
    base_alpha_hi = args.error_alpha if args.error_alpha_high is None else args.error_alpha_high
    if args.recycle_freq_split:
        inject_enabled = (base_alpha_lo > 0.0) or (base_alpha_hi > 0.0)
        if is_main:
            print(f"[error-recycle] FREQ-SPLIT on (band_k={args.recycle_band_k}): "
                  f"α_low={base_alpha_lo}, α_high={base_alpha_hi}")
    else:
        inject_enabled = (args.error_alpha > 0.0)

    if args.latent_cache and args.recent_aug_strength > 0 and is_main:
        print(f"[warn] --latent_cache with recent_aug_strength={args.recent_aug_strength}: the "
              f"random CLIP recent-aug is FROZEN on first cache write (cache key has no aug "
              f"seed). Under PLP set --recent_aug_strength 0 (aug is moot there), or accept "
              f"a fixed per-sample augmentation.")

    # ---------------- Training loop ----------------
    local_step = 0
    freq_inject_logged = False     # one-shot log of actual a_lo/a_hi at first real inject
    for epoch in range(start_epoch, start_epoch + args.num_epochs):
        sampler.set_epoch(epoch)   # reshuffle + apply skip on the start epoch only
        pbar = tqdm(dataloader, desc=f"epoch {epoch}", disable=not is_main)
        for batch in pbar:
            # ─── encode all conditioning once ───
            cond = encode_batch(student_pipe, batch, dtype, device, no_recent=args.no_recent, plp=args.plp, cache_dir=args.latent_cache, aspect_crop=args.aspect_crop, base_model_id=args.base_model_id, vel_recent=args.vel_recent)
            target_latents    = cond["target_latents"]
            control_latents   = cond["control_latents"]
            prompt_embed      = cond["prompt_embed"]

            # Route BOTH ref latent and CLIP feature per augment mode:
            #   off        → clean for everyone
            #   symmetric  → augmented for everyone (teacher must be augment-robust)
            #   asymmetric → clean to teacher, augmented to student/critic
            if args.recent_augment_mode == "off":
                ref_student = ref_teacher = cond["reference_latents_clean"]
                clip_student = clip_teacher = cond["clip_feature_clean"]
            elif args.recent_augment_mode == "asymmetric":
                ref_student  = cond["reference_latents_aug"]
                ref_teacher  = cond["reference_latents_clean"]
                clip_student = cond["clip_feature_aug"]
                clip_teacher = cond["clip_feature_clean"]
            else:  # symmetric
                ref_student = ref_teacher = cond["reference_latents_aug"]
                clip_student = clip_teacher = cond["clip_feature_aug"]

            B = target_latents.shape[0]

            # ─── SVI: prepare clean/corrupted reference latents for two rollouts ───
            # Clean reference (no error) is the target for self-correcting loss.
            # Corrupted reference (clean + α·sampled_error in recent slice) is what
            # student actually sees during the gradient-bearing rollout.
            ref_student_clean = ref_student      # baseline (may already have random aug)
            alpha_eff = 0.0
            ramp = 0.0                            # hoisted: used by freq-split injection
            # Use GLOBAL step for ramp calc so resume picks up where it left off.
            # (local_step resets to 0 on resume; global_step = offset + local_step.)
            cur_global_step = args.global_step_offset + local_step
            # Probabilistic injection: with prob `error_inject_prob`, do error
            # injection; otherwise use clean ref (SVI's clean_prob mechanism).
            this_step_inject = (random.random() < args.error_inject_prob)
            if (inject_enabled
                    and this_step_inject
                    and len(error_buffer) >= args.error_warmup_count):
                if error_warmup_done_step is None:
                    error_warmup_done_step = cur_global_step
                    if is_main:
                        print(f"[error-recycle] buffer warmup done at global_step={cur_global_step}, "
                              f"begin α ramp")
                if args.error_alpha_ramp_steps > 0:
                    ramp = min(1.0, (cur_global_step - error_warmup_done_step) / args.error_alpha_ramp_steps)
                else:
                    ramp = 1.0
                # alpha_eff = master "injection active + scale" used by sc-loss /
                # buffer-push gating. In freq mode use the stronger band.
                if args.recycle_freq_split:
                    alpha_eff = max(base_alpha_lo, base_alpha_hi) * max(0.0, ramp)
                else:
                    alpha_eff = args.error_alpha * max(0.0, ramp)

            if alpha_eff > 0.0:
                # SVI injection targets the "recent" slice (T=1) of reference_latents.
                # Under --no_recent, the reference is sink-only (T=1) and there is
                # no recent slice → skip injection entirely.
                if args.no_recent or ref_student.shape[2] < 2:
                    if local_step == 0 and is_main:
                        print("[error-recycle] --no_recent or single-frame ref detected; "
                              "skipping error injection.")
                    ref_student_corrupt = ref_student
                    alpha_eff = 0.0
                elif args.recycle_freq_split:
                    # Two-bank injection: α_low·low + α_high·high (each band sampled
                    # independently). base_alpha_{lo,hi} fall back to error_alpha.
                    a_lo = base_alpha_lo * max(0.0, ramp)
                    a_hi = base_alpha_hi * max(0.0, ramp)
                    lo, hi = error_buffer.sample_bands(device=ref_student.device, dtype=ref_student.dtype)
                    ref_student_corrupt = ref_student.clone()
                    _mw = motion_weight_map(control_latents, args.motion_weight_beta) if args.motion_weight_beta > 0.0 else 1.0
                    ref_student_corrupt[:, :, -1, :, :] = ref_student_corrupt[:, :, -1, :, :] + _mw * (a_lo * lo + a_hi * hi)
                    if not freq_inject_logged and is_main:
                        print(f"[error-recycle] first freq inject @ step {cur_global_step}: "
                              f"a_lo={a_lo:.4f} a_hi={a_hi:.4f} (ramp={ramp:.3f})")
                        freq_inject_logged = True
                else:
                    err = error_buffer.sample(device=ref_student.device, dtype=ref_student.dtype)
                    if args.motion_weight_beta > 0.0:
                        err = err * motion_weight_map(control_latents, args.motion_weight_beta)
                    ref_student_corrupt = ref_student.clone()
                    ref_student_corrupt[:, :, -1, :, :] = ref_student_corrupt[:, :, -1, :, :] + alpha_eff * err
            else:
                ref_student_corrupt = ref_student

            # ─── Shared randomness for two rollouts (clean vs corrupt) ─────
            # The two rollouts MUST differ only by reference_latents. Otherwise
            # sc_loss = L1(corrupt, clean) ends up regressing two different
            # random noise inputs to the same output, which pressures student
            # toward noise-invariance (= mode collapse).
            sc_loss_active = (alpha_eff > 0.0 and args.sc_weight > 0.0)

            # The clean rollout has TWO consumers:
            #   1. SC loss (gated by sc_loss_active)
            #   2. Buffer push, which requires a clean source whenever
            #      alpha_eff > 0 (otherwise x_pred is corrupt-ref output and
            #      pushing it creates a polluted feedback loop where injected
            #      noise gets re-injected).
            # When alpha_eff = 0, x_pred IS the clean rollout (ref_student_corrupt
            # falls back to ref_student above), so no separate clean rollout
            # is needed — the line-1020 fallback can safely use x_pred.
            will_push = (cur_global_step >= args.error_collect_start_step)
            need_clean_rollout = sc_loss_active or (alpha_eff > 0.0 and will_push)
            N_steps = len(denoising_step_list)
            shared_exit_idx     = random.randrange(N_steps)
            shared_initial_noisy = torch.randn(target_latents.shape, dtype=dtype, device=device)
            shared_renoise_noises = [
                torch.randn(target_latents.shape, dtype=dtype, device=device)
                for _ in range(shared_exit_idx)        # only need this many fresh re-noises
            ]

            if args.shared_base: use_adapter(student_pipe.dit, "student")
            # ─── CLEAN ROLLOUT (no grad) — used by SC loss and/or buffer push ───
            if need_clean_rollout:
                with torch.no_grad():
                    x_pred_clean_target, _ = rollout_student(
                        student_pipe.dit, denoising_step_list, tuple(target_latents.shape),
                        prompt_embed, control_latents, ref_student_clean, clip_student,
                        device, dtype,
                        use_gradient_checkpointing=args.use_gradient_checkpointing,
                        initial_noisy=shared_initial_noisy,
                        exit_idx=shared_exit_idx,
                        renoise_noises=shared_renoise_noises,
                    )
                    x_pred_clean_target = x_pred_clean_target.detach()
            else:
                x_pred_clean_target = None

            # ─── CORRUPT ROLLOUT (with grad) — main path ───
            # Uses the SAME initial noise + exit_idx as the clean rollout above.
            x_pred, t_gen = rollout_student(
                student_pipe.dit, denoising_step_list, tuple(target_latents.shape),
                prompt_embed, control_latents, ref_student_corrupt, clip_student,
                device, dtype,
                use_gradient_checkpointing=args.use_gradient_checkpointing,
                initial_noisy=shared_initial_noisy,
                exit_idx=shared_exit_idx,
                renoise_noises=shared_renoise_noises,
            )

            if args.shared_base: use_adapter(critic_pipe.dit, "critic")
            # ─── critic update × N (use x_pred.detach() — no grad to student) ───
            for _ in range(args.dfake_gen_update_ratio):
                t_c = sample_critic_timestep(shift=args.flow_shift)
                sigma_c = timestep_to_sigma(t_c)
                noise_c = torch.randn_like(x_pred)
                x_pred_noisy_c, _ = add_noise_flow(x_pred.detach(), sigma_c, noise_c)

                v_critic_pred = dit_forward(
                    critic_pipe.dit, x_pred_noisy_c, t_c, prompt_embed,
                    control_latents, ref_student, clip_student, device, dtype,
                    use_gradient_checkpointing=args.use_gradient_checkpointing,
                )
                critic_loss = compute_critic_loss(v_critic_pred, x_pred, noise_c)
                critic_optimizer.zero_grad()
                critic_loss.backward()
                all_reduce_grads(cls_params if args.fsdp else critic_params + cls_params, num_proc)   # sync
                if args.grad_clip > 0: torch.nn.utils.clip_grad_norm_(critic_params + cls_params, args.grad_clip)
                critic_optimizer.step()

            # ─── GAN-D update × N (One-Forcing's discriminator step) ───
            # Train critic + cls_branch to distinguish noised fake vs noised real.
            # MEMORY: stack fake+real into ONE forward (2× batch instead of 2 sequential
            # forwards keeping both graphs alive), and CPU-offload activations.
            for _ in range(args.dfake_gen_update_ratio):
                t_d = sample_critic_timestep(shift=args.flow_shift)
                sigma_d = timestep_to_sigma(t_d)
                noise_d = torch.randn_like(x_pred)
                fake_noisy_d, _ = add_noise_flow(x_pred.detach(), sigma_d, noise_d)
                real_noisy_d, _ = add_noise_flow(target_latents,   sigma_d, noise_d)

                stacked = torch.cat([fake_noisy_d, real_noisy_d], dim=0)
                # T5/CLIP/control conditioning: also stack 2× along batch dim
                prompt_2 = torch.cat([prompt_embed, prompt_embed], dim=0)
                control_2 = torch.cat([control_latents, control_latents], dim=0)
                ref_2 = torch.cat([ref_student, ref_student], dim=0)
                clip_2 = torch.cat([clip_student, clip_student], dim=0) if clip_student is not None else None

                with (contextlib.nullcontext() if args.no_gan_offload
                      else torch.autograd.graph.save_on_cpu(pin_memory=False)):
                    with FeatureCapturer(critic_pipe.dit, gan_layers) as cap:
                        _ = dit_forward(
                            critic_pipe.dit, stacked, t_d, prompt_2,
                            control_2, ref_2, clip_2, device, dtype,
                            use_gradient_checkpointing=args.use_gradient_checkpointing,
                        )
                        stacked_feats = cap.features()
                    stacked_logit = cls_branch(stacked_feats)
                fake_logit_d, real_logit_d = stacked_logit.chunk(2, dim=0)

                d_loss = gan_d_loss(real_logit_d, fake_logit_d) * args.gan_d_weight
                critic_optimizer.zero_grad()
                d_loss.backward()
                all_reduce_grads(cls_params if args.fsdp else critic_params + cls_params, num_proc)
                if args.grad_clip > 0: torch.nn.utils.clip_grad_norm_(critic_params + cls_params, args.grad_clip)
                critic_optimizer.step()

            # ─── generator update × 1 (DMD + GAN-G, with grad through x_pred) ───
            t_dmd = sample_critic_timestep(shift=args.flow_shift)
            sigma_dmd = timestep_to_sigma(t_dmd)
            noise_dmd = torch.randn_like(x_pred)
            x_pred_noisy_dmd, _ = add_noise_flow(x_pred, sigma_dmd, noise_dmd)

            with torch.no_grad():
                if args.shared_base: use_adapter(critic_pipe.dit, "critic")
                # fake score (critic), conditional only — student-side recent
                v_fake = dit_forward(
                    critic_pipe.dit, x_pred_noisy_dmd, t_dmd, prompt_embed,
                    control_latents, ref_student, clip_student, device, dtype,
                )
                pred_fake = velocity_to_x0(v_fake, x_pred_noisy_dmd, sigma_dmd)

                if args.shared_base: use_adapter(teacher_pipe.dit, None)   # base+sink only
                # real score (teacher), CFG-enhanced — teacher-side (clean) recent
                v_real_cond = dit_forward(
                    teacher_pipe.dit, x_pred_noisy_dmd, t_dmd, prompt_embed,
                    control_latents, ref_teacher, clip_teacher, device, dtype,
                )
                pred_real_cond = velocity_to_x0(v_real_cond, x_pred_noisy_dmd, sigma_dmd)
                if neg_prompt_embed is not None:
                    v_real_uncond = dit_forward(
                        teacher_pipe.dit, x_pred_noisy_dmd, t_dmd, neg_prompt_embed,
                        control_latents, ref_teacher, clip_teacher, device, dtype,
                    )
                    pred_real_uncond = velocity_to_x0(v_real_uncond, x_pred_noisy_dmd, sigma_dmd)
                else:
                    pred_real_uncond = None
                pred_real = cfg_real_x0(pred_real_cond, pred_real_uncond, args.real_guidance_scale)

            grad = compute_dmd_gradient(
                pred_fake, pred_real, x_pred, normalize=args.dmd_normalize_grad,
            )
            dmd_g_loss = compute_dmd_loss(x_pred, grad)

            # GAN-G: push x_pred toward "real" via the (frozen-this-step)
            # critic+cls_branch discriminator. Critic/cls grads from this loss
            # are computed but discarded — they get zeroed before the next
            # critic_loss step (`critic_optimizer.zero_grad()`).
            # MEMORY: CPU-offload activations (One-Forcing's gan_activation_cpu_offload).
            t_g = sample_critic_timestep(shift=args.flow_shift)
            sigma_g = timestep_to_sigma(t_g)
            noise_g = torch.randn_like(x_pred)
            fake_noisy_g, _ = add_noise_flow(x_pred, sigma_g, noise_g)
            if args.shared_base: use_adapter(critic_pipe.dit, "critic")   # discriminator
            with (contextlib.nullcontext() if args.no_gan_offload
                  else torch.autograd.graph.save_on_cpu(pin_memory=False)):
                with FeatureCapturer(critic_pipe.dit, gan_layers) as cap:
                    _ = dit_forward(
                        critic_pipe.dit, fake_noisy_g, t_g, prompt_embed,
                        control_latents, ref_student, clip_student, device, dtype,
                        use_gradient_checkpointing=args.use_gradient_checkpointing,
                    )
                    fake_feats_g = cap.features()
                fake_logit_g = cls_branch(fake_feats_g)
            g_loss = gan_g_loss(fake_logit_g) * args.gan_g_weight

            # ─── SVI Self-correcting loss ───
            # L1 between corrupted x_pred (with grad) and clean x_pred (detached).
            # Trains student to be invariant to error in recent_ref.
            if x_pred_clean_target is not None:
                sc_loss = F.l1_loss(x_pred, x_pred_clean_target)
            else:
                sc_loss = x_pred.new_zeros(())

            gen_loss = dmd_g_loss + g_loss + args.sc_weight * sc_loss
            student_optimizer.zero_grad()
            gen_loss.backward()
            if not args.fsdp: all_reduce_grads(student_params, num_proc)   # FSDP reduces LoRA grads itself
            if args.grad_clip > 0: torch.nn.utils.clip_grad_norm_(student_params, args.grad_clip)
            student_optimizer.step()

            # ─── SVI COLLECT: push student's drift direction to buffer ───
            # error = student - real  (drift direction; matches SVI sign)
            # Skipped until cur_global_step >= error_collect_start_step to avoid
            # contaminating buffer with cold-start undertrained-LoRA noise.
            #
            # CAVEAT (latent space mismatch):
            #   We collect from target_latents[:, :, -1] which is a slice of
            #   the 13-frame VIDEO latent (Wan VAE's temporal convs see context).
            #   We inject into reference_latents[:, :, 1] which is a SINGLE-FRAME
            #   image VAE encode (no temporal context). The two latents have
            #   similar but not identical distributions.
            #   A strictly correct collect would be:
            #     student_last_pix = vae.decode(x_pred[:, :, -1:])    # 1 frame
            #     student_enc      = vae.encode(student_last_pix)     # single-frame latent
            #     err              = student_enc - real_single_frame_enc
            #   That adds 1 VAE decode + 1 VAE encode per step (~20% overhead).
            #   First-pass uses video-latent-slice as a proxy — direction should be
            #   approximately right; if err_norm looks wrong or sc_loss collapses,
            #   switching to the strict version is the first thing to try.
            err_norm = 0.0
            if cur_global_step >= args.error_collect_start_step:
                with torch.no_grad():
                    # Invariant guaranteed by `need_clean_rollout` upstream:
                    #   - alpha_eff > 0 ⇒ x_pred_clean_target is computed
                    #   - alpha_eff == 0 ⇒ x_pred IS clean (no injection happened),
                    #     so falling back to x_pred here is also a clean source.
                    # The fallback NEVER reads a corrupt-ref rollout into the buffer.
                    ref_x_pred = x_pred_clean_target if x_pred_clean_target is not None else x_pred
                    real_last    = target_latents[:, :, -1, :, :]
                    student_last = ref_x_pred[:,    :, -1, :, :].detach()
                    err = (student_last - real_last)
                    err_norm = float(err.abs().mean())          # for logging
                    for b in range(err.shape[0]):
                        error_buffer.push(err[b])      # [C, h, w]

            # ─── EMA update (One-Forcing: ema_weight=0.99, start after step 200) ───
            local_step += 1
            global_step = args.global_step_offset + local_step
            if global_step >= args.ema_start_step:
                with torch.no_grad():
                    if not hasattr(main, "_ema_state"):
                        main._ema_state = {n: p.data.clone()
                                           for n, p in student_pipe.dit.named_parameters()
                                           if p.requires_grad}
                        if is_main:
                            print(f"[ema] initialized at step {global_step}")
                    else:
                        d = args.ema_decay
                        for n, p in student_pipe.dit.named_parameters():
                            if p.requires_grad and n in main._ema_state:
                                main._ema_state[n].mul_(d).add_(p.data, alpha=1 - d)
            if is_main:
                pbar.set_postfix(
                    g=f"{gen_loss.item():.4f}", c=f"{critic_loss.item():.4f}",
                    gg=f"{g_loss.item():.4f}", gd=f"{d_loss.item():.4f}",
                    sc=f"{float(sc_loss):.4f}", a=f"{alpha_eff:.2f}",
                    en=f"{err_norm:.4f}", buf=len(error_buffer),
                    t=t_gen,
                )
                log_file.write(
                    f"{time.strftime('%H:%M:%S')} epoch={epoch} step={global_step} "
                    f"t_gen={t_gen} gen_loss={gen_loss.item():.6f} "
                    f"dmd_g={dmd_g_loss.item():.6f} gan_g={g_loss.item():.6f} "
                    f"sc_loss={float(sc_loss):.6f} alpha_eff={alpha_eff:.4f} "
                    f"err_norm={err_norm:.6f} buf={len(error_buffer)} "
                    f"critic_loss={critic_loss.item():.6f} gan_d={d_loss.item():.6f}\n"
                )

            # ─── save (main process only — params are sync'd across ranks anyway) ───
            if global_step % args.save_steps == 0:
                accelerator.wait_for_everyone()
                # Under FSDP the LoRA is sharded → gather full params for saving.
                # summon_full_params is COLLECTIVE: every rank enters/exits; with
                # rank0_only only rank0 materializes (on CPU) → rank0 then writes.
                _save_es = contextlib.ExitStack()
                if args.fsdp:
                    from torch.distributed.fsdp import FullyShardedDataParallel as _FSDP
                    for _b in (mm for mm in student_pipe.dit.modules() if isinstance(mm, _FSDP)):
                        _save_es.enter_context(_FSDP.summon_full_params(
                            _b, writeback=False, rank0_only=True, offload_to_cpu=True))
                if is_main:
                    student_path = os.path.join(args.output_path, f"step-{global_step}.safetensors")
                    critic_path  = os.path.join(args.output_path, f"step-{global_step}_critic.safetensors")
                    state_path   = os.path.join(args.output_path, f"step-{global_step}_state.pt")
                    save_lora_state_dict(student_pipe.dit, student_path, remove_prefix="dit.",
                                         adapter="student" if args.shared_base else None)
                    save_lora_state_dict(critic_pipe.dit,  critic_path,  remove_prefix="dit.",
                                         adapter="critic"  if args.shared_base else None)
                    cls_path     = os.path.join(args.output_path, f"step-{global_step}_cls.pt")
                    torch.save(cls_branch.state_dict(), cls_path)
                    # EMA LoRA — this is typically what you want for inference
                    if hasattr(main, "_ema_state"):
                        ema_path = os.path.join(args.output_path, f"step-{global_step}_ema.safetensors")
                        ema_sd = {}
                        for n, v in main._ema_state.items():
                            if "lora_A" in n or "lora_B" in n:
                                ema_sd[n] = v.detach().cpu().contiguous()
                        save_file(ema_sd, ema_path)
                    else:
                        ema_path = "(not yet started)"
                    # Optimizer moments + step counter + SVI buffer for seamless resume.
                    torch.save({
                        "student_optimizer": student_optimizer.state_dict(),
                        "critic_optimizer":  critic_optimizer.state_dict(),
                        "global_step": global_step,
                        # SVI error recycle state (per-rank buffer; we save rank-0's only)
                        "error_buffer": error_buffer.state_dict(),   # flat buf + freq buf_low/high + band_k
                        "error_warmup_done_step": error_warmup_done_step,
                    }, state_path)
                    print(f"[save] {student_path} + ema:{ema_path} + {cls_path} + {state_path} "
                          f"(buf={len(error_buffer)})")
                _save_es.close()   # exit FSDP summon (collective; all ranks)

    if is_main:
        log_file.write(f"=== run end {time.strftime('%Y-%m-%d %H:%M:%S')} | "
                       f"total_steps={args.global_step_offset + local_step} ===\n")
        log_file.close()


if __name__ == "__main__":
    main()
