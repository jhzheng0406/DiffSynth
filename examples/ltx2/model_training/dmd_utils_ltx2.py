"""
DMD2 distillation math for LTX-2 — port of examples/wanvideo/model_training/dmd_utils.py.

The flow-matching convention is IDENTICAL to Wan (verified against
diffsynth/diffusion/flow_match.py and model_fn_ltx2):
    x_t = sigma * noise + (1 - sigma) * x_0,   sigma in [0, 1]
    model predicts velocity v ≈ noise - x_0    (training_target = noise - sample)
    inverse:   x_0 = x_t - sigma * v
    timestep label = sigma * 1000              (model_fn_ltx2 divides by 1000)

So add_noise_flow / velocity_to_x0 / the DMD gradient math carry over
unchanged. The only LTX-2-specific parts are the two schedule functions:

  - derive_denoising_step_list: LTX-2's discrete schedule
    (FlowMatchScheduler.set_timesteps_ltx2) uses a qwen-image-style
    exponential shift PLUS a terminal shift toward sigma_terminal=0.1.
    For num_inference_steps=1 the terminal shift divides by zero (the only
    grid point is sigma=1.0), so the 1-step case is special-cased to
    [1000.0] — which is also exactly what 1-NFE inference uses.

  - sample_critic_timestep_ltx2: warps a uniform draw with the same
    exponential shift the LTX-2 inference schedule uses (mu from
    dynamic_shift_len=4096, the pipeline default → mu = 2.05).
    CAVEAT: the terminal shift is NOT applied here — it is defined on the
    discrete grid (rescales so the LAST grid point lands at sigma=0.1) and
    has no clean continuous analogue. The sampled density is therefore
    slightly heavier near sigma=0 than the true inference grid. Same level
    of approximation as the Wan version, which also only mirrors the shift
    warp, and we clamp to [min_step, max_step] anyway.
"""
import math
import random
import torch

from diffsynth.diffusion import FlowMatchScheduler


NUM_TRAIN_TIMESTEPS = 1000


# ---------------------------------------------------------------------------
# Flow-matching schedule (identical math to Wan dmd_utils)
# ---------------------------------------------------------------------------
def timestep_to_sigma(t, num_train=NUM_TRAIN_TIMESTEPS):
    """sigma = t / 1000. Same warning as the Wan version applies: timestep
    labels are DEFINED as sigma*1000 (set_timesteps_ltx2 builds them that
    way, model_fn_ltx2 reads sigma = timestep/1000), so do NOT apply any
    shift warping here — that would double-count it."""
    return float(t) / num_train


def add_noise_flow(x0, sigma, noise=None):
    """Flow forward: x_t = sigma * noise + (1 - sigma) * x_0. Returns (x_t, noise)."""
    if noise is None:
        noise = torch.randn_like(x0)
    return sigma * noise + (1 - sigma) * x0, noise


def velocity_to_x0(v_pred, x_t, sigma):
    """Recover x_0 estimate from a flow-matching model's velocity output."""
    return x_t - sigma * v_pred


# ---------------------------------------------------------------------------
# LTX-2 schedule derivation
# ---------------------------------------------------------------------------
def derive_denoising_step_list(num_inference_steps: int, dynamic_shift_len: int = 4096):
    """Student's few-step timestep list, matching inference exactly.

    num_inference_steps=1 → [1000.0]: set_timesteps_ltx2's terminal shift is
    0/0 on a single-point grid (sigma grid = [1.0], one_minus_z[-1] = 0), and
    1-NFE inference denoises from pure noise at t=1000 regardless.
    """
    if num_inference_steps == 1:
        return [1000.0]
    sigmas, timesteps = FlowMatchScheduler.set_timesteps_ltx2(
        num_inference_steps=num_inference_steps,
        dynamic_shift_len=dynamic_shift_len,
    )
    return [float(t) for t in timesteps]


def _ltx2_shift_mu(dynamic_shift_len=4096, base_seq_len=1024, max_seq_len=4096,
                   base_shift=0.95, max_shift=2.05):
    """Replicates FlowMatchScheduler._calculate_shift_qwen_image with the
    LTX-2 parameters from set_timesteps_ltx2. Default len 4096 → mu = 2.05."""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return dynamic_shift_len * m + b


def sample_critic_timestep_ltx2(min_step=20, max_step=980,
                                dynamic_shift_len=4096,
                                num_train=NUM_TRAIN_TIMESTEPS):
    """Random timestep for the critic / DMD score evaluation, density-matched
    to the LTX-2 inference schedule's exponential shift (see module docstring
    for the terminal-shift caveat). Clamped away from the extremes like the
    Wan/Helios version."""
    s = torch.rand(1).item()
    mu = _ltx2_shift_mu(dynamic_shift_len)
    s = math.exp(mu) / (math.exp(mu) + (1 / max(s, 1e-9) - 1))
    t = s * num_train
    return int(min(max(round(t), min_step), max_step))


# ---------------------------------------------------------------------------
# Critic loss / DMD generator loss (identical to Wan dmd_utils, with optional
# spatial mask for LTX-2's in-frame conditioning — frame 0 is replaced by the
# recent latent inside model_fn_ltx2, so it must not contribute loss)
# ---------------------------------------------------------------------------
def compute_critic_loss(v_critic_pred, x_pred, noise, mask=None):
    """Flow-matching velocity MSE; optionally weighted by denoise_mask_video
    ([B,1,f,h,w], 0 at the conditioned frame-0 region when strength=1)."""
    target_velocity = noise - x_pred.detach()
    se = (v_critic_pred.float() - target_velocity.float()) ** 2
    if mask is None:
        return se.mean()
    w = mask.float()
    return (se * w).sum() / (w.sum() * se.shape[1]).clamp(min=1.0)


def compute_dmd_gradient(pred_fake, pred_real, x_pred, normalize=True, eps=1e-5, mask=None):
    """Helios-style DMD gradient. If mask is given, the gradient is zeroed in
    the conditioned (non-denoised) region AND the normalizer is computed over
    the masked region only — otherwise the conditioned frame, where fake and
    real agree by construction, dilutes the normalizer."""
    grad = pred_fake - pred_real
    if mask is not None:
        grad = grad * mask
    if normalize:
        p_real = (x_pred.detach() - pred_real)
        if mask is not None:
            p_real = p_real * mask
            denom = (mask.float().sum() * p_real.shape[1]).clamp(min=1.0)
            normalizer = p_real.abs().float().sum() / denom + eps
        else:
            per_sample_dims = list(range(1, p_real.ndim))
            normalizer = p_real.abs().mean(dim=per_sample_dims, keepdim=True) + eps
        grad = grad / normalizer
    return torch.nan_to_num(grad).detach()


def cfg_real_x0(pred_real_cond, pred_real_uncond, guidance_scale):
    """CFG on the teacher's x0 estimate: cond + (cond - uncond) * scale."""
    if guidance_scale == 0:
        return pred_real_cond
    return pred_real_cond + (pred_real_cond - pred_real_uncond) * guidance_scale


def compute_dmd_loss(x_pred, grad):
    """loss = (x_pred * grad.detach()).mean() — synthetic-gradient trick."""
    return (x_pred * grad).mean()


def masked_l1(a, b, mask=None):
    """L1 for the self-correcting loss, weighted by denoise_mask_video."""
    d = (a.float() - b.float()).abs()
    if mask is None:
        return d.mean()
    w = mask.float()
    return (d * w).sum() / (w.sum() * d.shape[1]).clamp(min=1.0)
