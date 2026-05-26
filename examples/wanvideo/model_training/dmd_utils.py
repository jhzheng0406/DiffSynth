"""
DMD2 distillation math (flow-matching variant for Wan).

Schedule (Wan rectified-flow):
    x_t = sigma * noise + (1 - sigma) * x_0     where sigma in [0, 1]
    model predicts velocity v ≈ noise - x_0
    inverse:   x_0 = x_t - sigma * v

References:
- Helios `_generator_loss` / `_critic_loss` (utils_helios_post.py:2141, :2788)
- DMD2 (Yin et al. 2023)
- Causal Forcing (2602.02214) — asymmetric DMD section
"""
import random
import torch


NUM_TRAIN_TIMESTEPS = 1000
# Fallback only. train_dmd.py derives the real list from the scheduler so it
# matches inference exactly (shift=5, 4 steps → ~[1000, 937, 833, 625]).
DEFAULT_DENOISING_STEPS = [1000, 937, 833, 625]


# ---------------------------------------------------------------------------
# Flow-matching schedule
# ---------------------------------------------------------------------------
def timestep_to_sigma(t, num_train=NUM_TRAIN_TIMESTEPS):
    """Noise level for a Wan timestep label is simply sigma = t / num_train.

    IMPORTANT: do NOT apply the flow `shift` here. In Wan's FlowMatchScheduler,
    timesteps are DEFINED as `timesteps = sigmas * 1000` after the shift is
    applied once when building the discrete schedule (set_timesteps_wan). So for
    any timestep label, add_noise reads sigma = timestep / 1000. Applying shift
    again in this conversion double-counts it and desyncs from the frozen
    teacher, which interprets sigma = timestep/1000."""
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
# Timestep sampling
# ---------------------------------------------------------------------------
def sample_generator_timestep(denoising_step_list=DEFAULT_DENOISING_STEPS):
    """Pick the timestep at which student does its 1-step denoising training pass."""
    return random.choice(denoising_step_list)


def sample_critic_timestep(min_step=20, max_step=980, shift=5.0,
                           num_train=NUM_TRAIN_TIMESTEPS):
    """Random timestep for the critic / DMD score evaluation.

    Mirrors Helios sample_dynamic_timestep (utils_helios_post.py:510): draw
    uniform in raw timestep space, apply the same flow `shift` warping the
    inference schedule uses (so sampled noise levels follow the inference
    density), then clamp to [min_step, max_step]. Note max_step=980 (not 1000)
    — never evaluate at the pure-noise extreme.
    """
    t = torch.rand(1).item() * num_train          # uniform raw timestep
    if shift > 1:
        s = t / num_train
        t = shift * s / (1 + (shift - 1) * s) * num_train
    return int(min(max(round(t), min_step), max_step))


# ---------------------------------------------------------------------------
# Critic loss: critic learns to denoise student's outputs
# ---------------------------------------------------------------------------
def compute_critic_loss(v_critic_pred, x_pred, noise):
    """Standard flow-matching velocity MSE.

    Args:
        v_critic_pred: critic's velocity prediction at (x_pred_noisy, t_critic)
        x_pred:        student's clean estimate (DETACHED — no grad to student)
        noise:         the random noise used to make x_pred_noisy

    Target velocity = noise - x_pred (flow-matching definition).
    """
    target_velocity = noise - x_pred.detach()
    return torch.nn.functional.mse_loss(
        v_critic_pred.float(), target_velocity.float()
    )


# ---------------------------------------------------------------------------
# DMD generator loss: synthetic gradient via (pred_fake - pred_real)
# ---------------------------------------------------------------------------
def compute_dmd_gradient(pred_fake, pred_real, x_pred, normalize=True, eps=1e-5):
    """Helios-style DMD gradient (compute_kl_grad, post.py:1826-1834).

    pred_fake: x0 estimate from critic (fake score), no_grad.
    pred_real: x0 estimate from teacher (real score), CFG-enhanced by the
               caller, no_grad. See cfg_real_x0().
    x_pred:    student's clean estimate (carries gradient back to student).

    Returns: detached grad tensor for the synthetic-gradient trick.
    """
    # Direction student is off by: pred_fake - pred_real.
    # Update -grad pushes student output FROM pred_fake TOWARDS pred_real.
    grad = pred_fake - pred_real

    if normalize:
        # Normalize by teacher's "how-off-are-we" magnitude (DMD eq.8).
        p_real = x_pred.detach() - pred_real
        per_sample_dims = list(range(1, p_real.ndim))
        normalizer = p_real.abs().mean(dim=per_sample_dims, keepdim=True) + eps
        grad = grad / normalizer

    return torch.nan_to_num(grad).detach()


def cfg_real_x0(pred_real_cond, pred_real_uncond, guidance_scale):
    """Classifier-free guidance on the teacher's x0 estimate (Helios :1783).

        pred_real = cond + (cond - uncond) * guidance_scale

    guidance_scale=3.0 is the Helios default. This sharpens the real
    distribution the student matches toward — WITHOUT it the student learns the
    unguided (blurry, low-saturation) teacher distribution.
    """
    if guidance_scale == 0:
        return pred_real_cond
    return pred_real_cond + (pred_real_cond - pred_real_uncond) * guidance_scale


def compute_dmd_loss(x_pred, grad):
    """Convert the synthetic gradient into a scalar loss usable by backward().

        loss = (x_pred * grad.detach()).mean()
        => ∂loss/∂x_pred = grad
        => ∂loss/∂θ_student = grad · ∂x_pred/∂θ_student
        => Adam step direction = -∂loss/∂θ
           = (pred_real - pred_fake) direction  (after the .mean() factor)
        => student output moves TOWARDS pred_real, AWAY from pred_fake. ✓
    """
    return (x_pred * grad).mean()
