"""Projected GAN discriminator using frozen CLIP image encoder.

Sauer 2021 (NeurIPS) — "Projected GANs Converge Faster". We adapt the idea to
1-step DMD video distillation: the D head operates on CLIP image features
rather than raw pixels, bringing billion-scale visual knowledge into the
adversarial signal.

Multi-layer variant (v2):
  Earlier single-CLS-token version (v1) found CLIP D's logits stuck at
  baseline — final-layer CLS is too semantically compressed for D to
  separate fake from real. Following Sauer, we hook MULTIPLE intermediate
  layers and use one D head per layer, averaging the logits. Lower layers
  retain finer detail differences that the D can latch onto.

Frozen CLIP. Multiple trainable MLP heads, ~few hundred K trainable each.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClipDiscriminator(nn.Module):
    """Projected discriminator: frozen CLIP encoder + per-layer trainable heads.

    Input pixels are in [-1, 1] (Wan VAE decode output range).

    hook_layers: indices into model.outputs.hidden_states (which is
        [embedding, block_0_out, ..., block_N_out]).  For ViT-B/32 (12 blocks),
        valid indices are 0..12; we default to (4, 8, 11, 12) for a mix of
        early / mid / late / final features.
    """

    def __init__(self,
                 clip_model_name: str = "openai/clip-vit-base-patch32",
                 hidden_dim: int = 512,
                 num_layers: int = 2,
                 hook_layers=(4, 8, 11, 12),
                 dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        from transformers import CLIPVisionModel
        self.clip = CLIPVisionModel.from_pretrained(clip_model_name)
        self.clip.eval()
        for p in self.clip.parameters():
            p.requires_grad_(False)
        self.clip.to(dtype=dtype)

        clip_dim = self.clip.config.hidden_size       # 768 for B/32, 1024 for L/14
        n_total_layers = self.clip.config.num_hidden_layers
        # Validate hook_layers (hidden_states is [n_total_layers+1] in HF)
        max_idx = n_total_layers
        hook_layers = tuple(min(max(0, int(l)), max_idx) for l in hook_layers)
        self.hook_layers = hook_layers

        # One D head per hooked layer.
        # KEPT IN FP32: bf16's ~0.008 relative precision is too coarse for
        # AdamW updates of magnitude lr*grad ≈ 1e-4 * 0.015 = 1.5e-6 to
        # accumulate without rounding-to-zero. Heads stay fp32; we cast the
        # input features to float() in forward().
        def make_head():
            layers = []
            in_dim = clip_dim
            for _ in range(num_layers - 1):
                layers += [nn.Linear(in_dim, hidden_dim), nn.GELU()]
                in_dim = hidden_dim
            layers.append(nn.Linear(in_dim, 1))
            return nn.Sequential(*layers)  # fp32 default

        self.heads = nn.ModuleList([make_head() for _ in self.hook_layers])

        # CLIP image preprocessing constants (OpenAI CLIP)
        self.register_buffer('mean', torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
        self.register_buffer('std',  torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))

        self.dtype_setting = dtype

    def _prep(self, pixels: torch.Tensor) -> torch.Tensor:
        x = (pixels + 1.0) * 0.5
        if x.shape[-1] != 224 or x.shape[-2] != 224:
            x = F.interpolate(x.float(), size=224, mode='bilinear', align_corners=False).to(self.dtype_setting)
        x = (x - self.mean.to(x.dtype)) / self.std.to(x.dtype)
        return x

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        """pixels: [N, 3, H, W] in [-1, 1].  Returns [N, 1] averaged logits (bf16)."""
        x = self._prep(pixels)
        outputs = self.clip(pixel_values=x, output_hidden_states=True)
        hiddens = outputs.hidden_states              # tuple of [N, num_tokens, D]

        total_logit = 0.0
        for layer_idx, head in zip(self.hook_layers, self.heads):
            feat_l = hiddens[layer_idx][:, 0].float()   # CLS token, cast to fp32 for head
            total_logit = total_logit + head(feat_l)
        return (total_logit / len(self.hook_layers)).to(self.dtype_setting)


def clip_gan_g_loss(fake_logit: torch.Tensor) -> torch.Tensor:
    """Non-saturating G loss: -log D(fake) form via softplus."""
    return F.softplus(-fake_logit.float()).mean()


def clip_gan_d_loss(real_logit: torch.Tensor, fake_logit: torch.Tensor) -> torch.Tensor:
    """Standard discriminator loss: -log D(real) -log(1-D(fake))."""
    return F.softplus(-real_logit.float()).mean() + F.softplus(fake_logit.float()).mean()
