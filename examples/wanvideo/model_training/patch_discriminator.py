"""Multi-scale PatchGAN discriminator for video distillation.

Discriminates real vs fake pixels (post-VAE-decode) at K spatial scales
via average-pooling-2x between scales (Pix2PixHD style).

  D_s(x_s) → [N, 1, h_s, w_s]   patch logits at scale s

Hinge loss G:  g = -mean_s E[D_s(fake_s)]
Hinge loss D:  d =  mean_s ( E[relu(1 - D_s(real_s))] + E[relu(1 + D_s(fake_s))] )

Per-frame 2D PatchD (treats video as independent frames).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchD(nn.Module):
    """Single-scale PatchGAN discriminator (Pix2Pix-style)."""

    def __init__(self, in_ch: int = 3, base_ch: int = 64, num_layers: int = 3):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, base_ch, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        in_c = base_ch
        for i in range(num_layers - 1):
            out_c = min(in_c * 2, 512)
            layers += [
                nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(out_c, affine=False),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            in_c = out_c
        # final stride-1 conv block before 1-channel output
        out_c = min(in_c * 2, 512)
        layers += [
            nn.Conv2d(in_c, out_c, kernel_size=4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(out_c, affine=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_c, 1, kernel_size=4, stride=1, padding=1),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MultiScalePatchD(nn.Module):
    """Multi-scale wrapper: same-arch PatchD at K progressively-pooled scales.

    `skip_full_res=True` pools once BEFORE the first D, so scales become
    {1/2×, 1/4×, ...} instead of {1×, 1/2×, ...}.  Useful when full-res
    texture is already covered by another discriminator (e.g. cls_branch) and
    you want the multi-scale path to focus on mid-/low-frequency structure
    (body parts, layout) where each D's effective receptive field covers a
    larger image region.
    """

    def __init__(self, num_scales: int = 3, in_ch: int = 3,
                 base_ch: int = 64, num_layers: int = 3,
                 skip_full_res: bool = False):
        super().__init__()
        self.num_scales = num_scales
        self.skip_full_res = skip_full_res
        self.D = nn.ModuleList([
            PatchD(in_ch, base_ch, num_layers) for _ in range(num_scales)
        ])

    def forward(self, x):
        """x: [N, C, H, W] in [-1, 1]. Returns list of K patch-logit maps."""
        outs = []
        cur = x
        if self.skip_full_res:
            cur = F.avg_pool2d(cur, kernel_size=3, stride=2, padding=1)
        for D in self.D:
            outs.append(D(cur))
            # avg-pool 3x3 stride 2 (matches Pix2PixHD downsampling style)
            cur = F.avg_pool2d(cur, kernel_size=3, stride=2, padding=1)
        return outs


def msgan_g_loss(fake_logits_list, loss_type: str = "hinge"):
    """G loss averaged across scales.

    loss_type:
      - "hinge"   : -mean(D(fake))  (saturates outside [-1, 1] margin)
      - "softplus": mean(softplus(-D(fake))) (always non-zero gradient, baseline log(2))
    """
    if loss_type == "softplus":
        return sum(F.softplus(-fl.float()).mean() for fl in fake_logits_list) / len(fake_logits_list)
    return -sum(fl.float().mean() for fl in fake_logits_list) / len(fake_logits_list)


def msgan_d_loss(fake_logits_list, real_logits_list, loss_type: str = "hinge"):
    """D loss averaged across scales.

    loss_type:
      - "hinge"   : relu(1 - real) + relu(1 + fake) — gradient cuts off at margin
      - "softplus": softplus(-real) + softplus(fake) — never saturates
    """
    assert len(fake_logits_list) == len(real_logits_list)
    n = len(fake_logits_list)
    loss = fake_logits_list[0].new_zeros((), dtype=torch.float32)
    if loss_type == "softplus":
        for fl, rl in zip(fake_logits_list, real_logits_list):
            loss = loss + F.softplus(-rl.float()).mean() + F.softplus(fl.float()).mean()
    else:
        for fl, rl in zip(fake_logits_list, real_logits_list):
            loss = loss + F.relu(1.0 - rl.float()).mean() + F.relu(1.0 + fl.float()).mean()
    return loss / n
