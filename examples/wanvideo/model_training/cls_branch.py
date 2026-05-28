"""
Discriminator head (cls_branch) for DMD + GAN distillation, adapted from
One-Forcing (Aurora-edu/One-Forcing, utils/wan_wrapper.py adding_cls_branch).

Architecture (attention variant, no conv3d):
    forward of critic = WanModel(...)
      └─ forward_hook on blocks[L] for L in feature_layers (default [13,21,29])
         each captures [B, seq_len, dim] feature tensor
    ↓
    ClsBranch(features_list):
      register_tokens : N learnable [dim] tokens, one per feature layer
      for each (register_i, feature_i):
          cross_attn + FFN  →  [B, 1, dim] aggregated
      concat all  →  [B, N*dim]
      MLP head   →  [B, num_class] logit
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
class RegisterTokens(nn.Module):
    """N learnable register tokens of dimension `dim`, RMS-normed."""
    def __init__(self, num_registers: int, dim: int):
        super().__init__()
        self.tokens = nn.Parameter(torch.randn(num_registers, dim) * 0.02)
        self.norm = nn.RMSNorm(dim, eps=1e-6)

    def forward(self):
        return self.norm(self.tokens)         # [N, dim]


class GanCrossAttnBlock(nn.Module):
    """One register cross-attends to one transformer layer's feature, then FFN.
    Mirrors One-Forcing's GanAttentionBlock but uses standard nn primitives."""
    def __init__(self, dim: int = 1536, ffn_dim: int = 4096, num_heads: int = 12):
        super().__init__()
        self.norm_q  = nn.RMSNorm(dim, eps=1e-6)
        self.norm_kv = nn.RMSNorm(dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_ffn = nn.RMSNorm(dim, eps=1e-6)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )

    def forward(self, register: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        """register: [B, 1, D].  feat: [B, L, D].  → returns [B, 1, D]."""
        q  = self.norm_q(register)
        kv = self.norm_kv(feat)
        attended, _ = self.attn(q, kv, kv, need_weights=False)
        x = register + attended
        x = x + self.ffn(self.norm_ffn(x))
        return x


class ClsBranch(nn.Module):
    """Discriminator head built on top of N intermediate features from a DiT.

    Forward inputs: list of N feature tensors, each [B, L, dim].
    Forward output: [B, num_class] logit.
    """
    def __init__(
        self,
        num_layers: int = 3,
        dim: int = 1536,
        ffn_dim: int = 4096,
        num_heads: int = 12,
        num_class: int = 1,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dim = dim
        self.registers = RegisterTokens(num_layers, dim)
        self.blocks = nn.ModuleList(
            [GanCrossAttnBlock(dim, ffn_dim, num_heads) for _ in range(num_layers)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(dim * num_layers),
            nn.Linear(dim * num_layers, dim),
            nn.SiLU(),
            nn.Linear(dim, num_class),
        )

    def forward(self, feature_list):
        assert len(feature_list) == self.num_layers, (
            f"expected {self.num_layers} features, got {len(feature_list)}"
        )
        B = feature_list[0].shape[0]
        regs = self.registers().unsqueeze(0).expand(B, -1, -1)   # [B, N, D]
        outs = []
        for i, feat in enumerate(feature_list):
            reg_i = regs[:, i : i + 1]                            # [B, 1, D]
            out_i = self.blocks[i](reg_i, feat)                   # [B, 1, D]
            outs.append(out_i)
        concat = torch.cat(outs, dim=1).flatten(1)                # [B, N*D]
        logit = self.head(concat)                                 # [B, num_class]
        return logit


# ---------------------------------------------------------------------------
# Feature hooks: capture outputs of selected blocks during a model forward
# ---------------------------------------------------------------------------
class FeatureCapturer:
    """Context-manager / hook holder.  Use as:

        capturer = FeatureCapturer(critic_dit, layer_indices=[13, 21, 29])
        with capturer:
            _ = critic_dit(...)            # any forward call
            feats = capturer.features()    # list of [B, L, D] in layer order
    """
    def __init__(self, model, layer_indices):
        self.model = model
        self.layer_indices = list(layer_indices)
        self._captured = {}
        self._handles = []

    def __enter__(self):
        self._captured.clear()
        for idx in self.layer_indices:
            block = self.model.blocks[idx]
            def make_hook(i):
                def hook(_module, _input, output):
                    # WanAttentionBlock.forward returns the updated x: [B, L, D]
                    self._captured[i] = output
                return hook
            self._handles.append(block.register_forward_hook(make_hook(idx)))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        return False

    def features(self):
        return [self._captured[i] for i in self.layer_indices]


# ---------------------------------------------------------------------------
# GAN losses (non-saturating softplus, standard DMD2/One-Forcing convention)
# ---------------------------------------------------------------------------
def gan_g_loss(fake_logit: torch.Tensor) -> torch.Tensor:
    """Generator non-saturating loss:  E[ softplus(-D(fake)) ]"""
    return torch.nn.functional.softplus(-fake_logit.float()).mean()


def gan_d_loss(real_logit: torch.Tensor, fake_logit: torch.Tensor) -> torch.Tensor:
    """Discriminator non-saturating loss:
       E[ softplus(-D(real)) + softplus(D(fake)) ]"""
    return (
        torch.nn.functional.softplus(-real_logit.float()).mean()
        + torch.nn.functional.softplus(fake_logit.float()).mean()
    )
