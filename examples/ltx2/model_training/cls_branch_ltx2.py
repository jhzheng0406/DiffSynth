"""
LTX-2 port of the One-Forcing discriminator head (cls_branch) used by the
Wan DMD students — see examples/wanvideo/model_training/cls_branch.py.
Kept local so the ltx2 example tree has no import edge into wanvideo/.

Differences vs the Wan version:
  - Defaults sized for LTX-2 19B: dim=4096 (inner_dim = 32 heads × 128),
    num_heads=32 (4096 is not divisible by the Wan default 12).
  - FeatureCapturer hooks `model.transformer_blocks[idx]` and the block
    returns a (video: TransformerArgs, audio: TransformerArgs|None) tuple,
    so the hook captures `output[0].x` — the video token stream
    [B, seq, 4096]. The sequence includes the appended reference tokens
    (sink); same as Wan, where FunReference tokens were also in-stream.
  - Gradient flow under checkpointing: DiffSynth's gradient_checkpoint_forward
    uses torch.utils.checkpoint with use_reentrant=False, which builds the
    autograd graph during the forward pass (only the saved activations are
    dropped/recomputed). Captured features therefore stay connected to the
    graph — same mechanism the Wan GAN steps rely on. Hooks fire again
    during backward recomputation and overwrite _captured; harmless because
    features are consumed before backward and hooks are removed at __exit__.
"""

import torch
import torch.nn as nn


class RegisterTokens(nn.Module):
    """N learnable register tokens of dimension `dim`, RMS-normed."""
    def __init__(self, num_registers: int, dim: int):
        super().__init__()
        self.tokens = nn.Parameter(torch.randn(num_registers, dim) * 0.02)
        self.norm = nn.RMSNorm(dim, eps=1e-6)

    def forward(self):
        return self.norm(self.tokens)         # [N, dim]


class GanCrossAttnBlock(nn.Module):
    """One register cross-attends to one transformer layer's feature, then FFN."""
    def __init__(self, dim: int = 4096, ffn_dim: int = 4096, num_heads: int = 32):
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
    """Discriminator head on N intermediate features from the LTX-2 DiT.

    Forward inputs: list of N feature tensors, each [B, L, dim].
    Forward output: [B, num_class] logit.
    """
    def __init__(
        self,
        num_layers: int = 3,
        dim: int = 4096,
        ffn_dim: int = 4096,
        num_heads: int = 32,
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


class FeatureCapturer:
    """Hook holder for LTXModel.transformer_blocks.

        with FeatureCapturer(dit, [21, 34, 47]) as cap:
            _ = model_fn_ltx2(dit=dit, ...)
            feats = cap.features()    # list of [B, L, 4096] in layer order
    """
    def __init__(self, model, layer_indices):
        self.model = model
        self.layer_indices = list(layer_indices)
        self._captured = {}
        self._handles = []

    def __enter__(self):
        self._captured.clear()
        for idx in self.layer_indices:
            block = self.model.transformer_blocks[idx]
            def make_hook(i):
                def hook(_module, _input, output):
                    # BasicAVTransformerBlock returns (video_args, audio_args);
                    # video_args.x is the [B, L, D] video token stream.
                    self._captured[i] = output[0].x
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


def gan_g_loss(fake_logit: torch.Tensor) -> torch.Tensor:
    """Generator non-saturating loss:  E[ softplus(-D(fake)) ]"""
    return torch.nn.functional.softplus(-fake_logit.float()).mean()


def gan_d_loss(real_logit: torch.Tensor, fake_logit: torch.Tensor) -> torch.Tensor:
    """Discriminator loss: E[ softplus(-D(real)) + softplus(D(fake)) ]"""
    return (
        torch.nn.functional.softplus(-real_logit.float()).mean()
        + torch.nn.functional.softplus(fake_logit.float()).mean()
    )
