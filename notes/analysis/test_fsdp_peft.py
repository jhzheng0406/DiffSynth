"""De-risk FSDP + PEFT(2 named adapters)+ adapter-switching BEFORE touching the
14B run. This is the exact risky interaction the shared-base DMD loop needs under
FSDP: one frozen base, two LoRA adapters (student/critic), switch active adapter
per forward, FULL_SHARD across GPUs, manual backward (no DDP all_reduce).

Run on 2 GPUs (seconds):
  cd DiffSynth-Studio
  torchrun --nproc_per_node=2 notes/analysis/test_fsdp_peft.py

Want to see (all must hold):
  - FSDP shards params (per-rank flat shard < full param count)
  - set_adapter('student') → backward routes grad ONLY to student lora params
  - set_adapter('critic')  → grad ONLY to critic lora params
  - teacher mode (adapters disabled) → forward runs, no lora grad
  - no crash from set_adapter / requires_grad under FSDP
If this is green, FSDP+PEFT mechanics are sound → wire --fsdp into training.
The key knob is use_orig_params=True (lets a flat param mix frozen base +
trainable LoRA, which PEFT needs).
"""
import os, torch, torch.nn as nn, torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from functools import partial
from peft import LoraConfig, inject_adapter_in_model


class Base(nn.Module):
    def __init__(self, dim=256, n=4):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n)])
    def forward(self, x):
        for b in self.blocks:
            x = torch.relu(b(x))
        return x


def use_adapter(model, name):
    """EXACT mirror of train_dmd_recycle.use_adapter (BaseTunerLayer +
    enable_adapters), so the test validates the real switching path."""
    from peft.tuners.tuners_utils import BaseTunerLayer
    for m in model.modules():
        if isinstance(m, BaseTunerLayer):
            if name is None:
                m.enable_adapters(False)
            else:
                m.enable_adapters(True)
                m.set_adapter(name)
    if name is not None:
        for n, p in model.named_parameters():
            if "lora_" in n:
                p.requires_grad_(True)


def grad_norm(model, tag):
    g = {"student": 0.0, "critic": 0.0}
    for n, p in model.named_parameters():
        if p.grad is not None and "lora_" in n:
            which = "student" if ".student." in n else ("critic" if ".critic." in n else "?")
            if which in g:
                g[which] += p.grad.float().norm().item()
    return g


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    is_main = rank == 0

    m = Base().to(dev)
    for p in m.parameters():
        p.requires_grad_(False)                                   # frozen base
    cfg = LoraConfig(r=8, lora_alpha=8,
                     target_modules=[f"blocks.{i}" for i in range(4)])
    inject_adapter_in_model(cfg, m, adapter_name="student")
    inject_adapter_in_model(cfg, m, adapter_name="critic")
    for n, p in m.named_parameters():
        if "lora_" in n:
            p.requires_grad_(True)
    full_lora = sum(p.numel() for n, p in m.named_parameters() if "lora_" in n)

    fm = FSDP(
        m,
        use_orig_params=True,                                     # PEFT enabler
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=rank,                                           # wrap whole model as one unit
    )
    if is_main:
        print(f"[ok] FSDP wrapped. full lora params={full_lora}")

    # student / critic: real grad-bearing forward+backward
    for role, name in [("student", "student"), ("critic", "critic")]:
        use_adapter(fm, name)
        x = torch.randn(2, 256, device=dev)
        y = fm(x)
        loss = (y ** 2).mean()
        if is_main:
            print(f"[{role:7s}] loss.requires_grad={loss.requires_grad}")
        fm.zero_grad(set_to_none=True)
        loss.backward()
        gn = grad_norm(fm, role)
        if is_main:
            print(f"[{role:7s}] grad(student)={gn['student']:.4f}  grad(critic)={gn['critic']:.4f}")
            assert gn[role] > 0, f"{role} adapter got NO grad — FSDP+PEFT grad path broken"
            other = "critic" if role == "student" else "student"
            assert gn[other] == 0, f"grad leaked to inactive {other} adapter"

    # teacher: adapters off, forward only (real loop runs this under no_grad)
    use_adapter(fm, None)
    with torch.no_grad():
        yt = fm(torch.randn(2, 256, device=dev))
    if is_main:
        print(f"[teacher] forward OK (adapters disabled), out norm={yt.float().norm().item():.2f}")

    if is_main:
        print("\nALL FSDP+PEFT MECHANICS OK ✅  → safe to wire --fsdp into training")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
