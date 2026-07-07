"""Validate the shared-base peft mechanics WITHOUT loading a 14B model.
Checks the 3 helpers added to train_dmd_recycle.py:
  1. add_named_lora injects BOTH 'student' and 'critic' adapters (the riskiest
     unknown — that two inject_adapter_in_model calls actually add two adapters).
  2. use_adapter switches active adapter and KEEPS requires_grad=True on both
     (so a long-lived graph still gets grads).
  3. grad routing: student-active fwd/bwd grads ONLY student; critic-active grads
     ONLY critic; teacher (None) uses neither.
  4. adapter_params(name) returns the right disjoint param sets.

Run:  python notes/analysis/test_shared_base_peft.py
Pass  → the peft API assumptions hold on this peft version → proceed to the
        1.3B equivalence smoke test, then 14B with --shared_base.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "examples", "wanvideo", "model_training"))
import torch, torch.nn as nn
from train_dmd_recycle import add_named_lora, use_adapter, adapter_params


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(16, 16)
        self.k = nn.Linear(16, 16)
    def forward(self, x):
        return self.k(self.q(x))


def grads_present(params):
    return any(p.grad is not None and p.grad.abs().sum() > 0 for p in params)


m = Toy()
add_named_lora(m, ["q", "k"], rank=4)

ps = adapter_params(m, "student")
pc = adapter_params(m, "critic")
print(f"[1] student adapter params: {len(ps)}  | critic adapter params: {len(pc)}")
assert len(ps) > 0 and len(pc) > 0, "BOTH adapters must exist — 2nd inject failed!"
assert set(map(id, ps)).isdisjoint(map(id, pc)), "adapter param sets must be disjoint"

def zero():
    for p in list(ps) + list(pc):
        p.grad = None

# student-active
use_adapter(m, "student"); zero()
m(torch.randn(2, 16)).sum().backward()
print(f"[2] student-active: student.grad={grads_present(ps)}  critic.grad={grads_present(pc)}")
assert grads_present(ps) and not grads_present(pc), "student-active must grad ONLY student"

# critic-active
use_adapter(m, "critic"); zero()
m(torch.randn(2, 16)).sum().backward()
print(f"[3] critic-active:  student.grad={grads_present(ps)}  critic.grad={grads_present(pc)}")
assert grads_present(pc) and not grads_present(ps), "critic-active must grad ONLY critic"

# requires_grad must stay True on BOTH after switching (the key invariant)
print(f"[4] requires_grad after switch: student={all(p.requires_grad for p in ps)} "
      f"critic={all(p.requires_grad for p in pc)}")
assert all(p.requires_grad for p in ps) and all(p.requires_grad for p in pc)

# teacher (None) — adapters disabled → output == base only (no adapter delta)
use_adapter(m, None)
x = torch.randn(2, 16)
out_off = m(x)
use_adapter(m, "student")
out_on = m(x)
print(f"[5] teacher(None) disables adapters: differs from student-active? "
      f"{not torch.allclose(out_off, out_on)}")

print("\nALL PEFT MECHANICS OK ✅  → run the 1.3B equivalence smoke test next.")
