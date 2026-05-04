"""
Diagnose whether the optimizer in HeliosWanTrainingModule actually picks up
the Helios parameters (patch_helios_*, history_key_scale).

We don't need full data — just construct the training module exactly the way
the trainer does, then enumerate model.trainable_modules() and check.

Run from DiffSynth-Studio repo root:
    python test_optimizer_sees_helios.py
"""
import sys
import os

sys.path.insert(0, "/mnt/vita/scratch/vita-students/users/jinghao/code/DiffSynth-Studio")
sys.path.insert(0, "/mnt/vita/scratch/vita-students/users/jinghao/code/DiffSynth-Studio/examples/wanvideo/model_training")

import torch
from train_helios import HeliosWanTrainingModule

VIDEOXFUN_ROOT = "/mnt/vita/scratch/vita-students/users/jinghao/code/VideoX-Fun"
MODEL_DIR = f"{VIDEOXFUN_ROOT}/models/Diffusion_Transformer/Wan2.1-Fun-V1.1-1.3B-Control"
DIT_CKPT = f"{MODEL_DIR}/diffusion_pytorch_model.safetensors"

print("Building HeliosWanTrainingModule (this loads the base Wan checkpoint)...")
model = HeliosWanTrainingModule(
    helios_history_sizes=[4, 2, 1],
    helios_init_scale_logit=-3.0,
    helios_corrupt_mode="random",
    helios_noise_ratio_short=1.0/3.0,
    helios_noise_ratio_mid=1.0/3.0,
    helios_noise_ratio_long=1.0/3.0,
    helios_keep_x0=False,
    helios_apply_saturation=True,
    helios_saturation_min=0.3,
    helios_saturation_max=1.7,
    helios_clean_history_prob=0.1,
    helios_drop_t2v_ratio=0.10,
    helios_drop_i2v_ratio=0.05,
    helios_supervise_history=False,
    helios_use_first_frame_anchor=True,
    model_paths=f'["{DIT_CKPT}","{MODEL_DIR}/models_t5_umt5-xxl-enc-bf16.pth","{MODEL_DIR}/Wan2.1_VAE.pth","{MODEL_DIR}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"]',
    model_id_with_origin_paths=None,
    tokenizer_path=f"{MODEL_DIR}/google/umt5-xxl",
    audio_processor_path=None,
    trainable_models="dit",
    lora_base_model=None,
    lora_target_modules="",
    lora_rank=128,
    lora_checkpoint=None,
    preset_lora_path=None,
    preset_lora_model=None,
    extra_inputs="control_video",
    use_gradient_checkpointing=True,
    use_gradient_checkpointing_offload=False,
    fp8_models="",
    task="sft",
    max_timestep_boundary=1.0,
    min_timestep_boundary=0.0,
)

print("\n" + "=" * 60)
print("Helios-related params in named_parameters():")
print("=" * 60)
all_helios = [(n, p) for n, p in model.named_parameters()
              if 'patch_helios' in n or 'history_key_scale' in n]
print(f"Total found: {len(all_helios)}")
n_grad_on  = sum(1 for _, p in all_helios if p.requires_grad)
n_grad_off = sum(1 for _, p in all_helios if not p.requires_grad)
print(f"  requires_grad=True:  {n_grad_on}")
print(f"  requires_grad=False: {n_grad_off}")
for n, p in all_helios[:8]:
    flag = "✓ trainable" if p.requires_grad else "✗ FROZEN"
    print(f"  {flag}  {n:65s} dtype={p.dtype}  device={p.device}")
if len(all_helios) > 8:
    print(f"  ... and {len(all_helios) - 8} more")

print("\n" + "=" * 60)
print("Would optimizer (AdamW from model.trainable_modules()) see them?")
print("=" * 60)
trainable_list = list(model.trainable_modules())
print(f"Total trainable params (filtered by requires_grad): {len(trainable_list)}")
trainable_ids = set(id(p) for p in trainable_list)
helios_in_optimizer = [(n, p) for n, p in all_helios if id(p) in trainable_ids]
helios_missed_by_optimizer = [(n, p) for n, p in all_helios if id(p) not in trainable_ids]
print(f"  Helios params optimizer WILL see:    {len(helios_in_optimizer)}")
print(f"  Helios params optimizer WILL MISS:   {len(helios_missed_by_optimizer)}")
if helios_missed_by_optimizer:
    print("\n  ⚠️  MISSED:")
    for n, p in helios_missed_by_optimizer[:5]:
        print(f"    {n}  requires_grad={p.requires_grad}")

print("\n" + "=" * 60)
print("Sanity: total trainable params count (should be ~most of dit):")
print("=" * 60)
print(f"  {len(trainable_list)} param tensors are in optimizer scope")

# Verdict
if helios_missed_by_optimizer:
    print("\n💀 BUG CONFIRMED: optimizer misses some Helios params (see ⚠️ above)")
elif n_grad_off == len(all_helios):
    print("\n💀 BUG CONFIRMED: ALL helios params have requires_grad=False (someone froze them)")
elif len(all_helios) == 0:
    print("\n💀 BUG CONFIRMED: no Helios params found at all (patch failed?)")
else:
    print(f"\n🤔 Helios params look reachable to the optimizer.")
    print(f"   If they still aren't being updated in v6 ckpt, the bug is elsewhere:")
    print(f"   maybe gradient never flows (DDP find_unused_parameters interaction)")
    print(f"   or the parameters are detached during forward, or model.to() somewhere casts")
    print(f"   them to a dtype where AdamW updates round to 0.")
