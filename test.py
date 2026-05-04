import torch, sys
sys.path.insert(0, "/mnt/vita/scratch/vita-students/users/jinghao/code/DiffSynth-Studio")
from diffsynth.models.wan_video_helios_attention import (
    corrupt_history_latents, add_saturation_to_history_latents
)

torch.manual_seed(0)
B, C, H, W = 1, 16, 32, 32
lat_long  = torch.randn(B, C, 16, H, W)
lat_mid   = torch.randn(B, C, 2,  H, W)
lat_short = torch.randn(B, C, 1,  H, W)

# 重要：测试新的 1/3 ratio 下，凸组合的方差应该接近 1.0（不会膨胀）
out_s, out_m, out_l = corrupt_history_latents(
    lat_short.clone(), lat_mid.clone(), lat_long.clone(),
    noise_ratio_short=1/3, noise_ratio_mid=1/3, noise_ratio_long=1/3,
    corrupt_mode="noise", is_keep_x0=False,
)

print(f"long  before std={lat_long.std():.4f}, after std={out_l.std():.4f}")
print(f"mid   before std={lat_mid.std():.4f},  after std={out_m.std():.4f}")
print(f"short before std={lat_short.std():.4f}, after std={out_s.std():.4f}")
print(f"long delta L2: {(out_l - lat_long).norm():.2f}  (应该 > 0，说明真的扰了)")
print(f"long != short corruption: {not torch.allclose(out_l, out_s.expand_as(out_l))}  (应 True)")
