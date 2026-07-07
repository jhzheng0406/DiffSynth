# FD-Loss × TurboTalk — 代码修改笔记

> 用途：对照现有 1-step distillation 代码，检查旧 FD loss 实现并评估改造方案。
> 项目：Fast Bidirectional 1-Step Video Distillation with Error Recycle (Wan2.1-Fun-Control-1.3B, DMD + GAN, sink/recent ref, double recycle)

---

## 1. TurboTalk (arXiv 2604.14580)

**任务**：音频驱动数字人 1-NFE 蒸馏（InfiniteTalk / Wan 14B 级别）。

**论点**：
- 直接 DMD 蒸 1-step → student/teacher 分布差距过大，训练不稳；
- 直接对抗蒸 1-step → 真假差距过大，discriminator 过早收敛，梯度消失。

**方案：两阶段渐进蒸馏**
1. **Stage 1**：标准 DMD 蒸到 4-step student（稳定中间态）。
2. **Stage 2**：对抗蒸馏 4→3→2→1 逐步减，配三个稳定性组件：
   - **只对最后一步回传梯度**：N 步 rollout 中前面步骤全部 stop_grad，省显存 + 控制相邻 phase 质量差距；
   - **Dynamic Timestep Sampling**：phase 切换 warm-up 期，最后一个 timestep 在旧/新 phase 之间随机采样（timestep gap 上的 curriculum）；
   - **Self-Compare Regularization**：discriminator 保留 4-step 去噪能力，产生介于真实数据和 student 之间的中间参考样本，relativistic 对齐：
     `L = -log σ(D(x̂_N) − D(x̂_4step))`
     λ ablation 呈 U 形——太强会把 student 锚死在中间参考上。

**GAN 稳定配置（14B 级别验证过）**：
- R3GAN（relativistic）损失——D 无法对所有 pair 同时饱和，梯度始终存在；ablation 中优于 vanilla 和 hinge；
- R1 + R2 梯度惩罚（噪声扰动近似实现），并扩展到 self-compare 样本（R3）；
- D 从 few-step generator 初始化，**分类头 zero-init**（APT trick）——训练初期对抗梯度从 0 平滑增长；
- Generator lr 极小（~4e-7 量级）；
- Ablation 贡献排序：step reduction > self-compare > dynamic TS。

**与本项目的关系**：
- ✅ Self-compare 的 relativistic 比较 = refinement anchor 的替代形式（分布层面，不要求样本对齐，绕开 paired L1 的平均化模糊）；
- ✅ GAN 崩溃（14B、loss 单边跑偏 = D 压死 G）的对症 recipe，改动最小的两项：**换 R3GAN loss + 判别头 zero-init**；
- ✅ 无 error recycle / drift correction（靠 50% 迭代混入 9 帧 context frames），任务域不同 → 对 novelty 无威胁，可作为对抗蒸馏路线的引用对照；
- ⚠️ 训练成本高（对抗阶段 32×H800 × 3000 步）→ 可用于论证 One-Forcing 式 DMD+GAN 联合训练不需要渐进 phase 的成本优势。

---

## 2. FD-Loss (arXiv 2604.28190, Yang et al., USC/CMU)

**核心 trick**：把 FD 统计估计的 **population size**（~50k 样本）和梯度计算的 **batch size**（~1024）解耦——μ、Σ 在大样本池上维护，梯度只流经当前 batch。

**三个发现**：
1. FD-loss post-train 在多种 representation space 下一致提升视觉质量（Inception 空间 ImageNet 256 → 0.72 FID）；
2. 同一 loss 可把多步生成器直接改成强单步生成器——**无 teacher 蒸馏、无对抗训练、无 per-sample target**；
3. FID 可能误判视觉质量：现代表示空间（如 DINOv2）能产出更好样本，即使 Inception FID 更差 → 别用 Inception。

**为什么和旧 FD loss 本质不同**（待对代码确认属于哪种）：

| 旧实现可能形态 | 失败机制 |
|---|---|
| Paired feature 距离（同 noise 下 student vs 参考逐样本回归） | Sample-to-sample 目标是 mode-covering 的；结构错位时期望最优解 = 所有参考的平均 → 糊。Feature 空间只是把糊推迟到语义层面 |
| Batch 内直接算 Fréchet | Batch 2–8 的协方差估计是纯噪声，loss 本身退化 |

解耦后的 FD 是**真·分布级目标**：匹配 feature 分布的一阶、二阶矩，无任何样本配对 → 不存在 misalignment → 平均化 → 模糊的通路。

**vs GAN**：提供同类的分布锐化信号，但目标是解析的矩匹配，没有 discriminator → 没有 D/G 失衡可崩。正好是 14B GAN 崩溃的替代出路。

**vs FD loss（paired）vs 对抗比较，一句话**：
- Paired FD/L1 = sample-to-sample 距离（mode-covering，被迫平均）；
- 对抗 loss = sample-to-distribution 方向（reverse-KL，mode-seeking，可选一个 mode 贴）；
- 解耦 FD-loss = distribution-to-distribution 矩匹配（无配对，无对抗不稳定）。

---

## 3. 改造方案

**目标形态**：`FD(student 1-step 输出分布, teacher K-step 输出分布)` in frozen representation space，替换 paired anchor。
**分工**：DMD 管低频结构（reverse-KL），FD-loss 管锐度/细节统计（矩匹配），信号性质不同不打架。

### 实施清单

- [ ] **Teacher 统计离线化**：teacher 冻结 → 离线跑一批 rollout（可复用 offline ODE backfill 产物）→ 抽 feature → 一次性算 μ_T、Σ_T 存盘，训练时为常数。
- [ ] **Student 侧解耦**：feature bank（最近 M 个 batch 的特征，detach 入队）+ 当前 batch（带梯度），FD 在 bank ∪ current 上算。
  - 检查：梯度是否只流经 current；bank 大小是否支撑协方差估计。
- [ ] **样本数放大**：每帧 / 每 temporal patch 的 feature 当独立样本（49 帧 × batch → 有效样本数 ×1–2 个数量级）；维度高先用对角 Σ 近似。
- [ ] **表示空间选择**（便宜 → 贵）：
  1. Teacher DiT 中间层 feature（免解码，但与 DMD 信号来源重叠）；
  2. **VAE 解码抽 4–8 帧过 DINOv2（推荐起点）**；
  3. V-JEPA / InternVideo（时序统计一起匹配，最贵）。
- [ ] **参考实现**：`github.com/Jiawei-Yang/FD-loss`（PyTorch，重点看 bank 维护和解耦梯度部分）。

### Caveats

1. 视频上无人验证过（论文只做了 ImageNet 图像）→ 风险即卖点：video 蒸馏 loop 里第一个用 FD-loss 的工作可作为贡献点；
2. 矩匹配只约束前两阶矩，理论上可在高阶统计上作弊 → 靠 DMD 兜底。

---

## 4. 待确认 / 代码定位

**三个待确认问题**：
1. 旧 FD loss 属于 §2 表格哪种？（决定论文里"为何之前失败、现在能行"的叙事线）
2. GAN 崩溃时 D 的结构和初始化来源？（决定 zero-init 和 R1 加在哪）
3. FD-loss 路线走通后，GAN 分支降级为 ablation 还是彻底移除？

**代码先定位三处**：
1. FD loss 的具体计算方式（配对 or 批内统计）；
2. Feature 从哪个模型 / 哪层抽；
3. Loss 权重与 DMD loss 的相对量级（以及有无 warm-up）。

---

## 参考

- TurboTalk: https://arxiv.org/abs/2604.14580
- FD-Loss: https://arxiv.org/abs/2604.28190 · code: https://github.com/Jiawei-Yang/FD-loss
- 相关：One-Forcing（DMD+GAN 1-step baseline）、ASD（n vs n+1 step 对齐，self-compare 同源思路）、APT（zero-init 判别头）、R3GAN
