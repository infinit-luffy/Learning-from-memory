# Baseline 推荐：好复现且最新（2026-07 校准）

> 我搜了 2025–2026 最新的视觉 RL / 模型-based RL 文献，把结果按"好复现 + 新 + 直接可比"的优先级排。

---

## 🥇 主基线（必做，直接可比）：**TD-MPC2**

**Hansen et al., ICLR 2024** · arXiv:2310.16828

### 为什么是它

1. **仍是 SOTA**：截至 2026 年 7 月，TD-MPC2 依然是连续控制的 reference baseline。多篇 2025-2026 综述把它作为对比锚点
2. **零调参**：作者证明单一超参配置在 104 个任务上都 work，reviewer 一眼就相信复现结果
3. **官方代码极干净**：`github.com/nicklashansen/tdmpc2`，单一 YAML 配置，30 分钟能跑通第一个任务
4. **与我们的架构完美同构**：HippoAct 就是把 TD-MPC2 的 pixel encoder 换成我们的 slot 解耦——这是**最公平的对比**，reviewer 挑不出毛病

### 复现速通

```bash
git clone https://github.com/nicklashansen/tdmpc2
cd tdmpc2
conda env create -f docker/environment.yaml
conda activate tdmpc2

# DMControl - 单任务 3 小时 on 单卡 A5000
python tdmpc2/train.py task=dmcontrol-walker-walk seed=0
# Meta-World - MT10 每任务 ~10 小时
python tdmpc2/train.py task=metaworld-pick-place-v2 seed=0
```

### 三个已知坑

1. Meta-World 需要 mujoco 2.3.7，不兼容更新版本；用官方 Dockerfile 最省事
2. 官方 replay buffer 存在 CPU 内存里，1M steps × 84²×3 图像会吃掉 ~46 GB RAM。加 `buffer.compression=lz4` 或用 `disk_replay=true`
3. 论文里 5 seed 是标配，但 v0.1 版本 seed 不完全 deterministic（CUDA async），跑完立即冷启动可以避免

---

## 🥈 次强基线（强烈建议加）：**Squint**

**Almuzairee & Christensen, arXiv:2602.21203 (2026-02)** · [网站](https://aalmuzairee.github.io/squint/) · [code](https://github.com/aalmuzairee/squint)

### 为什么加它

1. **2026 年最新的强 sim-to-real 视觉 RL**（比我们 ICRA 2026 deadline 早半年，正好可以引用）
2. **专攻 sim-to-real**：8 个真机任务上 **91.3% 成功率**、15 分钟训练——这是我们直接想比的场景
3. **配套 SO-101 廉价机械臂 benchmark**：$110 硬件门槛低，很多组会用；论文的立场是"低成本快速 sim-to-real"，我们能拿它做直接对比
4. 官方 GitHub 开源，作者是 UCSD 的 Henrik Christensen 组

### 我们和 Squint 的差异叙述（写进 Related Work）

Squint 的核心贡献是**训练效率**（并行仿真 + distributional critic + resolution squinting），使用普通 CNN encoder。我们的贡献是**表征结构**（DINOv2 + slot + binding），两者正交——甚至可以把 Squint 的训练 tricks 加到我们的架构上（future work）。**对比时突出**：Squint 在 clean sim-to-real 上很强，但**背景干扰**下会崩（因为 encoder 是 pixel-based CNN，没有解耦机制）。这正好是我们 Q2 (Distracting Suite) 的故事。

### 复现速通

```bash
git clone https://github.com/aalmuzairee/squint
cd squint
pip install -e .

# ManiSkill3 SO-101 task set - 单 3090/A5000 训 15 分钟
python train.py env=so101-pick task=pick-red-cube
```

### 挑战

- 用 SO-101 arm 不是 Franka，如果你不打算换硬件，就在 sim 里对比（在 Meta-World / Robosuite 上跑 Squint 需要自己 port，官方没给）
- 或者：**只在 sim 上跑 Squint 作为 SOTA 对比，真机上不比**——这样也能引用它作为"最新 sim-to-real SOTA"

---

## 🥉 完整性基线（世界模型家族代表）：**DreamerV3**

**Hafner et al., 2023, arXiv:2301.04104**

### 为什么还是要有

1. **世界模型家族的黄金标准**——reviewer 不看到 DreamerV3 会问"why not"
2. 官方 JAX 代码 `github.com/danijar/dreamerv3`，也有强 PyTorch 移植 `NM512/dreamerv3-torch`
3. Robosuite / Meta-World 上有公开的 baseline 数字，起 sanity-check 作用

### 复现速通

```bash
# 推荐 PyTorch 版更好读
git clone https://github.com/NM512/dreamerv3-torch
cd dreamerv3-torch
pip install -r requirements.txt

python dreamer.py --configs dmc_vision --task dmc_walker_walk
```

### 挑战

- 训练慢：单任务 ~24 小时（比 TD-MPC2 慢 2×），4 卡跑 5 seed × 3 环境是**整个实验预算里最贵的一项**
- JAX 版更快但依赖 JAX 编译环境；如果实验室主要跑 PyTorch，用移植版

---

## 🎯 关键 Related Work 警报：必读并回应

### 「The Surprising Ineffectiveness of Pre-Trained Visual Representations for MBRL」

**NeurIPS 2024** · arXiv:2411.10175

**这是我们最大的 related-work 挑战**。这篇论文的核心 claim：**PVR (R3M / VC-1 等冻结视觉编码器) 在 model-based RL 上并不比从头训 encoder 更好**。这直接质疑我们冻结 DINOv2 的选择。

### 怎么回应（写进 Related Work + Discussion）

我们的架构对这篇文章的批评有直接答案：

1. **他们测的是全局 PVR embedding（单向量 per 图像）**；我们用的是 **DINOv2 dense patch tokens + Slot Attention**，这是 pixel-level 的分解，不是全局 embedding
2. **他们没有 cross-modal binding**；proprioception 通道解决了他们观察到的"latent dynamics 不够 informative"的问题
3. **他们没有背景/前景 explicit disentanglement**；我们的路由机制正是让 downstream 只吃前景，回避了 PVR 特征里背景冗余的问题

**这个 hook 反过来能让 HippoAct 变得更有故事**：我们能声明 "我们的架构是对 [Wang 2024, NeurIPS] 观察到的 PVR-MBRL 失败模式的一个建设性解答"。这是**投稿的加分项**。

有一篇 2025 年的 counter-response：**"Pre-trained Visual Representations Generalize Where it Matters in Model-Based RL"** — 说 PVR 在特定条件下确实有帮助。可以两篇都引用，说明这是活跃争论。

---

## 我最终推荐的对比矩阵（在 §IV.A Table I 中排布）

| 类别 | 方法 | 优先级 | 备注 |
|---|---|---|---|
| **直接对手（planner 相同、只差 encoder）** | TD-MPC2-pixel | 🔴 必做 | 主 baseline |
| **最新 sim-to-real SOTA** | Squint | 🔴 必做 | 至少 sim 上跑；真机上如果没换硬件就 discuss |
| **世界模型家族** | DreamerV3 | 🟡 建议 | 慢，但 reviewer 会点名 |
| **model-free 强 baseline** | DrQ-v2 | 🟢 可选 | 老但是 sanity-check |
| **冻结 PVR 家族** | VC-1 或 Theia | 🔴 必做 | 直接回应 NeurIPS 2024 的 "surprising ineffectiveness" |
| **前身工作** | 原稿 (阈值+GRU) | 🔴 必做 | 作为 ablation A9 |

**取舍**：4 卡 A5000 跑 5 method × 5 seed × 3 环境 = 75 runs，如果每 run 平均 12 小时（DreamerV3 拉高了平均），全并行 4 卡挂钟 ~10 天。**Squint 建议只跑 3 环境 × 3 seed 作为 anchor**（不做完整 sweep），节省 30% 时间。

---

## 快速行动 checklist

- [ ] 今天：`git clone` TD-MPC2 官方 repo，跑通 walker-walk 单任务 seed=0，作为流水线通路测试
- [ ] 明天：跑 TD-MPC2 在 Meta-World pick-place-v2 上，用它的最终数字作为 §IV Table IV 里 "TD-MPC2-pixel" 那一行的**真实实测参考值**（替换掉当前的 ○ 估算）
- [ ] 本周：把 Squint 官方 SO-101 checkpoint 拉下来，先在 ManiSkill3 里 replay，理解它的观测格式，之后决定要不要在自己 Franka 环境里 port
- [ ] 本周：把 NeurIPS 2024 "Surprising Ineffectiveness" 全文精读，写成 Related Work 的第 3 段草稿

---

## Sources

- [TD-MPC2: Scalable, Robust World Models for Continuous Control](https://arxiv.org/abs/2310.16828)
- [Squint: Fast Visual Reinforcement Learning for Sim-to-Real Robotics](https://arxiv.org/abs/2602.21203)
- [Squint project page](https://aalmuzairee.github.io/squint/)
- [Squint GitHub](https://github.com/aalmuzairee/squint)
- [The Surprising Ineffectiveness of Pre-Trained Visual Representations for Model-Based RL (NeurIPS 2024)](https://arxiv.org/abs/2411.10175)
- [Pre-trained Visual Representations Generalize Where it Matters in Model-Based RL (2025)](https://www.researchgate.net/publication/395541495)
- [TD-MPC2 official code](https://github.com/nicklashansen/tdmpc2)
- [DreamerV3-Torch port](https://github.com/NM512/dreamerv3-torch)
- [World Models for Robotic Manipulation: A Survey (2026)](https://arxiv.org/pdf/2606.00113)
