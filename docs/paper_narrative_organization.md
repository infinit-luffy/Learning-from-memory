# HippoAct — 叙事组织：每个贡献 = 一个具体痛点

> 核心原则：**每一条贡献必须先答"为什么现在需要它"，再答"我们怎么做的"**。ICRA reviewer 用秒表读 abstract，如果第一句没痛点，第二句没差异，第三句没数字，直接扔进 borderline 堆。

---

## 四条贡献 ↔ 四个痛点（一一对应表）

| # | 现实痛点（Robot 场景真发生的事） | 前人失败模式（点名+简因） | 我们的解答（C 编号） | 一句话数字 |
|---|---|---|---|---|
| **P1** | 机械臂视野里 70% 是不变的桌面/夹具/机架，pixel encoder 每帧重编码，replay buffer 存到 46 GB 才够 1M 步 | (a) 原稿 [IJCAI'25] 用阈值 O−B > τ 分前景，光照/反射一变就崩；(b) 冻结 PVR 全局 embedding 在 MBRL 上被证明**无效** [Wang et al., NeurIPS 2024, "The Surprising Ineffectiveness of PVRs for MBRL"] | **C1**：冻结 DINOv2 **dense patch tokens** + Slot Attention + Gumbel 慢/快 router，在**特征级+物体级**双重解耦 | 228× 显存缩减 |
| **P2** | 拉抽屉、插拔、堆叠——**视觉一样但物理不同**。纯视觉 policy 频繁拉过头/松脱 | DreamerV3、TD-MPC2、DrQ-v2 都把 proprioception 当作**平凡 concat**，没有跨模态融合 | **C2**：cross-attention Transformer 记忆显式绑定 fast slots × proprio 时窗，用 **forward proprio prediction + action-cosine InfoNCE** 强制 actionable | c<sub>t</sub> 线性探针预测物体位置 R²=0.89 vs pixel-CNN R²=0.42 |
| **P3** | Sim-to-real 是每个真机 RL 论文的瓶颈。Pixel-level domain randomization 需要 texture library / 可微 renderer / GAN，训练成本 2–3× | Squint [Almuzairee & Christensen'26]、Robosuite 官方 DR 都靠 ManiSkill3 heavy DR；texture 库有限、组合爆炸；且监督在错误的地方（pixel 一致而非 policy 一致） | **C3**：**slot 空间**背景交换——两条轨迹的慢 slot 直接互换，用 policy consistency loss 而非 pixel loss | 真机 sim-to-real 成功率相比 pixel-DR TD-MPC2 **+28 点绝对** |
| **P4** | 真机部署时人手进入、未见物体、相机故障——策略照跑，事故风险高。目前没有原生 runtime OOD 信号 | 端到端策略输出的 Q 值/熵不校准；PVR 特征上 L2 距离受光照污染，误触发率 39% | **C4**：DINOv2 patch feature 上的 slot 重构误差 u<sub>t</sub>，光照不变、结构性 OOD 时才升高 | OOD AUROC 0.95，误触发率仅 12% |

---

## 为什么这四个痛点选得对——ICRA reviewer 心里的对照

| Reviewer 关心的 dimension | 我们能不能证明它？ |
|---|---|
| **是否是真实工程问题** | ✅ P1-P4 每个都是真机上跑过的、感受过的痛点，不是虚构 |
| **是否有前人具体失败** | ✅ 每个 P 都点名一个具体 paper 的失败模式（不是笼统的"prior work is limited"） |
| **是否能用数字验证** | ✅ 每个 P 对应一个 headline number |
| **是否互相独立可 ablate** | ✅ Table IX 每个 P 都有对应的 ablation 行（A2/A3/A8/safety gate section） |

---

## 落到写作：新的 Intro 结构（4-paragraph 版）

**§I 段 1（motivation, 4-6 句）**——描绘真机 RL 的现状与阻塞点，把 4 个痛点串起来。不点自己的方法名。

**§I 段 2（gap，4-5 句）**——分别点名前人失败：原稿阈值分解、冻结 PVR 无效（NeurIPS'24）、pixel DR 昂贵、无 runtime OOD。这段每一句都要有引用。

**§I 段 3（our approach，5-7 句）**——一段话讲 HippoAct 的四个模块，每个模块前用 "**To address P1**, we ..." 显式绑定痛点。

**§I 段 4（contributions + results one-liner）**——4 个 bullet + 一句 headline 数字（"On three real Franka tasks, HippoAct achieves 71% zero-shot sim-to-real success, +14 pt over VC-1"）。

---

## 落到写作：新的 Abstract 骨架（对照痛点重写）

> 用问题-答案节奏替代之前的"我们做了什么"节奏。每句都有一个痛点或数字。

**Draft**：
> Visual reinforcement learning for real-world robot manipulation is throttled by four coupled failures: pixel encoders redundantly re-represent stable background (46 GB replay buffers for 1 M steps); frozen visual priors, though sample-efficient in principle, have been shown ineffective in model-based RL because their *global* embeddings discard spatial structure [NeurIPS 2024]; pixel-level domain randomization for sim-to-real requires texture libraries and enforces the wrong invariance; and end-to-end policies emit no calibrated signal to detect out-of-distribution deployment conditions. We introduce **HippoAct**, which addresses each failure in turn. **(P1)** A frozen DINOv2 patch encoder feeds Slot Attention with a Gumbel-Softmax router that separates temporally slow (background) from fast (foreground) slots at the *entity* level. **(P2)** A cross-modal Transformer memory binds fast slots with a proprioception window into an episodic code trained by forward-state prediction and action-cosine contrastive alignment. **(P3)** A slot-space background-swap augmentation enforces policy-consistency directly at the representation level, eliminating the need for a texture library. **(P4)** The slot-space reconstruction residual is calibrated as a runtime OOD safety gate. Integrated with TD-MPC2 latent planning, HippoAct exceeds TD-MPC2-pixel by 8 % IQM success at 60 % of the environment steps on Meta-World, retains 81 % of clean-scene return under DCS-Hard backgrounds (vs. 48 %), and achieves 71 % zero-shot success on three real Franka manipulation tasks, +14 pt over VC-1. The disentangled representation reduces replay memory by 228 ×.

**注意**：每个 P 编号都出现在 abstract 里，这样 reviewer 一眼看到"哦四个问题四个解答"，不用猜。

---

## 落到 §II Related Work 的组织（也用痛点分节）

传统 Related Work 是按方法家族分节（"Visual RL...", "World Models...", "Object-Centric..."），reviewer 看完记不住我们在哪里。改成**按痛点分节**：

**§II.A  Redundant Pixel Encoding in Visual RL**（对应 P1）
- 讨论 DrQ-v2, DreamerV3, TD-MPC2 都用 CNN encoder；引用原稿 IJCAI'25 尝试阈值分解但真实场景 fail
- 引用 [Wang et al. NeurIPS 2024] 指出 PVR 全局 embedding 在 MBRL 上无效
- 说明 object-centric 方向 (Slot Attention, DINOSAUR) 的机会

**§II.B  Cross-Modal Fusion of Vision and Proprioception**（对应 P2）
- 讨论现有 fusion 方法（concatenation、film、cross-attention）；点名 DreamerV3/TD-MPC2 用了最弱的 concat
- 引用 SlotFormer, Perceiver-IO 作为 attention memory 的思路来源
- 引用 [Liu et al. Science 2023] 关于 hippocampus 联想编码作为设计动机

**§II.C  Sim-to-Real via Domain Randomization**（对应 P3）
- 讨论 pixel-level DR (DR-DR, VIRAL, Squint) 的局限：texture 库、渲染成本
- 引用 slot-based augmentation 的少数尝试（SlotSwap in DINOSAUR 是特征增广不是策略增广）
- 说明我们是 first policy-consistent slot-space augmentation

**§II.D  Runtime OOD Detection for Robot Deployment**（对应 P4）
- 讨论目前的 OOD 方法（Q-ensemble variance、reconstruction error）
- 说明 pixel-level reconstruction 受光照污染的问题
- 引用最近 uncertainty for robot control 相关工作

每一节结尾一句："HippoAct addresses this by ...（一句话）"——这是让 reviewer 记住我们在哪的关键。

---

## 落到 §V Discussion 的组织（对偶结构）

Discussion 也用四个 P 反照一遍——**每个 P 我们赢了多少、还剩多少 open question**：

**§V.A  P1 revisited: Feature-level entity decomposition works** — 引 Table IX A1 说明冻结 DINOv2 贡献最大；剩余问题：K 敏感、大场景（bin picking）可能需要 K > 32

**§V.B  P2 revisited: Actionable binding needs both modalities** — 引 Fig. 5 T2 drawer 结果说明 proprio 必要；剩余问题：force/torque 未纳入

**§V.C  P3 revisited: Slot-space swap generalizes without texture libraries** — 引 Table VI 说明 +28 点 gain；剩余问题：swap 假设背景独立于前景（复杂几何 attach 关系可能违反）

**§V.D  P4 revisited: Safety gate detects structural OOD** — 引 Table VIII AUROC 0.95；剩余问题：只检测*视觉* OOD，力控异常需要额外 signal

---

## Reviewer 常见 rebuttal 弹幕，用这个组织能挡住的

- **"Contributions read like a laundry list"** → 不再是 laundry list，是 4 个显式痛点各配一个解答
- **"Motivation is generic"** → 每个 P 都点名具体前人失败（尤其是 NeurIPS 2024 那篇）
- **"Ablations don't map to contributions"** → Table IX 每行明确标注是哪个 P 的组件
- **"Why is object-centric right? Why not just [X]?"** → 答案在 §II.A + 引 NeurIPS'24 说明 global embedding 已经被证伪
- **"Why include safety gate at all? Feels tacked on"** → §II.D 已经把它绑到 P4 上，不再显得随意

---

## 快速自检 checklist（给你审稿用）

回头审这个组织时问自己：

- [ ] Abstract 里能不能一眼数出 4 个"痛点+解答"对
- [ ] Intro §I.段 3 里"To address P<i>, we..." 结构是否完整覆盖 4 个
- [ ] Related Work §II 是否按 P1-P4 组织
- [ ] Table IX 每一行的 Removed 列能否 map 到某个 P
- [ ] Discussion §V 是否对应四个 P 各一小节
- [ ] Fig. 1 的四个色块（frozen visual, slot decomp, binding, planning+gate）能否与四个 P 对齐

**这六个"是"都能勾上，那这篇论文就有清晰的骨架，reviewer 读完能复述你的故事——这是被接收的必要条件。**
