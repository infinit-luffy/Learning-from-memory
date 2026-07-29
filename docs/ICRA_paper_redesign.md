# ICRA 重构方案：从 Atari 表征学习 → 机器人视觉运动策略

> 目标会议：ICRA 2026 (Contributed Paper, 6 pages + refs)
> 原稿基础：*Learning from Memory: Decision Making by Learning Associations in the Latent Space*（IJCAI 投稿版）
> 定位转向：从"Atari 上省显存的表征学习"→"机器人视觉运动策略的前景/背景解耦 + 情景记忆绑定"

---

## 1. 为什么原稿在 ICRA 会挂

| 硬伤 | ICRA 审稿人会怎么说 |
|---|---|
| 只在 Atari Alien 一个环境测 | "Not a robotics contribution." |
| 阈值 `D = (O − B) > thresh` 分离前景 | 真实相机噪声/阴影/光照下直接崩，reviewer 一句话就否 |
| 只对比原始像素 CNN | 没有 DrQ-v2 / DreamerV3 / R3M / MVP 就是拒稿 |
| 没有本体感 (proprioception) | ICRA 视觉运动学习论文标配 |
| 没有 sim-to-real | "How does this help a real robot?" |
| "hippocampus 启发"停在 GRU | 生物学动机与算法之间没有强绑定 |

---

## 2. 新故事线（一句话卖点）

> **原稿**：VAE 分背景+GRU 关联编码，Atari 上省 147× 显存。
>
> **重构**：机器人视觉运动学习中，场景 = 慢变背景 + 快变前景（物体+末端执行器）+ 本体感。我们把这三路解耦编码，再用海马体启发的**跨模态绑定记忆**把它们缝回策略输入。得到：更高的样本效率、天然的背景域随机化 → sim-to-real、以及一个可作为**安全兜底信号**的重构不确定度。

新名字候选（挑一个 → 让贯穿全文）：

- **HippoAct**：Hippocampus-Inspired Actionable Representations
- **BindRep**：Binding Representations for Visuomotor Policies
- **DiSeCoR**：Disentangled Scene-Conditioned Representation

建议用 **HippoAct**——生物学 hook 强，海马体绑定 (place × object × action) 正好对应 (background × foreground × proprioception)。

**新标题**（三选一）：

1. *HippoAct: Hippocampus-Inspired Disentangled Representations for Sample-Efficient Visuomotor Policy Learning*
2. *Binding Background, Foreground, and Proprioception: A Memory-Guided Representation for Robot Manipulation*
3. *Slow-Fast Scene Decomposition with Cross-Modal Memory for Sim-to-Real Robot Learning*

---

## 3. 四个新贡献（v2, 与强 backbone 匹配）

**C1. 无阈值物体中心解耦 (Object-Centric Slow-Fast Decomposition)**
弃用 VAE + 像素阈值。改用**冻结 DINOv2 patch tokens + Slot Attention**：K 个 slot 天然承载物体/区域粒度的表征；一个可学习 gate 用 Gumbel-Softmax 把每个 slot 路由到"慢通道"（背景/固定装置）或"快通道"（末端 + 可动物体）。**关键**：解耦是在特征空间而非像素空间做的，避开了阴影/反射/光照对像素阈值的破坏。

**C2. 跨模态绑定 Transformer 记忆 (Cross-Modal Binding Transformer)**
海马体 CA3 联想记忆的现代化实现：Perceiver-style cross-attention 把 (fast slots × proprioception × time-window) 绑成情景码 `c_t`。相比 GRU 有三处硬升级：(a) 显式跨模态注意力更符合"绑定 place × object × self"的生物动机；(b) 天然可扩展到长时窗；(c) 输出是 token 序列而非单一向量，供下游规划器直接消费。

**C3. TD-MPC2 潜规划集成 (Latent Planning on Disentangled Representations)**
不是外挂 SAC/PPO 完事——直接把 TD-MPC2 [Hansen'24, ICLR SOTA] 的隐世界模型 + MPPI 规划器嫁接在我们的 `[fast slots, c_t]` 之上。这是"表征解耦 × 高效规划"的**首次组合**——TD-MPC2 原版从 raw pixel encoder 取 latent，我们证明**把 encoder 换成解耦表征，样本效率和鲁棒性同时上升**。

**C4. Slot 级背景交换 → Sim-to-Real (Slot-Level Domain Swap)**
在 slot 空间直接做背景交换：两条不同背景 A、B 的轨迹，慢 slot `S^bg_A ↔ S^bg_B`，要求 fast slot、`c_t`、动作输出不变。这是**内生的、无渲染成本的** domain randomization——比像素级贴纹理更清洁、更节省算力，也更容易在真机上验证。

---

## 4. 方法（v2, 强 backbone stack）

### 4.1 骨架总览

```
raw obs O_t (224×224×3)
   │
   ▼
[frozen DINOv2 ViT-S/14]         ← 22M params, frozen, ~1ms/frame
   │  P_t ∈ ℝ^{196×384}   (patch tokens)
   ▼
[Slot Attention, K=16, 3 iters]   ← 学习槽位，物体中心
   │  S_t = {s_t^k}_{k=1}^{16}, s_t^k ∈ ℝ^{128}
   ▼
[Slot Router w/ Gumbel-Softmax]   ← 每 slot 分配 slow/fast
   │  → S_t^bg (慢),  S_t^fg (快)
   ▼
[Cross-Modal Binding Transformer] ← 4 层, 跨注意力
   │  input tokens: S_t^fg × T-window ⊕ q_{t-T+1:t}
   │  output: c_t ∈ ℝ^{128}  (情景码)
   ▼
z_t = concat([flatten(S_t^fg), q_t, c_t]) ∈ ℝ^{~300}
   │
   ▼
[TD-MPC2 latent world model + MPPI planner]
   │  π(a_t | z_t)  via  latent rollout + Q-guided planning
   ▼
action a_t
```

**参数量**：DINOv2 冻结 22M（不训） + Slot Attention ~0.5M + Router ~0.05M + Binding Transformer ~5M + TD-MPC2 head ~2M ≈ **7.5M 可训练**，A5000 单卡毫无压力。

### 4.2 记号

`O_t ∈ ℝ^{224×224×3}` 观测；`q_t ∈ ℝ^{d_q}` 本体感（关节 q, dq, 末端 6D pose）；`a_t` 动作；`r_t` 奖励。

### 4.3 冻结 DINOv2 视觉特征
- 用 DINOv2 ViT-S/14（22M，Meta AI 开源），冻结、只前向
- Patch tokens `P_t ∈ ℝ^{196×384}` (14² patch, 14×14 网格, 去 CLS)
- 为什么冻：(a) 已在 142M 图像上自监督预训练，语义粒度已经很好；(b) DINOv2 从头训要百卡；(c) 冻结让 A5000 撑得住剩余组件
- 备选：ViT-B（性能↑速度↓）；如任务里有语言可换 SigLIP

### 4.4 Slot Attention 物体中心解耦

按 [Locatello'20; Seitzer'23 DINOSAUR] 风格：
- K=16 个可学习 slot queries，128d
- 3 轮 iterative attention: `S_t = SlotAttention(P_t, K=16)`
- **训练目标**：从 slots 重构 DINOv2 patch feature（不是像素）——这就是 DINOSAUR 的关键：feature 重构比 pixel 重构更容易让 slot 聚成物体：

```
L_slot = ‖P_t − Decoder_slot(S_t)‖²
```

### 4.5 慢/快 Slot 路由

每个 slot k 的路由 gate：
```
π_k = softmax(MLP(s_t^k))              # ∈ Δ¹, [p_slow, p_fast]
g_k = GumbelSoftmax(π_k, τ=0.5)        # 可微硬分配
S_t^bg = {s_t^k : g_k = slow}
S_t^fg = {s_t^k : g_k = fast}
```

**慢正则**（让 background slot 真的时间平滑）：
```
L_slow = Σ_k g_k^slow · ‖s_t^k − sg(s_{t-1}^k)‖²
```

**路由先验**（避免全都被路到快通道）：
```
L_route = KL(mean_k(π_k) ‖ prior)      # prior = [0.7, 0.3]
```

### 4.6 Cross-Modal Binding Transformer

T=4 时间窗，输入 tokens：
```
X = [ S_{t-3}^fg, proj(q_{t-3}), ..., S_t^fg, proj(q_t) ]
```

4 层 Transformer decoder（causal mask），128d，4-head：
```
c_t = MeanPool( TransformerDec(X) ) ∈ ℝ^{128}
```

**c_t 上的辅助损失**：
```
L_predict = ‖q_{t+1} − MLP(c_t, a_t)‖²                    # 前向本体感
L_align   = InfoNCE(c_t, c_t^+ | action-similar pairs)    # 动作对齐
```

### 4.7 TD-MPC2 潜规划

按 [Hansen'24 TD-MPC2] 标准：
- 从解耦表征拼出 latent `z_t = [S_t^fg, q_t, c_t]`
- Latent dynamics `f_θ`, reward head `R_θ`, Q head `Q_θ`
- Planning: MPPI, horizon H=5, 512 samples

标准损失：`L_TDMPC2 = L_consistency + L_reward + L_Q + L_π`

### 4.8 Slot 级背景交换 (Sim-to-Real 关键)

两条不同背景轨迹 A、B（可 sim-sim 也可 sim-real）：
1. 分离得 `(S_A^bg, S_A^fg, c_A)`、`(S_B^bg, S_B^fg, c_B)`
2. 交换慢 slot：`Ŝ_A = S_B^bg ∪ S_A^fg`
3. 一致性：
```
L_swap = ‖c(Ŝ_A) − c_A‖² + ‖π(Ŝ_A) − π(S_A)‖²
```

**这个 loss 在 Stage 2 每 batch 以 30% 概率掺入**。无需渲染纹理库、无需 domain randomization 库——这是 slot 空间比像素空间清爽的地方。

### 4.9 两阶段训练管线

**Stage 1: 无监督表征预训练** (~2h on 1× A5000)
- 数据：离线 demo (~200k frames) 或 free exploration
- 冻结 DINOv2；训练 Slot Attention + Router
- 损失：`L_slot + λ_slow L_slow + λ_route L_route`

**Stage 2: TD-MPC2 在线学习** (~10h on 4× A5000, 每环境 1M steps)
- 冻结 DINOv2；Slot Attention 可选微调 (lr × 0.1)
- 训练 Binding Transformer + TD-MPC2 head
- 全部损失：`L_total = L_TDMPC2 + λ₁ L_predict + λ₂ L_align + λ₃ L_swap`

### 4.10 安全门（保留 v1 想法，换到 slot 空间）

不用像素重构误差，改用 slot 重构 patch feature 误差：
```
u_t = ‖P_t − Decoder_slot(S_t)‖   → 校准 τ_safe on training data (95%)
if u_t > τ_safe: fallback (hold joint position)
```
slot-space 的好处：DINOv2 特征对光照/背景更鲁棒，`u_t` 只在**结构性 OOD**（新物体、遮挡、相机故障）才升高，不会被平凡光照变化误触发。

---

### v1 vs v2 骨架对比

| 组件 | v1 (原稿风格) | v2 (当前) |
|---|---|---|
| 视觉编码 | 4-layer CNN + VAE | **冻结 DINOv2 ViT-S** |
| 前景分离 | 阈值二值化像素 | **Slot Attention + Gumbel 路由** |
| 时间绑定 | GRU | **Cross-attention Transformer** |
| 下游策略 | SAC / PPO on flat vec | **TD-MPC2 潜规划 + MPPI** |
| 域随机化 | 无 或像素纹理 | **Slot 级背景交换** |
| 可训练参数 | ~4M | ~7.5M |
| 外部依赖 | 无 | DINOv2 checkpoint (~90MB) |

---

<!-- 以下为 v1 legacy 方法段，保留供讨论；正式论文用 v2 -->

<details>
<summary>v1 legacy (仅供参考)</summary>

### v1.1 记号

`O_t ∈ ℝ^{H×W×3}` 观测图；`q_t ∈ ℝ^{d_q}` 本体感（关节位置/速度/末端 6D pose）；`a_t` 动作；`r_t` 奖励。目标：学一个紧致状态 `s_t = [b_t, f_{t−k:t}, c_t]` 供下游 RL/BC。

### v1.2 慢-快解耦 VAE（替换原文 §3.5）

- 背景编码：`μ_b, σ_b = E_b(O_t)`，`b_t ~ 𝒩(μ_b, σ_b²)`
- 掩码预测：`M_t = σ(g_ψ(O_t, D_b(b_t)))`，`M_t ∈ [0,1]^{H×W}`
- 前景图：`F_t = M_t ⊙ O_t`
- 前景编码：`f_t = E_f(F_t)`

**损失**（联合训练，端到端）：

```
L_slow  = ‖(1−M_t) ⊙ (O_t − D_b(b_t))‖²         # 背景像素重构
L_fast  = ‖M_t ⊙ (O_t − D_b(b_t) − D_f(f_t))‖²  # 前景残差重构
L_KL    = KL(𝒩(μ_b, σ_b²) ‖ 𝒩(0, I))
L_temp  = ‖b_t − sg(b_{t−1})‖²                   # 背景时间平滑（关键！）
L_mask  = λ₁‖M‖₁ + λ₂ TV(M)                     # 稀疏+平滑先验
```

`L_temp` 是让 `b` 真正学到"慢"变量的关键；`sg(·)` 是 stop-gradient。

### v1.3 跨模态绑定记忆（升级原文 §3.6）

```
h_0 = Conv(b_t)                           # 背景作为初始隐状态
h_i = GRU([f_{t−k+i}, q_{t−k+i}], h_{i−1})   # 融合前景 + 本体感
c_t = h_k                                 # 情景码
```

**三重损失**（联合训练）：

```
L_recon = ‖O_t − D_o(c_t, b_t)‖²                        # 观测重构
L_dyn   = ‖q_{t+1} − MLP(c_t, a_t)‖²                    # 前向本体感预测
L_act   = InfoNCE(c_t, c_t^+ | action-similar pairs)    # 动作对齐对比
```

`L_dyn` 强迫 `c_t` 编码可预测下一步的运动信息；`L_act` 是关键新加损失——把动作相似（角度距离 < ε）的样本在潜空间拉近。这两个是回应"actionable representation"批评的答案。

### v1.4 背景域随机化预训练

给定轨迹 `(O_{1:T}, q_{1:T}, a_{1:T})`：

1. 前向传播分离 `b_t, f_t`；
2. 采样纹理库 `𝒯` 中的随机纹理 `τ`，合成 `O'_t = paste(τ, F_t)`；
3. 一致性约束 `‖f_t − f'_t‖² + ‖c_t − c'_t‖²`。

预训练用离线机器人数据（Open-X-Embodiment 子集或 sim 采集），策略学习时可冻结/微调编码器。

### v1.5 策略网络与 RL

- 观测：`s_t = [f_{t−k:t}, q_t, c_t]`（**去掉 `b_t`**——它对策略无信息，只用于重构监督）
- 策略：3-layer MLP（原文也是 MLP，这里保留）
- RL 算法：SAC（连续）或 DrQ-v2 backbone（保证公平对比）

### v1.6 安全门

```
u_t = ‖O_t − D_o(c_t, b_t)‖   → 训练集分布上校准阈值 τ_safe (95th percentile)
if u_t > τ_safe:  fallback (hold joint position)
else:             π(s_t)
```

</details>

---

## 5. 实验设计（这是 ICRA 论文的命根子）

> **硬件确认**：4× A5000 (24GB each, 96GB 总) + 真机机械臂 + 顶级实验室
> 因此把**真机实验提到主结果**，sim 结果作支撑；这是本论文最强的差异化。

### 5.1 环境矩阵（已按硬件校准）

| 环境 | 任务 | 权重 | 为什么选它 |
|---|---|---|---|
| **真机机械臂** | pick-place / door-open / stacking，都在干扰场景 | 🟢 **主结果** | 顶会级差异化，其他表征学习论文极少 |
| **Distracting Control Suite** | walker/cheetah/hopper + 动态视频背景 | 🟢 **主结果** | 直接验证"背景/前景解耦"核心 claim |
| **Robosuite (Panda)** | pick-place, door open, nut assembly | 🟢 主 | 与真机同 URDF，是 sim-to-real 的桥梁 |
| **Meta-World MT10** | 10 任务操作 | 🟡 支撑 | 多任务样本效率，业界共识 benchmark |
| **RLBench** | 3 长时任务 | 🟡 可选 | 只做 1-2 个体现 long-horizon |
| **Isaac Lab locomotion** | ❌ 不做 | | 分散故事，除非有富余算力 |

**真机具体任务设计（3 个，每个 30 trial × 3 场景条件 = 270 trial 总量）：**

| 任务 | 干扰变量 | 目的 |
|---|---|---|
| **T1: Pick-and-Place** | 桌布纹理×3、光照×2、桌面干扰物有/无 | 验证背景不变性 |
| **T2: Door / Drawer Opening** | 门颜色×3、把手位置抖动 | 验证前景聚焦 |
| **T3: Cup Stacking**（长时） | 相机位姿抖动 ±5cm | 验证情景记忆 c_t 的效用 |

**加分项 (Cross-Embodiment)**：如果实验室有第二个 arm (UR5 / xArm)，做 sim(Panda) → real(UR5) 迁移，这是 CoRL/RSS 级别的加分。

### 5.2 基线（Table I 必备，按 4× A5000 现实剪裁）

| 类别 | 方法 | 优先级 | 4-GPU 单 seed 训练时间估算 |
|---|---|---|---|
| 原始像素 | SAC-CNN / PPO-CNN | 🟢 必做 | ~8 h |
| 数据增强 | **DrQ-v2** [Yarats'21] | 🟢 必做 | ~12 h（strong 主对比） |
| 对比学习 | **CURL** [Laskin'20] | 🟢 必做 | ~15 h |
| 世界模型 | **Dreamer-v3** [Hafner'23] | 🟢 必做 | ~24 h **← 瓶颈** |
| 表征基线 | SPR [Schwarzer'21] | 🟡 可选 | ~15 h |
| 冻结预训练 | **R3M** [Nair'22] | 🟢 必做 | ~4 h（frozen encoder，快） |
| 冻结预训练 | MVP / VC-1 | 🟡 至少 1 个 | ~4 h |
| 前身工作 | 原稿 (阈值+GRU, 无 proprio) | 🟢 必做 (ablation) | ~6 h |
| Ours | HippoAct | 🟢 | ~10 h |

**4-GPU 排布策略**：4 seed 并行 = 4 卡各跑一个 seed。5 seed × 5 主基线 × 3 sim 环境 ≈ 375 GPU-h ≈ **4 天挂钟**（4 卡并行）。够。

**统计**：每环境 5 seed，主图报 mean ± std + IQM；显著性检验用 stratified bootstrap (rliable 库)——ICRA reviewer 一定会问。

### 5.3 消融（Table II 必备）

| 变体 | 去掉的组件 | 假设的效果 |
|---|---|---|
| A0 | Full HippoAct | — |
| A1 | 去掉 `L_temp`（背景不"慢"） | 背景解耦崩，性能↓ |
| A2 | 阈值化替代软掩码 M | 复现原稿，测真实场景差距 |
| A3 | 去掉 proprioception 支路 | 复原到原稿架构 |
| A4 | 去掉 `L_dyn` + `L_act` | 表征"不可动作" |
| A5 | 去掉背景随机化 | Sim-to-real 掉幅测量 |
| A6 | 去掉安全门 | 分布外场景故障率↑ |

### 5.4 五组实验的具体设计

#### E1. 样本效率（Fig. 3, main result）
- Meta-World MT10、Distracting Control Suite (medium)、Robosuite pick-place
- x 轴：环境 step；y 轴：success rate 或 return
- HippoAct vs DrQ-v2 / Dreamer-v3 / CURL / raw CNN，5 seeds
- **必须赢**至少 2/3 环境的样本效率，否则整个论文没意义

#### E2. 背景鲁棒性（Fig. 4）
- 在 Distracting Control Suite 上，训练用 easy 背景，测试用 hard 背景（视频背景）
- 报告 zero-shot performance drop；HippoAct 应显著优于 DrQ-v2（因为背景被显式解耦）
- 这是最能讲故事的图

#### E3. Sim-to-Real 或 Sim-to-Sim（Fig. 5, Table III）
- Robosuite 训练 → 真机 Franka 部署 pick-place
- 指标：真机 success rate over 30 trials, 有/无背景干扰物
- 对比 HippoAct、DrQ-v2、R3M-frozen
- **加分项**：录视频作为 supplementary

#### E4. 表征质量（Fig. 6, 定性）
- t-SNE 可视化 `f` 和 `c`：验证前景聚类按物体/位置分离，`c` 聚类按 sub-task 分离
- 重构可视化：原图 vs 重构 vs 只用 `b` vs 只用 `f`
- 保留原稿 Figure 5 风格但换成机器人场景

#### E5. 安全门有效性（Fig. 7, Table IV）
- 人为注入 OOD：遮挡相机、切换灯光、放一个未见过的物体
- 报告 OOD 检测 AUROC；策略失败率对比
- 这一节可以只 1/2 页但一定要有

### 5.5 效率指标（原稿唯一亮点，保留并强化）

| 指标 | 原始像素 CNN | HippoAct | 比值 |
|---|---|---|---|
| Replay buffer size (100k steps) | ~11 GB | ~200 MB | 55× |
| Wall-clock training time (M steps) | X h | Y h | — |
| Inference latency (per step) | X ms | Y ms | — |

原稿说 147×，但那是 84×84×4 vs 192-dim；在 128×128×3 机器人分辨率下会更夸张，这段可以保留但更严谨地测量。

---

## 6. 论文结构（6+n 页）

```
1. Introduction                    ~1 页    (motivation: robotic visuomotor learning 视角切入)
2. Related Work                    ~0.75 页 (visual RL、robotic representations R3M/MVP/VC-1、世界模型)
3. Method                          ~2 页    (含 Fig 1: 整体架构, Fig 2: 解耦示意)
   3.1 Slow-Fast Scene Decomposition
   3.2 Cross-Modal Binding Memory
   3.3 Background-Randomized Pretraining
   3.4 Reconstruction-Uncertainty Safety Gate
4. Experiments                     ~2.25 页
   4.1 Setup
   4.2 Sample Efficiency (Fig 3)
   4.3 Background Robustness (Fig 4)
   4.4 Sim-to-Real Transfer (Fig 5, Table III)
   4.5 Representation Analysis (Fig 6)
   4.6 Safety Gate (Fig 7)
   4.7 Ablations (Table II)
5. Limitations & Conclusion        ~0.25 页
References                         ~1 页
```

---

## 7. 讲故事的三张核心图（务必视觉过硬）

- **Fig. 1**：Overview，输入 → 慢/快解耦 → CMB → 策略 + 安全门；用真实机器人操作场景做示意
- **Fig. 2**：Distracting Control Suite 上，同一时刻 `b`（背景）/ `f`（前景）/ `Ô`（重构）/ `M`（掩码）四联图
- **Fig. 5**：Sim-to-real 部署真机照片 + 成功率柱状图

---

## 8. 与原稿的差异表（写给自己 / 给合作者）

| 维度 | 原稿 | HippoAct |
|---|---|---|
| 前景提取 | 阈值化 `O−B > τ` | 学习软掩码 M |
| 模态 | 仅视觉 | 视觉 + 本体感 |
| 关联损失 | 仅重构 | 重构 + 前向本体感预测 + InfoNCE 动作对齐 |
| 时间正则 | 无 | `L_temp` 强迫 b 慢变 |
| 域随机化 | 无 | 背景纹理随机化 |
| Sim-to-real | 无 | 零适配部署 |
| 安全信号 | 无 | 重构误差作为 OOD gate |
| 环境 | Atari Alien | Meta-World + Robosuite + DCS + 真机 |
| 基线 | 只对 raw CNN | DrQ-v2, Dreamer-v3, R3M, MVP, CURL |

---

## 9. 时间线（4× A5000 + 真机，投稿 ICRA 2026 September）

**并行 track**：sim 训练在 GPU，真机数据在人力。两条腿走。

| 周 | GPU Track (4× A5000) | 真机 / 硬件 Track | 写作 Track |
|---|---|---|---|
| W1 | 实现 slow-fast VAE + 软掩码 | 机械臂标定、相机内参外参 | — |
| W2 | Distracting DMC 单任务跑通 → 验证解耦 | 遥操作采集 20 条 real demo（多背景） | — |
| W3 | 加入 CMB + proprioception；接 DrQ-v2 backbone | 搭建 T1 pick-place 桌面装置 | — |
| W4 | 背景随机化预训练；Meta-World MT10 | Sim Panda ↔ 真机对齐（joint mapping） | Related Work 草稿 |
| W5 | 主要基线复现开跑（4 seed 并行）| T1 数据采集完成 | Method 草稿 |
| W6 | Dreamer-v3 基线（挂机 5 天）| T2 door-open 装置 + 采集 | — |
| W7 | E1 样本效率主图数据齐 | T3 stacking 装置 + 采集 | — |
| W8 | E2 背景鲁棒性 (DCS hard split) | **真机部署评估 T1** (Ours + 3 baseline × 30 trial × 3 场景) | Intro 草稿 |
| W9 | 消融 A1–A6 全量 | **真机部署评估 T2、T3** | Experiments 草稿 |
| W10 | E4 表征可视化 + E5 安全门 | 拍摄 demo 视频（supplementary） | 全文串接 |
| W11 | Buffer：补跑翻车的 seed | 补录 corner case | 内部审阅 v1 |
| W12 | — | — | 改稿 v2 |
| W13 | — | — | 图表精修、reference 核对 |
| W14 | — | — | 提交 |

**挂钟总计**：GPU 主计算约 W5-W9 是 peak，估 ~500 GPU-h ≈ 5-6 天全速 4-卡。**足够，但 Dreamer-v3 别拖到最后**。

---

## 10. 优先级堆栈（if things fall behind, cut from bottom）

| # | 内容 | 弃保决策 |
|---|---|---|
| 1 | **HippoAct 在真机 T1 显著优于 DrQ-v2** | 🔴 死保，没有这个论文没意义 |
| 2 | Distracting DMC 背景鲁棒性主图 | 🔴 死保 |
| 3 | 消融 A2 (软掩码 vs 阈值)、A3 (proprio)、A5 (背景随机化) | 🔴 死保三项核心消融 |
| 4 | Sample-efficiency 三环境（MW / Robosuite / DCS） | 🟠 保二砍一 |
| 5 | Dreamer-v3 基线 | 🟠 保，砍别的也要保它 |
| 6 | 真机 T2 + T3 | 🟡 保 T2 砍 T3 也可，但会弱 |
| 7 | Safety Gate (E5) | 🟢 可砍到 half-page discussion |
| 8 | Cross-embodiment (UR5) | 🟢 加分项，砍无所谓 |
| 9 | Meta-World MT50 | 🟢 直接 MT10 就够 |
| 10 | RLBench 长时任务 | 🟢 砍 |

---

## 11. Risk & Backup Plan

- **风险 1（最高）**：软掩码 M 学不出稀疏——真实场景反射/阴影污染前景。
  **兜底**：Segment Anything 2 生成掩码伪标签监督 M；或改成 slot-attention (SLATE)。
- **风险 2**：Sim-to-real 掉幅太大。
  **兜底**：(a) 加 vision-based domain adaptation（style transfer）；(b) 论文改叙述为"encoder as robust visual backbone"，强调背景鲁棒（DCS 主图），弱化真机 zero-shot。
- **风险 3**：样本效率打不过 Dreamer-v3。
  **兜底**：主打**内存 (55×) + 背景鲁棒 + sim-to-real + 可解释解耦**四点；样本效率次要。
- **风险 4**：真机数据采集拖到 W7 后没做完。
  **兜底**：T1 单任务 + 加强 sim 结果也能撑住 ICRA（不出彩但能进）。
- **风险 5**：4 卡不够并行所有基线。
  **兜底**：CURL / SPR 二选一；VC-1 / MVP 二选一。

---

## 12. 顶级实验室加分动作（如果精力允许）

- **RSS/CoRL 同时投稿路线**：故事一致，但 CoRL 更吃真机 demo。可以先投 CoRL Workshop 拿反馈，再投 ICRA 正会
- **数据/代码开源**：ICRA 越来越看重 reproducibility；GitHub + Hugging Face 挂 checkpoint
- **Attention/interpretability 附录**：可视化 M_t 随时间演化的 heatmap，reviewer 喜欢
- **Video ready**：3 分钟 supplementary video（真机 pick + 干扰场景 + safety gate 触发），是 ICRA 强 signal
- **导师背书**：如果导师是 ICRA AE/senior reviewer，直接问他/她愿不愿意主导 discussion——很多论文的临门一脚就是这个
