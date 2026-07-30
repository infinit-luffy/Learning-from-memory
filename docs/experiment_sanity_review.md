# 全局审视：后续实验的合理性分析

> 2026-07-30，synthetic phase 关闭后、Phase 2 启动前的冷静盘点。
> 结论先行：**Phase 2 的方向对，但顺序有一处错误、范围有一处过大、
> 且论文 claim 结构需要按"已验证/未验证"重新分层。**

---

## 1. 诚实的资产负债表

### 已验证（有数据支撑）

| 资产 | 证据 | 强度 |
|---|---|---|
| Stage-1 定位（合成） | coverage 1.000, enrichment 18.8× | 强 |
| 时间信号系统性反转 | 3 target × 3 encoder, d≈−1.25 | 强，机制清楚 |
| 连通性路由（合成） | d=+1.89, AUC 0.925, router 超代理 | 强 |
| Diagnostic Protocol | 4 条规则各对应真实 caught failure | 强（方法学贡献） |

### 未验证（论文里写了但零数据）

| 负债 | 风险等级 | 说明 |
|---|---|---|
| 连通性在真实图像上成立 | 🔴 **最高** | 合成 disk ≠ walker 关节体 ≠ 真机桌面。整个故事的地基 |
| **任何 RL 结果** | 🔴 | 至今零 policy 训练。"Act"部分完全未触碰 |
| Binding Transformer + proprio | 🟠 | 从未训过，L_pred/L_align 从未执行过 |
| Slot-swap augmentation (Q3) | 🟠 | 从未测过 |
| Safety gate (Q5) | 🟡 | 从未校准过 |
| Q1 sample efficiency 能赢 | 🟠 | NeurIPS'24 "PVR 无效"论文提示可能只打平 |
| 真机全套 (Q3) | 🔴 | ~100h 采集，一人一卡，deadline 前不现实 |

**关键失衡**：8 轮迭代全部花在 Stage-1 表征上。论文 6 个 Q 里 5 个依赖
下游 RL，而 RL 侧连管线都还没通。

---

## 2. Phase 2 计划的三个问题

### 问题 1：顺序错了 — P2.3 应该第一个跑

现排序 P2.1(管线) → P2.2(baseline) → P2.3(Stage-1 on DCS) → P2.4(端到端)。

但 **P2.3 的 Stage-1 检查是唯一能杀死整个故事的实验**：如果连通性信号
在 walker（细长关节体，多个非紧凑部件）上不成立，后面全部作废，而且
论文核心章节（§III.D.3 / §IV.I）需要重新定位。它只要 1 天，而 P2.1/P2.2
是纯管线工程，风险为零但耗 GPU 数天。

**修正**：P2.3 的 Stage-1 部分提到最前（采 DCS 帧 → Stage-1 → 连通性
检查），P2.1/P2.2 与之并行排队（P2.2 挂机不占人力）。

具体风险点：walker 的躯干+四肢在 alpha 上可能是**多个中等紧凑区域**，
而 DCS 视频背景里的物体（车、人）是**真正紧凑的干扰**。最坏情形是
"背景视频里的车被判 fast，walker 大腿被判 slow"。这个一天内就能知道。

### 问题 2：首个端到端实验的复杂度过高

P2.4 现方案：完整 HippoAct（slots + router + binding transformer + proprio
+ L_pred + L_align）vs TD-MPC2-pixel。**一次性引入 5 个未验证组件**——
任何一个坏了都是一周级的归因成本，而我们在 synthetic 上的教训恰恰是
"一次一个变量"。

**修正**：端到端分三级递进：
```
E2E-0: z = flatten(S_fg) ⊕ q_t，无 binding、无辅助 loss   ← 最小可行
E2E-1: + Binding Transformer (c_t)，无 L_pred/L_align
E2E-2: + L_pred + L_align（完整方法）
```
E2E-0 能跑到 pixel baseline 的 80% 就说明表征可用；每级增量都是
一个天然的 build-up ablation，正好填 Table IX。

### 问题 3：论文 claim 范围 vs 时间与算力的现实

- **时间**：ICRA 2026 deadline ~9 月 15 日，剩 6-7 周。原计划 14 周。
- **算力**：当前实际是 1× RTX 5080（学校 4×A5000 还没挂上）。
  DCS 500K steps ≈ 1-2 天/run。Q1+Q2 最小矩阵（2 任务 × 2 方法 × 3 seed
  ≈ 12 runs）单卡串行就要 2-3 周。7 个 baseline × 3 环境的完整 Table IV
  在单卡上**不可能**。
- **真机**：T1/T2/T3 全套 ~100h 人力 + 尚未搭建。6 周内与 sim 实验并行
  完成不现实。

三个 scope 选项（见 §4 决策）。

---

## 3. 修正后的实验优先级（按"证据价值/成本"排序）

| # | 实验 | 成本 | 它证明什么 | 若失败 |
|---|---|---|---|---|
| 1 | **Stage-1 on DCS 帧** + 连通性检查 | 1 天 | 核心故事在真实图像分布上成立 | 故事重定位（回 cowork） |
| 2 | TD-MPC2-pixel baseline 对齐 | 2 天挂机 | 管线可信 + Table V 真实数字 | 查超参 |
| 3 | **E2E-0**（最小 z）DCS easy | 2 天 | 表征可驱动 policy | 查 adapter |
| 4 | **Q2 核心**：easy 训练 → {none,easy,hard} zero-shot，2 方法 × 2 任务 × 3 seed | ~2 周挂机 | 论文主 claim | 主图数据 |
| 5 | E2E-1/E2E-2 递进（=A5/A7 ablation） | 各 2 天 | binding/辅助 loss 的增量 | 如实报告 |
| 6 | Stage-1 on Meta-World 帧 + 1 个操作任务 | 1 周 | robotics 相关性 | 可砍 |
| 7 | 真机 T1 单任务小规模（30 trial demo 级） | 1-2 周人力 | Q3 降级版 | 可砍/延后 |
| 8 | Safety gate 校准 + OOD 测试 | 2 天 | Q5 | 可砍成 discussion |
| 9 | Slot-swap augmentation | 3 天 | Q3 的 sim 侧 | 可延后 |

1-4 是**必做**（缺一篇论文不成立）；5-6 强烈建议；7-9 按剩余时间。

---

## 4. 三个 scope 选项（需要你决策）

### 选项 A：ICRA 2026（9 月），表征为核心的紧凑论文
- Claim 收缩为：反转发现 + 连通性路由 + Diagnostic Protocol + **DCS 上的
  Q2 robustness**（2 任务 2 方法）+ Meta-World 1 任务示 robotics 相关性
- 砍掉：真机、slot-swap、safety gate（各降级为 discussion/future work）
- 风险：robotics 味道偏淡，ICRA reviewer 可能嫌"这是 RL 表征论文"
- 可行性：**高**（上面 1-6 刚好 6 周）

### 选项 B：CoRL 2027 春 / ICRA 2027（明年 3 月），完整系统论文
- 保留全部 6 个 Q，真机三任务，baseline 矩阵挂 4×A5000 跑满
- 风险：周期长；Squint 等竞品继续演进
- 可行性：高，且论文强度上限最高

### 选项 C：两步走 — 先 workshop/RA-L，再正会
- 6 周内把选项 A 的内容投 RA-L（rolling，可加 ICRA 2027 present 选项）
  或 NeurIPS workshop 拿反馈；真机与完整矩阵做完后投正会
- 可行性：高，反馈价值大，但要写两轮

**我的倾向：A 或 C**。当前最有价值的资产是反转发现 + 协议（新颖、扎实、
已完成），它们会因时间流逝贬值（别人也可能撞到）；完整系统故事可以在
期刊版补全。若学校 A5000 两周内能挂上，A 的实验矩阵还能再宽一档。

---

## 5. 不变的纪律

后续每个实验沿用 synthetic phase 尾声定型的规矩：训练前先量信号、
一次一个变量、判据前置、先渲染数据、真值来自生成过程、卡住带数据回 cowork。
Phase 2 的新增一条：**每个 GPU-天开销超过 1 天的 run，启动前把预期数字
写进 TODO_RESULT 的预注册段。**
