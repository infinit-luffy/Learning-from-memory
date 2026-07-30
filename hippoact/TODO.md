# TODO — 服务器 Agent 执行清单

> 权威任务清单。判据不过 → 停下带数据回 cowork。
> 最后更新：2026-07-30（**Scope 决策落定：ICRA 2026 紧凑版 + 4×A5000 立即可用**）
> 全局分析见 `docs/experiment_sanity_review.md`（为什么这样排序）。

---

## Scope A 声明（ICRA 2026，deadline ~9 月 15，剩 6 周）

**论文 claim 收缩为四件已验证/可验证的事**：
1. 时间路由信号系统性反转（已验证，§IV.I）
2. 空间连通性路由（合成已验证 d=+1.89；DCS 待验证 ← W1 关键）
3. Diagnostic Protocol（已验证，§IV.G.0）
4. **Q2 背景鲁棒性**（DCS，待跑 ← 主实验）+ Meta-World 1-2 任务示 robotics 相关性

**砍掉/降级**（写 future work，期刊版再补）：
真机三任务、slot-swap augmentation (Q3)、safety gate (Q5)、
7-baseline 完整矩阵（保 TD-MPC2-pixel 主对比 + DrQ-v2 一个 model-free 代表）。

**算力排布**：4×A5000 跑长 run（500K baseline、Q2 矩阵挂机）；
本地 5080 跑短实验（Stage-1、检查、E2E-0 调通）。

---

## Week 1（现在）— 生死检查 + 管线并行

### W1.1 🔴 最优先：Stage-1 on DCS + 连通性检查（1 天，5080）

**这是唯一能杀死故事的实验，先跑。**

```
1. 装 dm_control + distracting_control + DAVIS（PHASE2_PLAN §2）
2. walker-walk clean + easy 各采 25K 帧（random policy，含 qpos/qvel，clip 结构）
3. Stage-1 定型配方训练（connectivity 路由）
4. 检查（无 GT objectness，用 proxy）：
   a. 渲染 slot alpha 图（规则4：先看数据）——walker 躯干/四肢是否有专属 slot
   b. walker 区域 proxy 掩膜：clean 背景下 walker 是唯一运动源，
      用帧差 + 形态学闭运算得 walker 掩膜（clean 上可靠；easy 上不用）
   c. router fast slots 对 walker 掩膜的富集度
```

| 判据 | 通过 | 失败 |
|---|---|---|
| walker 富集 ≥ 3×（fast slots） | → W1.2 继续 | 停，带 slot 图回 cowork |
| 视频背景（easy）被判 slow 为主 | 记录数字进论文 | 若背景紧凑物被判 fast，如实记录（这本身是 §V 已预告的 failure mode，量化它） |

### W1.2 并行挂机（A5000 ×2 卡）：TD-MPC2-pixel baseline

walker-walk + cheetah-run，clean + distracting-easy，500K steps × 3 seeds
= 12 runs，2 卡 ~4 天。判据：clean walker 500K return 650-750（对齐官方）。
**这些数字直接进 Table V。**

### W1.3 并行（A5000 ×1 卡）：DrQ-v2 baseline 同矩阵

官方实现，同 12 runs。作为 model-free 代表。

---

## Week 2 — 最小端到端

### W2.1 E2E-0：最小可行 HippoAct（5080 调通 → A5000 跑）

**一次一个变量：z = flatten(S_fg) ⊕ q_t，无 binding transformer、无辅助 loss。**

```
adapter 只做：Stage-1 ckpt 加载 → encode_frame → argmax 路由 → fast slots
→ flatten ⊕ qpos/qvel → MLP → z (256)
```

判据：walker-walk easy 500K return ≥ 0.8 × TD-MPC2-pixel。
过 → W2.2；不过 → 查 adapter（对照清单在 PHASE2_PLAN §6），仍不过带数字回 cowork。

### W2.2 E2E-1：+ Binding Transformer（c_t，无 L_pred/L_align）
### W2.3 E2E-2：+ L_pred + L_align（完整方法）

每级 vs 前级 = build-up ablation，直接填 Table IX 的 A5/A7 行。
E2E-1/2 若无增益也如实报告（§IV.I 有先例，reviewer 吃这套诚实）。

---

## Week 3-4 — Q2 主矩阵（A5000 挂满）

**论文主图**：{walker-walk, cheetah-run, hopper-hop} × {HippoAct(最优级),
TD-MPC2-pixel, DrQ-v2} × 3 seeds，easy 训练 → {none, easy, hard} zero-shot 评测。

= 27 训练 runs（12 已在 W1 完成）+ 评测。4 卡 ~1.5 周。

判据（论文成立线）：HippoAct retention(hard/none) 显著高于两个 baseline
（目标 ≥0.65 vs pixel ~0.45-0.50；参照 Table V 的 ○ 估值）。

**每个 run 启动前**：预期数字写进 TODO_RESULT 预注册段（新纪律）。

---

## Week 4-5 — Robotics 相关性 + 补充

### W4.1 Meta-World：pick-place-v2 + drawer-open-v2（A5000）
Stage-1 on MW 帧 → E2E 最优配置 vs TD-MPC2-pixel，1M steps × 3 seeds。
判据：成功率 ≥ 0.9× pixel（打平即可，叙事是"表征可迁移到操作任务"）。

### W4.2 补充 ablation（Table IX 剩余行，A6/A11 已由 synthetic 数据支撑）
A1（CNN 换 DINOv2）、A2（去 slot）在 walker-easy 上各 3 seed。

### W4.3 内存/延迟测量（Table X 实测替换估值）——半天

---

## Week 5-6 — 写作冲刺（cowork 主导）

- 全部 ⧫/○ 替换为实测；图定稿（Fig.4 slot alpha on DCS hard 是主视觉）
- §V limitation 补：真机/Q3/Q5 移入 future work 的措辞
- 内审 → 改稿 → 提交

---

## 失败预案

| 失败点 | 预案 |
|---|---|
| W1.1 连通性在 walker 不成立 | 停。故事重定位候选：(a) 只做操作类场景（MW/真机桌面，物体紧凑假设更合理）(b) connectivity+motion 混合信号。回 cowork 定 |
| W2.1 E2E-0 < 0.8× | 表征-RL 接口问题，逐项查 PHASE2_PLAN §6 清单；仍不过则 Q2 改用 frozen-encoder linear-probe 类评测降级论证 |
| Q2 retention 无显著差异 | 论文主图改为反转发现 + 协议（贡献 1-3 独立成立），Q2 如实报告为 mixed |
| 时间不够 | 按 experiment_sanity_review §3 优先级表从底往上砍 |

---

## 常备规则（v5，新增第 7 条）

1. 训练前先量数据
2. 单变量原则（E2E 三级递进是其体现）
3. 判据前置
4. 先渲染数据再信度量
5. 真值来自生成过程，不从观测反推
6. 卡住带数据回 cowork
7. **GPU-天 > 1 的 run，启动前在 TODO_RESULT 写预注册段（预期数字+判据）**

## 汇报格式（不变）

```
Step N 结果：判据对照表 / 关键数字 / 异常观察 / 卡点
```
