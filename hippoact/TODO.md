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

## ⚡ 算力切换 + 插队指令（2026-08-01 更新：全量 4×A5000，5080 退役）

### 迁移注意（一次性）
- A5000 是 sm_86：**重建 venv**，torch 用常规 cu121/cu124 wheel 即可
  （5080 上被迫用的 torch 2.11+cu128 不必带过去；但 dm_control 最新版 +
  mujoco 3.11 + distracting_control 补丁三件套照搬 PHASE2_PLAN §2）
- **Stage-1 checkpoint 直接迁移**：torch ckpt 跨架构兼容，W1.1 训好的
  DCS Stage-1 ckpt scp 过去即可，不用重训
- 迁移后先跑 `pytest tests/ -v` + `scripts/overfit_test.py` 确认环境等价

### 4 卡排布

| 卡 | 任务 | 时长 |
|---|---|---|
| gpu0 | **W2.1 E2E-0** walker easy ×1 seed（多点位判据：50K/100K/250K/500K 各 ≥0.8× pixel 同点位） | ~2 天 |
| gpu1 | **零样本 retention 评测**：12 个 pixel ckpt 对 {none, hard} 纯 eval → 完成后接 cheetah clean s1 重跑 | 数小时 + 9h |
| gpu2 | cheetah clean s2 重跑 → 完成后接 **W1.3 DrQ-v2** 矩阵起步 | 9h + 后续 |
| gpu3 | **W1.3 DrQ-v2** walker/cheetah × clean/easy × 3 seed 起步 | ~4 天 |

E2E-0 判据通过 → gpu0 立刻接 E2E-0 补 2 个 seed + E2E-1。

## ⚡ R1/R2/R4 拍板结果（2026-08-02，cowork）

1. **R1 指标定义（已定）**：Q2 采用双比值，均以训练分布为锚 —
   escalation retention = R_hard/R_easy；background-presence invariance
   = R_none/R_easy。none 崩塌现象本身写入论文（pixel 把背景存在性学进
   表征）并升级为可测预测：HippoAct 应三分布近平坦，pixel 实测
   invariance 0.65/0.30。§IV.C 已改。
2. **R2 判据（已定）**：P2.2b 的 480 单点阈值作废（sd≈140 下无统计意义，
   判据修订透明记录）。cheetah clean 报全量 5-run 445±139，与官方
   537±74 做 Welch（t≈1.2, n.s.）→ 管线与官方无显著差异。固定 seed
   不可复现现象写入 §IV.A。加 seed 至 8 = 低优先级排队项。
3. **R4（先试零成本解法）**：
   a. **先试从 5080 磁盘 scp W1.1 checkpoint**（退役 ≠ 磁盘不可访问，
      几分钟即可消除全部歧义）。拿到后用 W1.1 ckpt 起 E2E-0。
   b. 5080 确实不可访问 → 用重训 ckpt_final 起 E2E-0（判据 0.8× 不变），
      富集差距未必传导到 return；E2E-0 过 → 歧义不 material；
      不过 → 起 3-seed Stage-1 方差实验（~22h）判断 W1.1 是否幸运抽样。
   c. 代码已修：Stage-1 ckpt 现在自带完整 config 快照（trainer_config +
      encoder_arch 从模块实读，不可漂移）。以后不再有"配置没留档"。
4. **R4.5 三件集成事项**按 agent 清单执行：encoder_type 分发接线、
   224² ImageNet-norm obs wrapper、MPPI 单次调用 smoke test。
   三件齐 → E2E-0 起跑（多点位判据不变）。

## ⚡ R4.5–R4.7 / R3.2 拍板（2026-08-02，cowork）

1. **E2E-0 三 encoder 设计批准，立即重启**（gpu0/1/2，Stage-1 seed 1/3/5，
   env 前移快路径，~5.5h/run）。多点位判据不变（R4.7.3 的表）。
   完成后**立刻接零成本 Q2 评测**：三个 500K ckpt 对 {none, easy, hard}
   纯 eval → flat-invariance 预测（R1 新轴）的第一次检验。
   预注册：HippoAct invariance (none/easy) 应显著高于 pixel 的 0.65/0.30；
   escalation retention (hard/easy) 应高于 pixel 的 0.74/0.49。
2. **slot_init_seed 定性采纳**：init 抽样是编码器身份的一部分，随 ckpt
   报告。已写入论文 §IV.I。E2E 全部 run 记录所用 slot_init_seed。
3. **R4.7.2 遵守**：论文引用 easy Cohen's d 用 5-seed 的 1.46±0.13，
   不用 W1.1 单点 1.844。W1.1 数字保留在 §III.D.3 的富集度（在分布内）。
4. **R3.2 双峰报告方式采纳**（学会率 + 学会者分数，已写入 §IV.A Metrics）。
   **DrQ-v2 clean walker 2/6 失败先排查再披露**：半小时 diff 官方
   walker_walk config（num_train_frames 1M vs 1.1M 线索）。查出 → 修 +
   重跑失败的 2 个 clean seed；查不出 → 如实披露 + 列出已排查项。
5. **低优先级队列**（卡空时）：cheetah clean 补 seed 至 8；
   walker easy seed 7-9 跑完让 Fisher 检验收口。

## ⚡ R4.8 / R3.3 拍板（2026-08-04，cowork）—— proprio 分场景协议

**架构澄清（本轮核心）**：proprio 通道的正当性取决于任务类型——
- **Locomotion (DCS)**：proprio = 全状态 → **作弊**。Q2 全部方法纯视觉
  （含我们，`hippoact_include_proprio=false`）。这是 DCS 文献标准做法。
- **Manipulation (Meta-World)**：proprio 只含手臂，物体位置必须从视觉读
  → **正当**。两边都给 vision+proprio，公平。binding transformer +
  proprio 融合在这里才被正当检验。

**因此 Meta-World 从"点缀"升级为完整方法的主战场**（Week 4 权重上调）。

执行队列：
1. `e2e0vis_s{1,3,5}` + `proprio_only` 跑完（在跑）→ 判据对照
   （预注册 pixel 同点位判据沿用；带 proprio 的三个 run 达标作废，
   保留作混淆证据）
2. e2e0vis 过判据 → 三个 500K ckpt 的 {none,easy,hard} invariance 评测
   （flat-invariance 预测，预注册目标不变）
3. proprio_only 数字进论文作 context row（大多数 DCS 研究不报这个，
   报了是加分）
4. cheetah 的 E2E 同样走纯视觉协议
5. **Meta-World 准备工作提前启动**（与 DCS 收尾并行）：
   MW 视觉版环境调研（obs 里物体位姿必须不可见）、Stage-1 on MW 帧采集
   计划、TD-MPC2 的 MW 配置确认。先出一页 spec 再动手。

**R3.3 采纳**：DrQ-v2 clean 失败 = 官方已知性质（其曲线 2/10 同形），
官方 10-seed 并入基线后干扰效应显著（25%→78%, Fisher p=0.016）。
论文引官方曲线，无需排查清单。

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
