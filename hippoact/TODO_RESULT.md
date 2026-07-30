# TODO Step 1 执行结果 — 三个 eval-only probe

> 执行环境：RTX 5080 (sm_120), torch 2.11.0+cu128, Python 3.10.12
> 数据：`data/frames/synthetic`，1250 clips × 4 帧，`disk_speed=12`（位移 2.28 patch）
> 度量：物体掩膜用**交集**定义 `motion(t-1,t) ∩ motion(t,t+1)`（TODO 规则 4）
> 脚本：`scratchpad/step1_probes.py`（P1/P2/P3 一次跑完，含并集/交集双报）
> 结论：**P1 第二档 + P2 第三档，按规则停，Step 2 未启动**

---

## 1. 判据对照表

| Probe | TODO 判据 | 实测 | 结论 |
|---|---|---|---|
| **P1** fresh init | on-object 恢复 ~0.4–0.5 → 权重没坏；仍 ~0.2 → 停 | **0.189**（dim192: 0.251） | ❌ 仍 ~0.2 → **停** |
| **P2** 矩匹配 | on-object ≥0.4 **且** tracking ≥0.6 → Step 2 | on-object **0.215**，tracking 0.538 | ❌ on-object 不恢复 → **停** |
| **P3** iters sweep | 仅参考，不阻塞 | 0.196 / 0.219 / 0.245 | 单调上升但封顶 |

---

## 2. 完整数据

三个 encoder × 三种 eval 模式。`clips_used=46`（见 §4 异常观察 1）。

| encoder | eval 模式 | on-object(交集) | on-object(并集) | tracking |
|---|---|---|---|---|
| **shared-init 训练（参照系）** | fresh init | **0.510** | 0.723 | 0.349 |
| | 矩匹配 | 0.501 | 0.698 | 0.355 |
| | raw carryover | **0.469** | 0.641 | 0.359 |
| **carryover 训练 iters3/dim128** | raw carryover | 0.217 | 0.310 | 0.506 |
| | fresh init | **0.189** | 0.274 | 0.115 |
| | 矩匹配 | 0.215 | 0.293 | 0.538 |
| **carryover 训练 iters5/dim192 @step4000** | raw carryover | 0.209 | 0.283 | 0.578 |
| | fresh init | 0.251 | 0.312 | 0.135 |
| | 矩匹配 | 0.209 | 0.281 | **0.604** |

### P3 测试时 iters sweep（fresh init，on-object 交集）

| encoder | iters=3 | iters=5 | iters=8 |
|---|---|---|---|
| shared-init 训练（参照） | 0.510 | 0.478 | 0.510 |
| carryover iters3/dim128 | 0.196 | 0.219 | 0.245 |
| carryover iters5/dim192 | 0.205 | 0.217 | 0.224 |

**判据公平性**：参照 encoder 在同一脚本、同一严格掩膜下达到 0.510，说明「恢复到 0.4–0.5」是可达的，门槛没有定高。

---

## 3. 决定性观察：损害在权重里，不在 init 分布

```
shared 训练   + carryover eval :  定位 0.469 (好)   跟踪 0.359 (差)
carryover 训练 + carryover eval :  定位 0.217 (坏)   跟踪 0.506 (好)
carryover 训练 + fresh eval     :  定位 0.189 (坏)   跟踪 0.115 (差)
```

参照 encoder **用 carryover init 去 eval，定位几乎不掉**（0.469 vs 自身 0.510）。
而 carryover **训练**出来的 encoder，**喂 fresh init 也只有 0.189**。

→ 这不是推理时的 init 分布错配。carryover 训练把定位能力**从权重里换掉了**，
换来了跟踪。任何 init 侧的修法都救不回来。这正是 P1 要区分的事，答案是「权重坏了」。

### P3 独立佐证同一结论

- 参照 encoder 对迭代数**不敏感**（0.510 / 0.478 / 0.510）→ 3 次迭代已收敛
- carryover encoder 随迭代**单调上升但封顶 0.245**，远低于 0.510

如果只是迭代预算不足，加到 8 次应当追上参照——没有追上。所以是权重，不是预算。

### 矩匹配几乎是 no-op

| encoder | raw carryover | 矩匹配 | 差值 |
|---|---|---|---|
| iters3/dim128 on-object | 0.217 | 0.215 | −0.002 |
| iters5/dim192 on-object | 0.209 | 0.209 | 0.000 |
| iters3/dim128 tracking | 0.506 | 0.538 | +0.032 |
| iters5/dim192 tracking | 0.578 | 0.604 | +0.026 |

init 流形的一二阶矩**不是**阻碍定位的因素。P2 的假设被干净地否证：
矩匹配只对 tracking 有约 +0.03 的边际影响，对 localization 完全无效。

---

## 4. 异常观察

1. **有效样本仅 46 / 150 clips。** 严格掩膜要求 `d01 ∩ d12` 与 `d12 \ d01` 各 ≥30 像素；
   `disk_speed=12` 位移偏大（2.28 patch），多数圆盘的交集过小被滤掉。上表数字因此是
   「每 clip 单物体」统计，置信区间偏宽。收紧需把 `disk_speed` 降到位移略小于圆盘直径
   （约 8–16 px），但那会靠近 sub-patch 量化风险（CP5d 的坑）。**这是个需要决策的取舍。**

2. **tracking 与 localization 在三个 encoder 上严格反向**，没有任何配置同时拿到两者：

   | | localization | tracking |
   |---|---|---|
   | shared | 0.510 | 0.349 |
   | carryover dim128 | 0.217 | 0.506 |
   | carryover dim192 | 0.209 | 0.578 |

   这比「部分成立」更像一个**守恒关系**，值得作为 §IV.I negative result 的核心论据。

3. 本轮未测 `Cohen's d`（P1–P3 未要求）。最近一次为 CP5e 的 **−0.71**；
   `corr(content diff, motion)` 最近一次 **−0.379**。

---

## 5. 前序实验补充（本轮之前，同一诊断链）

### slot_iters / slot_dim 两个 lever 的完整轨迹

| 配置 | 峰值 tracking | 峰值 step | final tracking |
|---|---|---|---|
| carryover, iters=3, dim=128 | 0.654 | 8000 | 0.568 |
| carryover, iters=5, dim=128 | 0.665 | 6000 | 0.506 |
| carryover, iters=5, dim=192 | **0.724** | **4000** | 0.462 |

**容量提升峰值（0.654 → 0.724）但让峰值提前、final 更差。**
峰值 step 单调前移 8000 → 6000 → 4000 是「优化动力学」而非「容量不足」的签名：
容量让绑定形成更快更好，也更快被重建损失覆盖。

**跨帧绑定是训练瞬态**——三个独立配置复现。含义：Stage-1 不能用 `L_slot` 收敛
判断「训好了」，否则系统性挑到绑定最差的 checkpoint。

### 一处需要修正的先前结论

先前报告「carryover 让定位从 0.501 掉到 0.201」使用的是**并集**掩膜。
换严格交集掩膜后，按「每个物体上有几个 slot」重测：

| | 每物体 slot 数 | 这些 slot 两两 IoU |
|---|---|---|
| shared init | **4.33** | 0.399 |
| carryover dim192 @4000 | **2.00** | 0.691 |

理想是 1 slot = 1 物体。shared 的高 on-object 比例部分来自**冗余**（4.33 个 slot
压在同一物体上）。所以「carryover 以定位换绑定」这句话需要限定——carryover 的
slot/物体比更接近理想，但那 2 个 slot 彼此 IoU 0.691（近似重复）。两种模式都
未达 1:1，冗余形态不同。

### 一处会印进论文的笔误

`docs/paper_section_IV_experiments.md` §IV.E.1 写「196 patch grid」，
但 DINOv2 ViT-S/14 在 224 输入下是 **16×16 = 256**。
`configs/default.yaml` 的同一处注释已在 cd4e76b 修正，docs 里这处仍在。
`viz.py` 当初的 AssertionError 正是这个数导致的，不是无害笔误。

---

## 6. 卡在哪 / 下一步

**卡住的判据**
- P1 第二档：`仍 ~0.2 → carryover 训练损害了编码器 → 停`
- P2 第三档：`on-object 不恢复 → init 流形比一二阶矩复杂 → 停，SAVi-lite predictor 由 cowork 出`

**Step 2（`carryover_norm` 训练）未启动。** 理由：P2 已证明矩匹配在 eval 上是 no-op
（on-object 差值 0.000 / −0.002），拿它训 40 分钟大概率复现同样的定位数字。
若仍需训练动态下的数据（训练时行为可能异于 eval），可另行指示。

**执行侧判断**（不作为方法决策，仅供参考）：P1/P2/P3 三条合起来指向
**SAVi-lite predictor**——让跨帧身份由显式转移模块承载，而不是靠「上一帧 slot 当 init」
去挤占 Slot Attention 自身的定位机制。与 TODO 中写的下一步一致。

---

## 附：可复用诊断脚本（scratchpad）

| 脚本 | 用途 |
|---|---|
| `step1_probes.py` | 本轮 P1/P2/P3 一次跑完，交集/并集双报 |
| `track_check.py` | 无歧义跟踪判据（交集掩膜 + 新位置增益） |
| `redundancy_check.py` | 每物体 slot 数 + 两两 IoU，判断 on-object 高是否只是冗余 |
| `binding_check.py` | centroid shift + NN 匹配恒等比例 |
| `anchor_check.py` | 内容锚定 vs 位置锚定 |
| `target_compare.py` | 候选 target（content diff / centroid shift / alpha L1）对运动的预测力 |
| `validate_semantics.py` | Cohen's d + oracle(18.62)/uniform(1.00) 正负对照 |
| `savi_probe.py` | random / shared / carryover 三 regime 对比 |
| `decomp_check.py` | 场景分割质量（归属熵、IoU、空间铺展、前后景覆盖） |
| `inversion_check.py` | 跟踪型 vs 静止型 slot 的 content diff 对比 |
| `ab_eval.py` | SNR 分解 + Cohen's d + gumbel/argmax gap |
