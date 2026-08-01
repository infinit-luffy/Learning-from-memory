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

# TODO Step M1 执行结果 — Tracking-by-Matching probe

> 数据：`data/frames/synthetic_diag`，400 clips × 4 帧，`--min-radius 14 --max-radius 26`
> 有效样本 **393/400 = 98.2%**（旧诊断集为 46/150 = 30.7%，样本瓶颈已解决）
> checkpoint：shared-init 训练版（定位 0.510 那个），**编码器完全未改动**
> 每帧独立 fresh init（不共享，避免冻结分区），跨帧 identity 由事后 Hungarian 恢复
> 物体掩膜两侧都用交集：`obj1 = d01 ∩ d12`，`obj2 = d12 ∩ d23`（4 帧 clip 使之可行）
> 脚本：`m1_matching.py`（首轮）、`m1_sweep.py`（一轮 sweep）、`m1_oracle.py`（天花板）
> 结论：**matched tracking 0.564 < 0.60，一轮重测后仍不过，按中间档规则停**

---

## M1.1 判据对照表

| 指标 | TODO 判据 | 首轮 | sweep 最佳 | 结论 |
|---|---|---|---|---|
| localization | ≥ 0.45 | **0.521** | 0.518 | ✅ 过（与参照 0.510 一致，证实编码器未动） |
| matched tracking | ≥ 0.60 → M2 | **0.556** | **0.564** | ⚠️ 落 0.45–0.60 中间档 |
| assignment 稳定性 | 顺带记录 | 0.777 | 0.774 | 良好 |

中间档授权「调 w_iou/w_feat 和 IoU 二值化阈值，一轮内重测」已执行（20 组合），
仍不过 → 停。

## M1.2 关键数字

```
localization (on-object 交集)   0.518        [参照 shared-init 0.510]
matched tracking 最佳           0.564        w=(0.5,0.5), topk=64
assignment 稳定性               0.774
经验地板 identity / random      0.387 / 0.398   <- 不是 0.5
双射可达上界 (Hungarian oracle) 0.837
非双射硬上界 (greedy oracle)    0.930
归一化捕获率                    0.379
相邻帧 alpha 分解最优对齐余弦    0.689        <- 不是 1.0
```

## M1.3 超参 sweep 全表（clips_used=390）

| (w_iou, w_feat) | top8 | top16 | top32 | top64 |
|---|---|---|---|---|
| (1.0, 0.0) | 0.528 | 0.538 | 0.539 | 0.542 |
| (0.7, 0.3) | 0.537 | 0.549 | 0.553 | 0.558 |
| (0.5, 0.5) | 0.544 | 0.556 | 0.559 | **0.564** |
| (0.3, 0.7) | 0.549 | 0.560 | 0.559 | 0.561 |
| (0.0, 1.0) | 0.560 | 0.560 | 0.560 | 0.560 |

跨度仅 **0.036**；纯 IoU 与纯 cosine 两个极端相差不到 0.02。
→ **上限不在代价函数里**，继续调匹配超参无收益。

## M1.4 两个决定 M2 走向的发现

### (1) 天花板本身是 0.837，不是 1.0

双射 oracle（用真值挑最优配对）0.837，greedy 非双射硬上界 0.930。
即使给出**完美匹配**，仍有约 16% 的 case 找不到一个「在 obj2 上质量更高」的 slot。
这部分不是匹配算法能解决的——是分解里根本没有对应的 slot。

### (2) 相邻帧的分解不是同一套划分：最优对齐余弦 0.689

Hungarian 的前提是「两帧的 slot 集合是同一批实体的两次呈现」。0.689 说明前提
**部分不成立**：fresh init 下两帧各自收敛到不同划分。这与 CP5e 测到的
「同图换 init → centroid 移 4.7 patch、alpha 余弦 0.189」是同一现象的轻量版
（同一 init 分布下不同抽样导致的划分漂移）。

### 0.564 的构成分解

```
0.387  经验地板 (identity/random)
+0.171  匹配实际捕获        (可达空间 0.450 的 37.9%)
------
 0.558  best_learned
 0.273  剩余, 被分解不稳定性锁住 (天花板 0.837 之下)
```

要过 0.60 需把捕获率从 0.379 提到 0.473。缺口 0.036 在「提高捕获率」内够拿，
但 sweep 显示收益递减的位置正在此附近。

## M1.5 异常观察

1. **我自己的 oracle 实现最初有 bug，已修正。** 代价矩阵写成 `mass2[j] − base_i`
   是**可加分离**的：任何双射的总和恒等于 `Σmass2 − Σbase`，Hungarian 退化成
   任意分配，得到 oracle（0.351）**低于**实测（0.558）的自相矛盾。改为「在被选中
   的行上最大化 gain>0 的计数」后得到正确的 0.837。
   → 「上界低于实测」是发现度量 bug 的强信号，建议写进 Diagnostic Protocol。

2. **经验地板是 0.387/0.398，不是 0.5。** 被选中的 slot 本身已在 obj1 上，与 obj2
   邻近，所以任意重配对的期望增益为负。TODO 的 0.60 绝对门槛相对于这个地板的
   位置与设计意图可能不同——归一化捕获率（0.379）比绝对值更能反映匹配质量。

3. localization 0.521 vs 参照 0.510 的微小差异来自诊断集不同（大圆盘），
   非编码器变化。

## M1.6 卡在哪 / 执行侧观察

**卡在 matched tracking ≥ 0.60，差 0.036。**

不作为方法决策、仅供参考：缺口的性质已经明确——**不是匹配算法不够好，
而是相邻帧的分解不够一致（0.689）**。三个方向的性质不同：

- **提高分解一致性**（让两帧收敛到更接近的划分）能同时抬高天花板（0.837）
  和捕获率（0.379），是唯一动到根因的方向
- 继续调匹配代价函数：sweep 已证明收益 < 0.04，不够
- SAVi-lite predictor：回到「让网络学 identity」，Step 1 已证明该路径损害权重

M2 未启动，未改动任何代码 / loss / 判据。

---

# 精确真值重测 — 推翻三个旧结论，第一次可信地测到 P1 的真实问题

> 触发：把合成场景画出来后发现，此前所有的物体掩膜都是从像素差分反推的，
> 而圆盘位移（15–20px）小于直径（28–52px），`motion(t−1,t) ∩ motion(t,t+1)`
> 退化成**圆盘边缘的细碎条带**——中位数 670 px，p10 仅 141 px，对照圆盘面积
> 616–2124 px。视觉确认见 §附图说明。
> 修法：生成器新增 `--save-annotations`，落盘每帧每个圆盘的精确 `(cx, cy, r)`。
> 真值掩膜覆盖 93.6% 的实际变化像素，面积 3775 px。
> 白拿的好处：annotations 中圆盘逐帧同序索引，**物体身份跨帧已知**，
> 「跟踪」从代理指标变为直接测量（随机基线 1/16 = 0.0625，而非 0.5）。
> 脚本：`tools/diagnostics/gt_eval.py` / `gt_router.py` / `gt_target.py`

## GT.1 定位与跨帧身份（gt_eval.py，400 clips，3.5 物体/clip）

| encoder | 模式 | 富集倍数 | coverage | identity 持续性 | exclusivity |
|---|---|---|---|---|---|
| shared 训练 | shared | 18.16 | 1.000 | 0.415 | 0.420 |
| shared 训练 | fresh + Hungarian | 18.20 | 1.000 | 0.345 | 0.421 |
| carryover iters3/dim128 | carryover | 15.39 | 1.000 | **0.693** | 0.357 |
| carryover iters5/dim192 @4000 | carryover | 10.74 | 0.999 | 0.659 | 0.254 |
| carryover iters5/dim192 final | carryover | **18.12** | 1.000 | 0.627 | 0.417 |

（identity 随机基线 = 1/16 = 0.0625）

## GT.2 三个旧结论被推翻

**(1) 定位其实是解决了的。** 全部 5 个 checkpoint 的 coverage = 1.000，
每个圆盘都有 slot 盖住，富集 10–18 倍。此前「on-object 0.19–0.51」量的是边缘碎片。

**(2) 「tracking / localization 权重守恒」不成立。**
`dim192 final` 富集 18.12、exclusivity 0.417（与 shared 的 18.16 / 0.420 持平），
identity 0.627 vs shared 0.415。两个维度都不差，不存在取舍。
→ §IV.I 原计划引用的守恒关系**不能写**。

**(3) 「绑定是训练瞬态」不成立。**
旧指标：dim192 峰值 0.724@4000 → final 0.462（崩）。
精确真值：0.659@4000 → 0.627 final（降 0.03），同期富集 10.74 → 18.12、
exclusivity 0.254 → 0.417。**继续训练在改善，不是在毁掉绑定。**
→ 「checkpoint 必须按 tracking 指标选而非 L_slot」这条常备规则失去依据。

## GT.3 P1 核心测量：router 系统性反转（gt_router.py）

度量对照通过：oracle slot objectness = 11.13，uniform slot = 1.00。

| encoder | Cohen's d | 秩 AUC | FAST slots objectness | SLOW slots objectness |
|---|---|---|---|---|
| shared 训练 | **−1.248** | 0.204 | 1.31（≈chance） | 4.17 |
| carryover 训练 | **−1.252** | 0.253 | 0.19（远低于 chance） | 2.85 |

router 的 P(fast)：真物体 slot 上 0.09–0.12，非物体 slot 上 0.14–0.53。

**router 把物体 slot 判成「慢」、背景 slot 判成「快」，|d| ≈ 1.25。**
强信号，方向反了，不是噪声。

### 机制（比先前的「概念漏洞」更一般）

背景占 95% 像素、无锚点，slot 在其上漂移；物体 slot 被物体锚住反而稳定。
因此**任何基于「slot 时间变化量」的指标都会优先挑出背景 slot**。
跟踪越成功，router 越错。

## GT.4 三个候选 target 全部无效（gt_target.py）

| 候选 | carryover pearson / top25% objectness | shared pearson / top25% |
|---|---|---|
| (a) content diff（现行 L_slow） | **−0.342** / 0.54 | +0.295 / 3.93 |
| (b) **centroid shift（M2 计划）** | **−0.111** / 1.01 | **+0.280** / 3.70 |
| (c) alpha IoU drop | −0.045 / 0.81 | +0.176 / 3.41 |
| *全体 objectness 均值* | *1.73* | *4.08* |

判读依据是最后一行：有效 fast 信号的 top25% objectness 应**显著高于**全体均值，
实际全部**等于或低于**。按这三者中任一取阈值，选出的 slot 都不比随机更像物体。

**这否决的不是某个实现，而是「用 slot 的时间变化量判断慢/快」这一整类信号，
包括 Step M2 计划中的 alpha displacement。M2 patch 不应实现。**

## GT.5 一个有效的非时间信号：alpha 空间紧凑度

物体 slot 空间紧凑，背景 slot 弥散。逐帧可测，**不需要跨帧身份**。

| 信号 | carryover spearman | shared spearman | carryover top25% objectness |
|---|---|---|---|
| **alpha 空间标准差（低=紧凑）** | **+0.652** | +0.269 | **4.26**（均值 1.75） |
| alpha 归一化熵（低=紧凑） | +0.568 | +0.166 | 3.90 |
| alpha 峰值富集（高=紧凑） | +0.208 | +0.069 | 2.51 |

+0.652 对比时间类信号最好的 −0.342：方向正确且强度约两倍。
top25% objectness 4.26 vs 均值 1.75 = **2.4 倍**，是唯一能筛出物体 slot 的信号。

### 证伪测试：它是否只是在测「物体小」

场景中物体恰好都小（半径 14–26 = 2–4 patch 直径）而背景占 95%，
所以「紧凑 = 物体」可能是场景构造的产物。用 `--min-radius 8 --max-radius 60`
重新生成（881 物体，直径 1.1–8.4 patch）后按半径分箱：

| 半径箱 | 物体数 | 富集倍数 | owner 的紧凑度排名 |
|---|---|---|---|
| 8–16 px | 148 | 21.35 | 0.833 |
| 16–24 px | 123 | 16.37 | 0.774 |
| 24–32 px | 144 | 12.06 | 0.725 |
| 32–44 px | 208 | 8.93 | 0.684 |
| 44–60 px | 258 | 5.86 | 0.699 |

`corr(物体半径, owner 紧凑度排名) = −0.247`

**结论：存在真实的尺寸混淆，但信号存活。** 即使直径 8.4 patch 的最大物体，
其 owner slot 的紧凑度排名仍为 0.699，明显高于 0.5 基线。
**必须披露**：真机场景中大件物体（抽屉、箱体）会被弱化，
而只覆盖桌面局部的 slot 会被误判为紧凑。

## GT.6 对论文的影响

- **§III 的设计前提有逻辑问题**：「按时间不变性路由慢/快」在 slot 跟踪成功时失效——
  时间不变性成了「跟踪成功」的标志，而非「世界中该物体不动」的标志
- **§IV.I 不能再引用**守恒关系与绑定瞬态（GT.2 已推翻）
- **Table VII purity** 对应本文的 exclusivity = 0.42，低于论文声称的 0.86，仍有差距；
  但 coverage 1.000 与富集 18× 是强结果，可直接引用
- 紧凑度路线若成立，可**去掉整条跨帧身份依赖**（carryover / 匹配 / shared init
  的复杂度），前面七层中相当一部分是在为该前提搭地基

## GT.7 方法学：这次是怎么发现的

前七层每一层都在「测出的数不对 → 往上游找一层原因 → 修 → 再测」的循环里，
但从未回头验证**场景与度量本身能否回答论文的问题**。直到把数据渲染出来看。

新增两条应写入 Diagnostic Protocol 的规则：

1. **先看数据本身**。任何度量投产前，把输入和真值叠加渲染出来目视检查一遍。
   碎片掩膜在数值上完全「合理」（面积 670 px、有效样本 98%），只有画出来才暴露。
2. **真值优先用生成过程的已知量，不要从观测反推**。生成器知道精确物体位置，
   而从像素差分反推引入了一整类可以完全避免的失效模式。

另：`docs/paper_abstract_contributions.md` 明确声明「trained without segmentation
supervision」。本文真值仅用于**评估**，不进 loss，与该声明不冲突。

---

# CP6 — 路由信号改为 alpha 空间连通性：P1 机制成立

> 决策链：GT.4 否掉整类「slot 时间变化量」信号（含 Step M2 计划的 centroid shift）。
> GT.5 发现空间紧凑度可用但有尺寸混淆。本节把它做成尺度不变的可训练信号并验证。
> 新增 `loss.slow_signal: {content_diff, alpha_connectivity}`，**默认仍为 content_diff**，
> 新路径可选可回退。19 个测试全过。

## CP6.1 为什么选连通性而不是像素运动

两个候选都能绕开「slot 时间变化量」的失效：

| 候选 | spearman(objectness) | 尺寸混淆 | 在 Q2（DCS 视频背景）下 |
|---|---|---|---|
| pixel motion in footprint | **+0.727** | −0.475 | ❌ **崩** —— 视频背景本身在动，会被判 fast |
| alpha 空间连通性 | +0.623 | **−0.063** | ✅ 不看运动，背景动不动无所谓 |

选连通性的主要理由不是那 0.1 的相关系数，而是**它恰好在论文最难的 Q2 场景下稳健**。
pixel motion 保留作 ablation 对照。

## CP6.2 实现：neighbor coherence（GPU，连续，尺度不变）

`slot_neighbor_coherence`：top-1/16 的 alpha 二值化后，统计这些 patch 的 4-邻居
中有多少比例也在集合内。一次 3×3 卷积，无 Python 循环。

| 实现 | spearman | top25% obj (均值 1.39) | 并列率 | 尺寸混淆 |
|---|---|---|---|---|
| 连通分量占比 (scipy) | +0.578 | 2.28 | 0.519 | −0.005 |
| **neighbor coherence (GPU)** | **+0.623** | **2.44** | **0.177** | −0.063 |

两者一致性 +0.812。GPU 版更强、并列率低三倍。自检：3×3 连通块 coherence 0.800，
散点 0.000。

## CP6.3 训练前置检查（两条都过才动代码）

- 信号对 objectness 的 spearman = **+0.623**
- 信号能否从 slot 向量学到：线性探针 **AUC 0.966**
  （对照上界：objectness 本身 AUC 0.956 —— 基本触顶）

## CP6.4 结果：Cohen's d 从 −1.25 翻到 +1.89

配置刻意退回最简：`slot_iters 3` / `slot_dim 128` / `slot_init_mode shared` /
`lambda_slow 0.5`，数据 `synth_gt`（400 clips，带精确 annotations）。

```
Cohen's d:  −1.248  →  +1.889        秩 AUC:  0.204  →  0.925
FAST slots objectness  4.44 (中位数 4.24)
SLOW slots objectness  0.09 (中位数 0.01)      <- 几乎纯背景
router P(fast): 真物体 slot 0.847 / 非物体 slot 0.541
```

### 定位未被牺牲，反而是历次最好

| | 富集倍数 | coverage | identity | exclusivity |
|---|---|---|---|---|
| shared 训练（此前最好） | 18.16 | 1.000 | 0.415 | 0.420 |
| **connectivity 路由** | **18.78** | 1.000 | **0.466** | **0.430** |

三项均为历次最好。**不存在「修路由需牺牲定位」的取舍。** identity 0.466 是副产品，
本轮未为其做任何设计。

### router 超越了训练它的代理

```
纯 connectivity 阈值路由（代理上限）  d = +1.503
训练后的 router                      d = +1.855
corr(router P(fast), connectivity)   = +0.886
```

router 比代理高 0.35 个 d —— 它看的是 slot 向量（比 alpha 派生的代理更丰富），
连通性只是把它引到正确概念上。这说明方案不是「把手工特征硬塞进模型」。

## CP6.5 被一并简化掉的东西

连通性是**逐帧**信号，因此 carryover / Hungarian 匹配 / 跨帧 slot 身份**全部不需要**。

前七层中相当一部分工作（CP5b pair loader、CP5c shared init、CP5e-g carryover、
M1/M1b matching）都是在为「跨帧身份」这个前提搭地基。**该前提本身并非必要**，
它只是原设计用时间不变性做路由的副产物。

## CP6.6 三个必须记下的保留

1. **`route_prior_slow=0.7` 现已标定错误。** router 给出 41–44% slow，先验要 70%，
   `L_route` 升到 0.15–0.26（健康区间 [0.001, 0.1]）。两个损失在打架，调整先验
   应还有提升空间。
2. **identity persistence 0.466 仅是副产品。** 若 Stage 2 的 binding memory 需要
   稳定跨帧身份，此数不足，需单独设计。（TODO 中的判断是 Binding Transformer 为
   attention over set、对 slot index 不敏感，若成立则不需要。）
3. **必须在 Robosuite 上重验，不可直接写入论文。** 本信号在合成场景有效靠的是
   「物体紧凑、背景弥散」。真机中桌面局部的 slot 也可能紧凑，抽屉/箱体虽连通但
   形状复杂。宽半径证伪测试只排除了**尺寸**混淆（−0.063），**未排除形状混淆**。

## CP6.7 对论文的影响

- **§III 的路由设计需重写**：从「按时间不变性路由」改为「按空间连通性路由」，
  并给出前者失效的实证（GT.3 的 d = −1.25 + 机制解释）
- **§IV.I negative result 换素材**：守恒关系与绑定瞬态已被 GT.2 推翻，
  应替换为「时间不变性在 slot 跟踪成功时反转」这一更强的发现
- **Table VII purity** 对应 exclusivity = 0.430，仍低于论文声称的 0.86；
  但 coverage 1.000、富集 18.78、路由 d = +1.89 是可直接引用的强结果
- 方法可**大幅简化**：跨帧身份相关的全部机制可从 §III 移除

---

# W1.1 — Stage-1 on DCS：connectivity 路由在真实渲染场景上成立

> 「唯一能杀死故事的实验」。CP6 的 connectivity 路由是在合成圆盘上验证的，
> 那里我明确标注了「未排除形状混淆」——walker 是细长四肢结构、占画面 7.1%，
> 与紧凑圆盘差别很大，正是该风险的直接检验。
> 数据：walker-walk，clean + easy(DAVIS) 各 25K 帧，random policy，含 qpos/qvel。
> 配方沿用 CP6 定型：`slow_signal=alpha_connectivity` / `shared init` /
> `slot_iters 3` / `slot_dim 128`，clean+easy 合训（48000 对）。
> 脚本：`tools/dcs/{collect_frames,patch_distracting_control,walker_mask,walker_enrichment}.py`

## W1.1.1 判据：通过

| | fast slots 富集 | 判据 ≥3.0 | Cohen's d | 秩 AUC | slow slots 富集 |
|---|---|---|---|---|---|
| clean | **4.15×** | ✅ | +2.262 | 0.943 | 0.20 |
| easy | **4.26×** | ✅ | +1.844 | 0.833 | 0.33 |

对照通过：oracle slot 富集 9.54，uniform slot **1.00**（定义上必然为 1.00）。
walker 占画面 7.06%，故 4.15× 等价于 fast slots 有约 29% 的 alpha 质量落在 walker 上。

**形状混淆风险未兑现**：细长关节结构上 connectivity 依然有效。

## W1.1.2 第 4d 步：easy 背景的路由分布

```
router 判 fast 占比:  clean 0.677   easy 0.627
slow slots 富集度:    clean 0.20    easy 0.33
```

视频背景**主要被判 slow**，且是在背景贡献了 47.9% 像素变化的条件下做到的
（clean 仅 12.7%）。这是 Q2 的关键前提。

### 失败模式（§V 预告的那个，已量化）

easy 相对 clean：Cohen's d **2.262 → 1.844**（−0.42），AUC **0.943 → 0.833**，
slow slots 富集 **0.20 → 0.33**。背景中的紧凑物体确实会被误判 fast。
slot 图中可见（s12 的注意力块落在背景区）。**这组数字应写入 §V。**

## W1.1.3 一个直接支持 CP6 设计决定的实测

```
相邻帧变化像素占比:  clean 0.127   easy 0.479
```

CP6 中放弃 `pixel_motion`（合成上 +0.727，强于 connectivity 的 +0.623）而选
connectivity，理由是前者会在 DCS 视频背景上崩。**现在这是实测的**：easy 下
背景贡献近半像素变化，pixel_motion 路由会把背景整体判成 fast。

## W1.1.4 两处偏离原计划（均为改进）

**(1) 用 MuJoCo 分割渲染替代 proxy 掩膜。**
TODO 第 4b 步指定「clean 帧差 + 闭运算」。照做并渲染检查后发现它把**地板倒影
和一条横线**一并圈入。而 dm_control 原生支持分割渲染：

```
geom 0 = floor    geom 1-7 = torso/thighs/legs/feet    id = -1 = 天空
```

walker 掩膜 = `geom_id >= 1`，精确、无倒影、无形态学操作。**天空盒被 DAVIS
替换后分割仍返回 −1，故 easy 上同样精确**——判据表中「easy 上不用掩膜」的
限制不再成立，第 4d 步得以定量而非定性。
这是 GT.7 规则 2（真值优先用生成过程的已知量）的直接应用。

**(2) 独立的精确真值评估集。** clean / easy 各 2500 帧，`--save-segmentation`，
**同 seed 故物理完全相同**（两边 walker 占画面均为 0.0706）——姿态一致、
仅背景不同的受控对照。训练仍用 25K×2。

## W1.1.5 环境安装：三个未预告的坑

均已修，`PHASE2_PLAN.md §2` 已回填。

1. **`dm_control==1.0.14` 的 pin 必须去掉。** 它需要 `MjModel.bvh_geomid`；
   实测 mujoco 3.1.6 / 3.0.1 / 3.0.0 均无该字段。解法是升级 dm_control 到最新版、
   保留 mujoco 3.11.0。
2. **`distracting_control` 依赖 `cv2` 但未声明** → `opencv-python-headless`
   （非 `opencv-python`，避免无显示器机器上拉 GUI 依赖）。
3. **`distracting_control` 与现代 mujoco 不兼容**：用 `model.tex_rgb` 写天空盒，
   该字段已改名 `tex_data`。实测天空纹理 `nchannel=3, adr=0`，布局与旧版一致，
   故为纯改名。写成幂等补丁 `tools/dcs/patch_distracting_control.py`。
4. DAVIS 路径须给到 `DAVIS/JPEGImages/480p`（含视频序列的那一层）。

## W1.1.6 一个自查出的度量 bug

首次计算富集度时把掩膜归一化成和为 1 的分布，导致全部数值缩小 `mp.sum()=17.9` 倍
（clean FAST 显示 0.24 而非 4.15），几乎误报「不通过」。

**由 uniform 对照抓出**：它算得 0.056 而非定义上必然的 1.00。
该对照规则在合成阶段已救过一次（oracle 低于实测），此为第二次。
→ 强化 Diagnostic Protocol：**任何富集/分离类度量，uniform 对照必须精确等于
其解析值，不能只看「数量级合理」。**

## W1.1.7 采集脚本的两个设计决定

- 用 `env.physics.render` 而非 pixels wrapper，以保留 `qpos/qvel`（W1.1 要求存，
  后续 P2 的 proprio 侧直接可用）；干扰作用于 physics/skybox，背景照常出现在渲染中
- 「重建环境」（换 DAVIS 视频，代价高）与「重置回合」（换姿态，代价低）分离：
  首版每 8 clip 才重置，随机策略下 walker 约 25 步倒地，导致 7/8 的帧为倒地姿态。
  渲染检查发现后改为每 2 clip 重置

---

## 附：可复用诊断脚本

全部已整理进 `hippoact/tools/diagnostics/`，含 README（度量约定、参考数值、已知局限）。
从 `hippoact/` 目录直接运行，无需设 PYTHONPATH。

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
| `m1_matching.py` | Step M1 首轮：Hungarian 匹配 + identity/random 对照 |
| `m1_sweep.py` | M1 超参 sweep，单遍数据内评估全部组合 |
| `m1_oracle.py` | M1 天花板：双射/非双射 oracle 上界 + 分解对齐余弦 |
| `step1_probes.py` | Step 1 的 P1/P2/P3 一次跑完 |
| `gt_eval.py` | **精确真值**下的定位 / coverage / 跨帧身份 / exclusivity |
| `gt_router.py` | **P1 核心测量**：router 慢快划分 vs 真值，含 oracle/uniform 对照 |
| `gt_target.py` | 精确真值下对比三个候选 L_slow target |
| `gt_signal_search.py` | 搜索尺度不变的路由信号，含尺寸混淆检验 |

DCS 相关脚本在 `hippoact/tools/dcs/`：

| 脚本 | 用途 |
|---|---|
| `collect_frames.py` | DMC / DCS 采帧，clip 结构 + qpos/qvel + 可选精确分割掩膜 |
| `patch_distracting_control.py` | 幂等修复 `tex_rgb` → `tex_data`（现代 mujoco 兼容）|
| `walker_mask.py` | 帧差 + 闭运算的 proxy 掩膜（已被分割渲染取代，保留作对照）|
| `walker_enrichment.py` | W1.1 判据：fast slots 对 walker 的富集度 + oracle/uniform 对照 |

---

# W1.2 预注册 — TD-MPC2-pixel baseline 矩阵（A5000 gpu0+1）

> 常备规则 7：GPU-天 > 1 的 run，启动前写预注册段（预期数字 + 判据）。
> 执行环境：4×RTX A5000 服务器（sm_86, driver 535.288.01 / CUDA 12.2），**本轮只用 gpu0+gpu1**
> （gpu2 空闲但未获授权，gpu3 被他人占用）。
> 矩阵：{walker-walk, cheetah-run} × {clean, distracting-easy} × seed{1,2,3} = **12 runs × 500K steps**。

## W1.2.0 步数单位：`cfg.steps` 是 agent step，官方 CSV 是 env step（2×）

> **本节已按论文一手证据重写（原版本把两种单位混为一谈，据此推出的判据差一倍，
> 并由此产生了一个虚假的"比官方高 82%"的发现。修订理由与撤回见 §W1.2.7。）**

论文 Table 6（Environment details）：

```
                 DMControl
Episode length     1,000        <- env step
Action repeat        2
Effective length     500        <- agent step，与代码 Timeout(max_episode_steps=500) 一致
Total env. steps   4M - 14M
```

- 代码侧：`OnlineTrainer._step` 每次 `env.step()` 加 1，而 `DMControlWrapper.step`
  内部跑 2 个物理步 → **`cfg.steps` / 日志 `step` = agent step**
- 论文与仓库自带 `results/*.csv` 侧：报的是 **env step**
- 交叉验证 1：README 的 `task=dog-run steps=7000000` = 7M agent = **14M env**，
  正好是 Table 6 的预算上限
- 交叉验证 2：state 版 CSV 最大 4M。若那是 agent step 则等于 8M env，超出 Table 6 的 4M

**故：官方 CSV 的 step = 2 × 我们的 `cfg.steps`。**

### 用我们自己的曲线反向验证（决定性）

| | 我们 @agent step | = env step | 官方 @同一 env step | 比值 |
|---|---|---|---|---|
| walker-walk | 842.0 @100K | 200K | 836.1 | **1.007** |
| walker-walk | 509.0 @50K | 100K | 463.7 | 1.098 |
| cheetah-run | 257.0 @50K | 100K | 286.4 | 0.897 |

三点全部落在 seed 噪声内。**我们的 pixel baseline 与官方一致**——
这正是 P2.2「对齐官方」要的结论，原 TODO 的措辞没错。

### 判据（按 agent step 表述，本轮实际使用）

阈值取「对应 env step 处官方均值的 0.9×」，该值落在官方最差 seed 上或略低：

| 里程碑 | 判据（agent step） | 官方对照（env step） |
|---|---|---|
| P2.1 管线通 | walker-walk clean **100K ≥ 700** | @200K env = 836.1（seeds 784 / 834 / 890），0.9× = 752 |
| P2.2 baseline 对齐 | walker-walk clean **500K ≥ 850** | @1M env = 939.6（929 / 942 / 949），0.9× = 846 |
| P2.2b | cheetah-run clean **500K ≥ 480** | @1M env = 537.3（453 / 570 / 590），0.9× = 484 |

原 TODO / PHASE2_PLAN 的「100K ≥ 500」与「500K 650–750」**在两种单位约定下都对不上**
官方数字（500K env=250K agent 处官方 949；500K agent=1M env 处官方 939.6），
来源不明，仍按上表执行。

**写进论文时**：`cfg.steps=500000` 应报为 **1M environment steps**（action repeat 2），
与 TD-MPC2 / DrQ-v2 系论文的横轴一致。直接写「500K steps」会被 reviewer 读成
一半的预算。

**预算含义**：1M env steps 是论文 DMC 最低预算（4M）的 1/4。walker 在此已饱和
（官方 250K agent 处 949、500K agent 处 939.6），cheetah 仍在涨
（454.9 → 537.3），故 500K agent 的选择给了 cheetah 一个明显更强的 baseline。

## W1.2.1 预期数字（预注册，跑完后逐格对照）

**下表按 agent step 索引，官方值取对应 env step（2×）处的数字。**
clean 两列直接引用官方 CSV；easy 两列**无官方数字**，给的是区间预期而非硬判据：

| cell | 100K agent (=200K env) | 500K agent (=1M env) | 来源 / 理由 |
|---|---|---|---|
| walker-walk clean | 836 | **940** | 官方 CSV，3 seeds |
| cheetah-run clean | 343 | **537** | 官方 CSV，3 seeds |
| walker-walk easy | — | 预期 **400–750** | 无先例可引。W1.1 实测 easy 背景贡献 47.9% 的像素变化（clean 12.7%），CNN encoder 无对象先验，预期显著掉点但不至于崩到 0 |
| cheetah-run easy | — | 预期 **150–400** | 同上，cheetah 本身分数低、相对掉幅可能更大 |

easy 两格**故意不设通过/失败线**：它们是 Q2 的被测量本身（baseline 掉多少，
正是 HippoAct 要赢的空间）。预先设线会把"想要的结果"写进判据。
若 easy 掉到接近 0，则记录为"pixel baseline 在背景干扰下崩溃"，这是有利于论文的结果，
但需检查是否为管线故障（对照：同 seed 下 easy 与 clean 的 reward 序列在随机策略下必须完全相同——
smoke test 已验证 max|Δreward| = 0）。

## W1.2.2 环境与集成（已完成，实测记录）

conda env `hippoact`（python 3.11）。**版本选择与 W1.1（5080）不同，理由如下**：

| 包 | 本机 | W1.1 (5080) | 为什么不同 |
|---|---|---|---|
| dm_control | **1.0.16** | latest | 1.0.16 是 tdmpc2 官方 docker pin，公平对比优先 |
| mujoco | **3.1.2** | 3.11.0 | **`tex_rgb` → `tex_data` 的改名发生在 mujoco 3.2**。停在 3.1.2 则 `distracting_control` 原样可用，PHASE2_PLAN 坑 3 的补丁**本轮不需要**（已实测：easy 背景正常渲染） |
| numpy | 1.24.4 | <2.0 | 同 |

新踩到的坑（PHASE2_PLAN 未预告）：

1. **`pip install distracting-control` 会把 numpy 拉到 2.4.6**（它依赖老 `gym`），
   必须装完再 `pip install numpy==1.24.4` 压回去。
2. **`opencv-python-headless` 最新版（5.0.x）强制 numpy≥2**，与上一条互斥。
   固定 `opencv-python-headless<4.12`（实测 4.11.0 与 numpy 1.24.4 共存）。
3. **DAVIS 路径是 `DAVIS/JPEGImages/480p` 这一层**（PHASE2_PLAN §2 已写对，
   但本模块首版按"数据集根目录"理解写错，报 `FileNotFoundError: .../DAVIS/boat`）。
   数据放在 `/usr1/home/s125mdg56_03/datasets/davis/`（**不放 /var/tmp**：
   4 天的 run 不能依赖可能被清理的临时目录）。

### fork 改动量：2 个文件，+22 −1 行，**算法零改动**

`third_party/tdmpc2` @ `e9f59321933cbc8e11a002b842adc7d4ffae8ff1`（记于 `PINNED`）。
diff 存为 `experiments/dcs/tdmpc2_fork.patch`：

- `envs/__init__.py`：注册 `make_dcs_env`（+6 −1）
- `envs/dcs.py`：17 行 shim，实际代码在版本控制内的 `experiments/dcs/dcs_env.py`

任务名 `dcs-<difficulty>-<domain>-<task>`。`difficulty=none` **绕开 distracting_control
直接走 dm_control**，因此 clean 组用的是官方原生 `walker-walk` / `cheetah-run` 路径。

## W1.2.3 集成正确性检查（`experiments/scripts/smoke_env.py`，全过）

| 检查 | 结果 | 意义 |
|---|---|---|
| `dcs-none-walker-walk` vs 官方 `walker-walk`：同 seed 同动作序列 | max\|Δreward\| = **0.000e+00**，max\|Δpixel\| = **0** | shim 未改变官方 baseline 路径，Table V 的 clean 列可信 |
| obs / action / episode 长度 | 两边同为 (9,64,64) / (6,) / 500 | — |
| easy vs clean 像素差异 | **51.1%** 的像素不同 | 背景确实被 DAVIS 替换 |
| 帧间变化量 easy / clean | **21.30 / 9.09** | 背景是动态的（`dynamic=True`），且与 W1.1 测到的"easy 背景贡献近半像素变化"一致 |
| easy vs clean 的 reward 序列 | Δ = **0.000e+00** | 干扰只动像素、不动任务；这是 easy 组掉点时区分"真掉点"与"管线故障"的对照 |

**注意分辨率**：TD-MPC2 官方 pixel 配置是 **3 帧 stack × 64×64**（`envs/dmcontrol.py`
的 `Pixels(num_frames=3, size=64)`），**不是 PHASE2_PLAN §4 写的 84×84**。
本轮按官方 64×64 跑（保持 baseline 的最优配置才叫公平）。
PHASE2_PLAN §4 的"84 vs 224"那段需改为"64 vs 224"。

## W1.2.4 资源预算

| 项 | 实测 / 估算 |
|---|---|
| replay buffer | capacity = min(1M, steps) = 500K；rgb 9×64×64 uint8 → **18.45 GB/run** |
| buffer 落点 | **CPU 内存**（tdmpc2 启发式：`2.5×18.45 GB > 24 GB` 显存，自动退 CPU）——非静默降级，日志明写 `Using CPU memory for storage.` |
| 显存/run | < 1 GB（模型仅 4.88M 参数，batch 256×horizon 3） |
| CPU 内存/run | ~21.5 GB（buffer 18.45 + 进程 ~3） |
| 主机内存 | 251 GB 总 / ~225 GB 可用 → 并发上限 ~8，取 **6 并发**（~130 GB）留余量 |
| env-only 吞吐 | clean **459 step/s**；easy **91.7 step/s**（**5× 慢**：`dynamic=True` 每步重传天空盒纹理） |
| 启动开销 | torch.compile + step-0 eval ≈ 5–10 min/run（一次性） |

**easy 比 clean 慢 5 倍是本轮最重要的排期事实**：easy 的 6 个 run 会成为关键路径。

## W1.2.5 执行记录：一次把我误导了的"假死"

首轮两个 run（walker clean on gpu0 / dcs-easy on gpu1）并发起跑后，
**clean 那个卡了两次**：`I: 2,500 → 3,000` 用了 15 分钟，随后 `I: 3,500` 之后
又整整 13 分钟无任何输出。同时段 dcs-easy 已经跑到 28,500 步，一切正常。

现象：进程 state `R`、单核 100% CPU、GPU util **0%**、RSS 不涨、
voluntary context switch 极低（8566，对照健康进程 49358）。

**我的第一次判断是错的。** 我去看 inductor 缓存目录，45 秒内大小零增长、
最新文件停在 13:06，于是排除了"还在编译"。基于这个我准备去查 EGL 回退、
SliceSampler 采样退化等方向。

实际原因就是编译——**dynamo 追踪阶段是纯 Python、不写 inductor 缓存**，
用"缓存目录不增长"去否证"正在编译"是个无效判据。
验证方式（单变量）：杀掉重启，此时 inductor 缓存已被 dcs-easy 那个 run 预热：

| | 冷缓存（首轮） | 热缓存（重启后） |
|---|---|---|
| seed pretraining（2500 updates） | **15 min** | **2 min 15 s** |
| `I: 3,500` 之后 | **停 13 min** | 无停顿，26 SPS 稳定 |

TD-MPC2 的 `act()` 按 `(t0, eval_mode)` 的取值分别编译（日志里的 recompile
guard 明写 `___check_obj_id(kwargs['t0'])` / `kwargs['eval_mode']`），
所以一个 run 前期会连着触发数次编译；两个 run 同时冷编译时互相抢 CPU，
单次可拖到十几分钟。**这不是故障，是冷启动开销**，但会让人以为 run 死了。

### 留下的工具

服务器 `kernel.yama.ptrace_scope=1` 且无免密 sudo，**py-spy 挂不上已在跑的进程**。
为此加了 `experiments/scripts/train_dbg.py`：透明包一层 tdmpc2 的 `train.py`，
注册 `SIGUSR1 → faulthandler.dump_traceback(all_threads=True)`，
并调 `prctl(PR_SET_PTRACER, ANY)` 让 py-spy 可以附加。argv 完全一致，
**不碰 TD-MPC2 任何代码**。之后所有 run 都经它启动，卡住时 `kill -USR1 <pid>` 即可取栈。

## W1.2.6 吞吐实测与排期

单 run 独占一卡（热缓存后稳态）：

| cell | SPS | 500K 单 run 耗时 |
|---|---|---|
| walker-walk clean | **~26** | ~5.3 h |
| dcs-easy-walker-walk | **~18–20** | ~7.2 h |

easy 比 clean 慢，与 env-only 测到的 5× 差距（459 vs 91.7 step/s）方向一致但幅度小得多——
说明**瓶颈是 update + MPPI 规划而非渲染**（env 只占单步约 10%）。

排期：12 run 串行约 73 GPU-h；2 卡 × 3 并发 = 6 并发。
GPU util 单 run 仅 20–44%，显存 < 1.3 GB/run，瓶颈在 CPU 与 PCIe（buffer 在 CPU 内存，
每次 update 要搬 256×4 帧 ≈ 37 MB）。**预计 1.5–2 天跑完**，早于 TODO 预算的 4 天。

## W1.2.7 ~~P2.1 判据：通过，但比官方高 82%~~ —— **已撤回，是我的单位错误**

> **撤回声明。** 本节原先报告"我们的 pixel baseline 比官方高 82%（842.2 vs 463.7）"，
> 并据此写了三条"对论文的影响"，其中包括"Table V 必须报实测、不能写对齐官方"。
> **这个发现是假的**，根源是我把我们的 agent step 直接对上了官方 CSV 的 env step。
> 三条影响全部作废。修正后的单位推导见 §W1.2.0。

### 错在哪

原文写着"已排除单位错位"，理由是"日志的 `step` 与 CSV 的 `step` 都来自同一个
`Logger.log(step=self._step)`"。**这个理由是我编的**——仓库里没有任何文档说明
`results/*.csv` 是怎么生成的，我把"看起来像同一个 logger 的产物"当成了事实。
真正的证据在论文 Table 6（`Action repeat 2` / `Episode length 1,000` /
`Effective length 500`）和 README 的 `dog-run steps=7000000`（= 14M env steps），
两者都指向 CSV 用的是 env step。**我当时没去查论文就下了结论。**

### 对齐单位后的实际结论

```
walker-walk clean seed1:  25K:368  50K:542  75K:669  100K:842.2   (agent steps)

我们 100K agent = 200K env  ->  官方 @200K env = 836.1     比值 1.007
我们  50K agent = 100K env  ->  官方 @100K env = 463.7     比值 1.098  (仅 seed1 时 542)
cheetah 50K agent = 100K env -> 官方 @100K env = 286.4     比值 0.897
```

**我们的 baseline 与官方逐点一致（walker 100K 处 1.007）。**
P2.1 判据（修正后 100K ≥ 700）**通过**，且是以"复现了官方曲线"的方式通过——
比原先那个虚假的"更强"结论更有价值：它同时验证了环境、超参、评测协议三者都对。

### 对论文的影响（修正版）

1. **Table V 可以如实写"与 TD-MPC2 官方 pixel 曲线一致"**，并注明 commit
   `e9f5932`、`model_size=5`、1M environment steps。原先"必须撇清、报自己数字"
   的措辞不需要。
2. **横轴一律用 environment steps**（= 2 × `cfg.steps`），与 TD-MPC2 / DrQ-v2 一致。
3. **一条方法学教训**：这是第三次「上界/参照低于实测」类信号（前两次见 M1.5 与
   W1.1.6）。前两次都被我当成度量 bug 的信号抓出来了，这次我却给它编了一个
   "归档 CSV 与当前代码不是同一实现"的故事。**参照系与实测差出量级时，
   先怀疑单位和对齐方式，再怀疑对方**——应写进 Diagnostic Protocol。

### 同期 easy 组（同一 seed，唯一差别是背景）

```
walker-walk clean  100K: 842.2
dcs-easy-walker-walk 100K: 329.4        <- 掉到 39%
```

clean 与 easy 在随机策略下 reward 序列完全相同（smoke test 已验证 Δ = 0），
故这 513 分的差距是**学习被背景干扰破坏**，不是任务被改变。
这正是 Q2 要的 baseline 空间。

**待办（下一轮）**：500K 跑完后，在 `docs/paper_section_IV_experiments.md`
注明 TD-MPC2 的 commit hash `e9f5932`、`model_size=5`，横轴改用
**environment steps（= 2 × cfg.steps）**，并说明 clean 组复现了官方 pixel 曲线。

## W1.2.8 事故记录：渲染上下文跑到了未授权的卡上（我的失误，已修，全部重启）

首轮 6 个 run 跑了 1.8 小时后，用户发现 **4 张卡全被挂上了**。查 `nvidia-smi` 的
进程表才看清：**计算（`C`）确实只在 gpu0/gpu1，但每个进程还在 gpu2 或 gpu3 上
开了一个 73 MiB 的 EGL 图形上下文（`G`）**——MuJoCo 的渲染跑到了别的卡上。

```
CUDA 在 gpu0 的 3 个进程  ->  EGL 落在 gpu2
CUDA 在 gpu1 的 3 个进程  ->  EGL 落在 gpu3        <- 该卡有他人 22.7 GB / 95% 的任务
```

**根因**：我在队列里写了 `MUJOCO_EGL_DEVICE_ID = str(cuda_id)`，
默认 EGL 的设备枚举顺序等于 CUDA 顺序。**在这台机器上不等于。**
用 `experiments/scripts/egl_probe.py` 实测（逐个 id 起进程渲染，看 nvidia-smi 认到哪张卡）：

| `MUJOCO_EGL_DEVICE_ID` | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| 实际物理 GPU | **2** | **3** | **0** | **1** |

即相对 CUDA 顺序**整体转了两位**。修正后复验：
`CUDA_VISIBLE_DEVICES=0 + MUJOCO_EGL_DEVICE_ID=2 → 渲染在 gpu0` ✅，
`1 + 3 → gpu1` ✅。`nvidia-smi` 现在每个进程都是 `C+G` 在同一张卡。

**为什么没早发现**：渲染跑错卡**不会报错也不会变慢到明显**，
`nvidia-smi --query-gpu` 只看显存/利用率汇总，不看进程类型；
我全程用的就是那个汇总视图，而 gpu2/gpu3 本来就有他人任务占着，
显存与利用率的变化被掩盖了。**教训：多卡作业启动后必须看
`nvidia-smi` 的完整进程表（含 `C`/`G` 类型列），不能只看汇总。**

**处置**：按用户决定，杀掉全部 6 个 run（丢弃 ~1.8 h 墙钟 / 约 61 万步），
用单一队列 + 正确映射重跑 12 个（`--gpus 0,1 --slots-per-gpu 3`）。
额外收益：之前 EGL 落在 gpu3 的那 3 个 run，渲染排在他人 95% 满载任务后面，
本身就被拖慢（14–15 SPS vs gpu2 侧的 17.8）；修正后这部分损失也一并消失。

**首轮已取得且仍然有效的结论**（环境未变，只是渲染卡选错，不影响数值正确性）：
P2.1 判据通过（walker clean 100K = 842.2）、§W1.2.7 的"比官方高 82%"、
以及 clean/easy 的差距（100K：842 vs 329）。这些数字在重跑后应能复现，
届时以重跑结果为准写入 Table V。

## W1.2.9 Table V 的读数口径：final checkpoint × 30 episodes（不用 500K 单点）

### 现象

前 4 个跑完的 clean run，**最后一次 eval 全部低于此前的平台**：

```
walker s1  >=300K: [953,964,965,964,961,955,966,963, 819]   前8点 961.4 ± 5.0
walker s2:         [963,954,956,970,964,972,952,962, 928]         961.5 ± 7.3
cheetah s1:        [577,517,601,580,568,537,591,507, 418]         559.6 ± 34.9
cheetah s2:        [480,478,493,487,479,469,477,453, 419]         477.1 ± 12.1
```

值得单独记一笔：cheetah 两个独立 seed 的末点是 **418.4 / 418.7**，几乎重合。

### 不是 eval 假象——独立进程重测复现了它

加载保存的 `final.pt`，在全新进程里用 30 个 episode 重测（`evaluate.py`）：

| | 训练内 500K（10 ep） | 独立重测（30 ep） | 300–475K 平台 |
|---|---|---|---|
| walker s1 | 818.6 | **846.9** | 961.4 ± 5.0 |
| walker s2 | 928.5 | **935.6** | 961.5 ± 7.3 |

两次测量一致 → **末端策略确实略差**，不是测量问题。walker s1 掉约 115 分是实的，
s2 掉 26 属正常波动。

### 为什么必须处理，而不是照抄

用 500K 单点填 Table V 的话（n=2 seeds 时）：

```
walker-walk  873.6 vs 官方 939.6  ->  0.93   P2.2  勉强 PASS
cheetah-run  418.6 vs 官方 537.3  ->  0.78   P2.2b FAIL
```

而同样这两个 run 在 375K 处是 `cheetah 533.3 vs 官方 510.9 = 1.04`。
**判据会因为一个末点而误报失败**，更要命的是它把我们自己的 pixel baseline
压低约 9%——**这个偏差的方向对 HippoAct 有利**，正是最不该留在论文里的那一类。

### 决定（用户裁定，选项 2）

**Table V 报 final checkpoint 在 30 个 episode 下的重测值**，理由：

1. 评的是"实际会拿去和 HippoAct 比"的那个策略，而不是训练轨迹上的某个采样点
2. 30 episode 是训练内 eval（10）的 3 倍，把单点噪声压下去
3. 口径对 clean 与 easy 两组一致，不存在偏袒

同时**如实报告末端退化这一观察**（§V 或附录），不掩盖。

实现：`experiments/scripts/final_eval.py`（结果缓存在 `experiments/logs/final_eval.json`，
可重入），`w12_report.py` 增加 TABLE V 段与 retention(easy/clean) 行。

**未解释的疑点（留给下一轮）**：为何 4/4 都在末点掉、且 cheetah 两 seed 末点
几乎相同（418.4 / 418.7）。已排除 eval 假象；尚未排除 replay buffer 在
`capacity = min(1M, steps) = 500K` 处**恰好填满**是否与此有关（buffer 在训练
结束的同一时刻写满，此后开始覆盖）。若下一轮 8 个 run 复现同一现象，值得查。

---

# W1.2 完成 — TD-MPC2-pixel baseline 矩阵（12/12 runs）

> 12 个 run 全部跑完（2026-08-01 22:26）。总耗时约 31 小时墙钟，2×A5000、6 并发。
> 读数口径：final checkpoint × 30 eval episodes（§W1.2.9 决定），
> 脚本 `experiments/scripts/final_eval.py`，缓存 `experiments/logs/final_eval.json`。

## W1.2.10 Table V

| cell | mean ± sd (3 seeds) | per-seed |
|---|---|---|
| walker-walk clean | **915.3 ± 60.8** | 847 / 936 / 963 |
| dcs-easy-walker-walk | **853.7 ± 89.0** | 754 / 925 / 882 |
| cheetah-run clean | **442.0 ± 28.7** | 475 / 428 / 423 |
| dcs-easy-cheetah-run | **389.7 ± 84.9** | 301 / 470 / 398 |

### 与官方 pixel 曲线对照（我们 agent step，官方 env step = 2×）

| cell | agent | ours (n3) | official (n3) | 比值 |
|---|---|---|---|---|
| walker-walk | 100K | 869.1 ± 17.0 | 836.1 ± 53.1 | **1.04** |
| walker-walk | 250K | 952.4 ± 2.7 | 949.0 ± 6.3 | **1.00** |
| walker-walk | 375K | 966.1 ± 3.6 | 958.9 ± 11.5 | **1.01** |
| walker-walk | 500K | 902.9 ± 74.9 | 939.6 ± 10.1 | 0.96 |
| cheetah-run | 100K | 352.0 ± 36.8 | 343.3 ± 38.6 | **1.03** |
| cheetah-run | 250K | 489.7 ± 81.1 | 454.9 ± 41.3 | **1.08** |
| cheetah-run | 375K | 500.3 ± 73.8 | 510.9 ± 89.5 | **0.98** |
| cheetah-run | 500K | 425.6 ± 12.1 | 537.3 ± 73.9 | 0.79 |

**两个任务在 100K / 250K / 375K 三点全部落在 0.98–1.08。管线正确、超参正确、
评测协议正确，这一点没有疑问。** 偏差只出现在 500K 这一个端点。

### 判据

```
P2.1  walker-walk @100K  869.1 vs >= 700  -> PASS
P2.2  walker-walk @500K  915.3 vs >= 850  -> PASS
P2.2b cheetah-run @500K  442.0 vs >= 480  -> FAIL   (差 38)
```

**P2.2b 如实记为不通过。** 不改判据、不换读数点——cheetah 在 375K 是 0.98×官方，
若把 Table V 的读数点挪到 375K 就能"通过"，但那是为了好看挑点，不做。

## W1.2.11 末端退化：12 个 run 的完整证据

`z` = 500K 单点相对 400K–475K 四点的标准分；`ckpt30` = final checkpoint 30 episode 重测。

| run | 400–475K | 500K | z | ckpt30 | 平台 |
|---|---|---|---|---|---|
| walker-walk s1 | 961 955 966 963 | 819 | **−29.5** | 847 | 961 |
| walker-walk s2 | 964 972 952 962 | 928 | **−4.1** | 936 | 962 |
| walker-walk s3 | 975 959 955 963 | 962 | −0.1 | 963 | 963 |
| cheetah-run s1 | 568 537 591 507 | 418 | **−3.6** | 475 | 551 |
| cheetah-run s2 | 479 469 477 453 | 419 | **−4.2** | 428 | 470 |
| cheetah-run s3 | 452 448 446 431 | 440 | −0.5 | 423 | 444 |
| dcs-easy-walker s1 | 723 727 773 784 | 817 | +2.1 | 754 | 752 |
| dcs-easy-walker s2 | 858 899 896 909 | 933 | +1.9 | 925 | 891 |
| dcs-easy-walker s3 | 852 868 868 851 | 886 | +2.7 | 882 | 860 |
| dcs-easy-cheetah s1 | 295 277 326 317 | 357 | +2.4 | 301 | 304 |
| dcs-easy-cheetah s2 | 376 470 444 471 | 474 | +0.8 | 470 | 440 |
| dcs-easy-cheetah s3 | 348 361 394 353 | 359 | −0.2 | 398 | 364 |

**掉点的恰好是第一波的 4 个 clean run，其余 8 个（含第一波的 2 个 easy）都没掉。**

关键对照：没掉的 run，`ckpt30 ≈ 平台`（walker s3: 963 vs 963）；掉了的，
`ckpt30` 明显低于平台（walker s1: 847 vs 961）。两者用同一脚本、同一负载测的，
**差别在权重本身，不是测量环境。**

### 已排除

- **eval 假象**：独立进程加载 `final.pt`、30 episode 重测复现（§W1.2.9）
- **replay buffer 在 `capacity=min(1M,500K)` 处写满**：对 12 个 run 都成立，
  而 8 个没掉。此前 §W1.2.9 记的这条猜测**否定**
- **第一波/第二波之分**：第一波的 2 个 easy run 没掉
- **clean/easy 之分**：第二波的 2 个 clean run 没掉

### 未解释

「第一波 × clean」这个交集为何特殊，我没有能站住的解释。**不编。**
四个 run 恰好是最早跑完的 4 个，其 final eval 期间队列正在启动替补 run
（torch.compile，CPU 密集）——但最先跑完的 cheetah s1 结束时并无替补在编译，
该解释不自洽。

**若要追**：唯一能证伪的实验是重跑 cheetah clean s1/s2（2 个 run，2 卡并行 ~9h），
看退化是否复现。代价可接受，因为 cheetah baseline 偏弱**对 HippoAct 有利**，
是最不该留的偏差方向。

## W1.2.12 一个会影响 Q2 主图设计的发现

**easy 背景干扰对 pixel baseline 的代价几乎全在样本效率，不在最终性能：**

| agent step | walker clean | walker easy | easy/clean | cheetah clean | cheetah easy | easy/clean |
|---|---|---|---|---|---|---|
| 50K | 550 | 214 | **0.39** | 258 | 80 | **0.31** |
| 100K | 869 | 391 | 0.45 | 352 | 138 | 0.39 |
| 200K | 944 | 572 | 0.61 | 465 | 214 | 0.46 |
| 300K | 961 | 744 | 0.77 | 486 | 292 | 0.60 |
| 400K | 966 | 811 | 0.84 | 500 | 340 | 0.68 |
| 500K | 903 | 879 | **0.97** | 426 | 397 | **0.93** |

按 Table V 口径（final ckpt × 30ep）：walker **0.93**、cheetah **0.88**。

而且 **easy 组在 500K 时尚未收敛**（dcs-easy-walker 的末段仍在 723→817 单调上升），
所以 0.93 / 0.88 是比值的**下界**，给足步数很可能进一步逼近 1.0。

### 对论文的影响

TODO §Week 3-4 的主图设定隐含「pixel baseline 在 DCS 上明显吃亏」。
**在 easy 难度、1M env steps 下这个前提不成立**——baseline 最终只差 7–12%。

需要区分清楚两件不同的事：

1. **本轮测的**：easy 上训练 → easy 上评测。差距主要是样本效率（50K 时 0.31–0.39）
2. **论文 Q2 真正要的**：easy 上训练 → **hard 上零样本评测**，retention = R(hard)/R(none)。
   **W1.2 完全没测这个**，它是 Week 3-4 的内容

所以本轮结果**不能**推翻 Q2，但它改变了「HippoAct 该赢在哪」的判断：
若只比 easy 上的最终分数，天花板只有 7–12%；**真正的空间在样本效率与零样本泛化**。
建议 Q2 主图把样本效率曲线一并画出，而不只报最终分数。

## W1.2.13 交付物

| 文件 | 内容 |
|---|---|
| `experiments/logs/<task>/<seed>/w12_pixel/eval.csv` | 12 条学习曲线（每 25K 一点） |
| `experiments/logs/final_eval.json` | 12 个 final checkpoint 的 30-episode 重测值 |
| `experiments/scripts/w12_report.py` | Table V + 官方对照 + 判据裁决 |
| `experiments/scripts/final_eval.py` | 重测脚本（可重入、带缓存） |
| `experiments/dcs/tdmpc2_fork.patch` | 对 tdmpc2 的全部改动：2 文件 +22 −1，算法零改动 |

**未做**：W1.3（DrQ-v2 baseline）——需要第三张卡，本轮只授权 gpu0/gpu1。

