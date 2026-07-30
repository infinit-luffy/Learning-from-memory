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
