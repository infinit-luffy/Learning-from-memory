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

---

# W1.2 后续三项（TODO 插队指令，2026-08-01 起）

四卡排布（TODO §算力切换）。**gpu0 原定的 W2.1 E2E-0 阻塞**：
需要 5080 上训好的 DCS Stage-1 checkpoint，本机全盘 `*.pt/*.ckpt` 搜索为空，
迁移说明里的 scp 尚未发生。gpu0 改派给 retention 评测与 W1.3。

迁移自检已过：`pytest tests/ -v` → **23 passed**（A5000 / torch 2.7.1+cu126 环境等价）。

## R1 零样本 retention 评测（插队第 2 项，gpu0+gpu1）

脚本 `experiments/scripts/retention_eval.py`：12 个 W1.2 checkpoint ×
{none, easy, hard} × 30 episodes = 36 次纯评测，无训练。
`none` 走 tdmpc2 原生路径，`easy/hard` 走 `experiments/dcs/dcs_env.py`
（difficulty 决定取几个 DAVIS 视频：easy=4 / hard=全部）。

一致性对照通过：`cheetah s1 on none = 475.0`，与 `final_eval.json` 的 475.0
逐位一致（同 ckpt、同 30 episode、同 seed）——评测管线可复现。

**早期数字（clean 训练的 checkpoint，零样本到干扰环境）**：

| | eval none | eval easy | eval hard |
|---|---|---|---|
| cheetah-run s1 | 475.0 | 27.8 | — |
| cheetah-run s2 | 431.4 | 19.9 | 21.5 |
| cheetah-run s3 | 423.1 | 28.5 | — |

`retention(hard/none)` ≈ **0.05**（s2）。clean 训练的 pixel 策略在视频背景下
**几乎完全失效**。这与 W1.2.12 的发现并不矛盾而是互补：
easy 上**训练**的策略最终能达到 clean 的 0.93/0.88（§W1.2.12），
但 clean 上训练的策略**零样本迁到 easy/hard 会崩到 5%**。
→ 论文 Q2 的两条轴（样本效率 + 零样本 retention）都有实测支撑。

easy 训练的 checkpoint 才是协议行（train easy → eval {none, easy, hard}），
跑完后补表。

## R2 cheetah clean s1/s2 重跑（插队第 3 项，gpu2+gpu3）

`exp_name=w12_pixel_rerun`，同 seed 同配置，原始结果保留在 `w12_pixel` 下对照。
**预注册**：若末端退化（原 z = −3.6 / −4.2）复现 → 记为 TD-MPC2 末期方差，
如实写入 §V；若不复现 → 说明是偶发，Table V 换用重跑值并记录两次结果。
**无论哪种都不挪读数点。**

## R3 W1.3 DrQ-v2 baseline（预注册）

**矩阵**：{walker_walk, cheetah_run} × {none, easy} × 3 seeds = 12 runs，
各 `num_train_frames=1_000_000`（= 500K agent steps，与 W1.2 预算逐点可比）。

**集成**（`experiments/dcs/drqv2_dcs.py` + fork 4 文件 +29 −8）：
干扰由**新增 config 字段 `distraction`** 选择，而非新任务名——这样
`cfgs/task/*.yaml` 的 per-task 超参（`nstep` / `batch_size` /
`stddev_schedule` / `num_train_frames`）在 clean 与 easy 两臂上**完全一致**，
两臂唯一差别就是背景。原始 dm_env 由 **TD-MPC2 侧同一个 `dcs_env.make_dm_env`**
构造，故两个 baseline 看到的背景分布完全相同。

**smoke test 全过**（`experiments/scripts/smoke_drqv2.py`）：

| 检查 | 结果 |
|---|---|
| `distraction=none` vs DrQ-v2 原生路径 | max\|Δreward\| = 0，max\|Δpixel\| = 0 |
| easy 与 clean 的像素差异 | 50.9% |
| 帧间变化量 easy / clean | 22.67 / 9.20（背景动态） |
| easy vs clean 的 reward 序列 | Δ = 0（干扰不改任务） |
| env-only 吞吐 | clean 432.6 / easy 97.4 step/s |

**分辨率差异需在论文披露**：DrQ-v2 用 **84×84**，TD-MPC2 用 **64×64**，
各自的已发表配置，不做统一（统一会让某一方偏离其最优设置）。

**两个安装期坑（已记入 `experiments/README.md`）**：
1. DrQ-v2 README 的 `task=walker_walk` 是 hydra 1.1 写法，
   hydra 1.3 下必须写 `task@_global_=walker_walk`，否则直接报错退出。
2. `replay_buffer._worker_init_fn` 把 `np.uint32` 传给 `random.seed()`，
   Python 3.11 拒绝 numpy 整数类型 → 崩在 DataLoader worker 里。
   已修为 `int(...)`，**播种语义不变**。这是上游与新 numpy/Python 的不兼容，
   不是算法改动。

**预期**（DrQ-v2 论文 medium 组 1M frames 处）：walker-walk ~900+、
cheetah-run ~500 上下；easy 两格无先例，不设硬判据，理由同 §W1.2.1。

**资源**：单 run ~43 FPS（= 21.5 agent step/s），1M frames ≈ 6.5 h 独占；
6 并发预计 ~1.5 天。replay buffer 落**磁盘**（非内存），
每 run ~21 GB × 12 = ~254 GB，/usr1 余 1.1 T，够。

---

# R1 结果 — 零样本 retention（12 个 W1.2 checkpoint × {none, easy, hard} × 30 ep）

脚本 `experiments/scripts/retention_eval.py`，缓存 `experiments/logs/retention_eval.json`。
36/36 全部完成。一致性对照通过：`cheetah s1 on none = 475.0` 与 `final_eval.json`
逐位一致。

| 训练于 | eval none | eval easy | eval hard | hard/none |
|---|---|---|---|---|
| **dcs-easy-walker-walk** | 556.6 ± 65.9 | **853.7 ± 89.0** | 633.0 ± 85.6 | **1.137** |
| walker-walk (clean) | **916.0 ± 59.6** | 75.8 ± 7.4 | 97.1 ± 10.9 | **0.106** |
| **dcs-easy-cheetah-run** | 118.6 ± 13.6 | **389.7 ± 84.9** | 191.8 ± 47.4 | **1.618** |
| cheetah-run (clean) | **443.2 ± 27.9** | 25.4 ± 4.8 | 28.6 ± 6.2 | **0.064** |

（粗体 = 该行的训练分布）

## R1.1 论文 §IV.C 的 retention 定义在这组数据上失效

`retention = R_hard / R_none` 对 **easy 训练**的两行给出 **1.137 / 1.618**——
大于 1，作为"鲁棒性"指标毫无意义。原因不是 hard 上表现好，而是**分母塌了**：

```
dcs-easy-walker-walk :  none 556.6  <  easy 853.7    (差 297)
dcs-easy-cheetah-run :  none 118.6  <  easy 389.7    (差 271)
```

**easy 上训练出来的策略，回到干净背景反而更差。** 它把"画面里有视频背景"
一并学进了状态表征，去掉背景同样是分布外。所以 `none` 不是一个中性参照，
不能当分母。

### 建议改用 `R_hard / R_easy`（相对训练分布）

| | R_hard / R_easy |
|---|---|
| dcs-easy-walker-walk | **0.741** |
| dcs-easy-cheetah-run | **0.492** |

这两个数才有"从训练分布走到更强干扰、保住多少"的含义，
且分母是策略自己的训练条件、不含额外分布偏移。

**这是需要 cowork 拍板的指标定义变更**（§IV.C 现文写的是 hard/none）。
在改定之前，两种口径的数都留在上表里，不做取舍。

## R1.2 真正的大差距在"训练时见没见过干扰"

```
walker :  clean 训练 → hard 97.1     easy 训练 → hard 633.0     6.5×
cheetah:  clean 训练 → hard 28.6     easy 训练 → hard 191.8     6.7×
```

clean 训练的策略零样本迁到干扰环境 **retention 0.064–0.106**，几乎完全失效
（walker 916 → 97，cheetah 443 → 29）。

这与 §W1.2.12 互补而非矛盾：
- **训练时见过干扰**：最终分数只差 7–12%（§W1.2.12），代价几乎全在样本效率
- **训练时没见过**：零样本崩到 6–11%

→ Q2 的两条轴（样本效率 + 零样本泛化）都拿到了 baseline 实测值，
且两条轴上 pixel baseline 都有明确弱点可打。

---

# R2 结果 — cheetah clean s1/s2 重跑：末端退化是**随机的**，不是该 seed 的固有性质

同 seed、同配置、`exp_name=w12_pixel_rerun`，原始结果保留在 `w12_pixel` 下对照。

| run | 400–475K 四点 | 500K | z | 平台 | final ckpt ×30ep |
|---|---|---|---|---|---|
| 原始 s1 | 568 537 591 507 | 418 | **−3.6** | 551 | 475.0 |
| **重跑 s1** | 518 520 450 527 | **260** | **−6.8** | 504 | **260.9** |
| 原始 s2 | 479 469 477 453 | 419 | **−4.2** | 470 | 427.9 |
| **重跑 s2** | 639 640 660 650 | **648** | **+0.1** | **647** | **638.3** |

## R2.1 预注册的两个分支都不成立，结果是混合的

预注册写的是「复现 → 记为末期方差；不复现 → 换用重跑值」。实际：
**s1 复现了（且更狠，z −6.8），s2 完全没复现。** 所以退化不是"某个 seed 会掉"，
而是**每次 run 独立地以某概率发生**：5 个 cheetah clean run 里 3 个掉、2 个没掉。

## R2.2 更要紧的发现：固定 seed 也不可复现，方差极大

**同一个 seed 2**，同配置、同代码，两次运行的平台是 **470 vs 647**（差 38%）。

TD-MPC2 设了 `torch/numpy` 种子，但 `cudnn.benchmark=True`、GPU 原子操作、
以及 CPU replay buffer 的采样时序都不确定，故 run 之间只能视为**独立抽样**，
不能视为"同一次实验的复现"。

含义：
1. **cheetah clean 的 3-seed 均值不足以支撑 ±30 以内的结论。** 5 个 run 的
   final-ckpt 值：475.0 / 427.9 / 423.1 / 260.9 / 638.3 —— 极差 377。
2. 官方 CSV 的 3 seed 跨度（453 / 570 / 590，sd 74）与我们同量级，
   **不是我们的管线更不稳**。
3. P2.2b 判据（≥480，取自官方均值 537.3 的 0.9×）在这种方差下**本身就是
   一条噪声线**：官方自己的最差 seed 453 也过不了它。

## R2.3 P2.2b 怎么写 —— 留给 cowork 定，我不单方面改

三种口径的数都摆在这里，**不做取舍**：

| 口径 | cheetah clean final ckpt | vs 判据 480 |
|---|---|---|
| 预注册的原始 3 seeds | 442.0 ± 28.7 | FAIL |
| 5 个 run 全量（含重跑） | 445.0 ± 138.5 | FAIL |
| 若只换掉两个重跑过的 seed | (260.9 + 638.3 + 423.1)/3 = 440.8 | FAIL |

三种都不过，**结论不因口径而变**。但注意第二行的 sd = 138.5：
真实不确定度远大于原先 3-seed 报出的 28.7。

**我的判断（供参考，非方法决策）**：判据 480 是从官方单点均值推出的 0.9×，
而官方那三个 seed 自己的 sd 就是 74；用一条 ±0 的线去卡一个 sd≈140 的量，
统计上没有意义。更该做的是把 cheetah clean 的 seed 数加到 5–8，
用置信区间而非单点阈值来判断"是否复现官方"。这需要 cowork 拍板。

---

# R4 Stage-1 重训（A5000）— 判据通过，但比 W1.1 弱一档，且加训救不回来

W1.1 的 checkpoint 留在已退役的 5080 上未迁移，按用户指示重训。

**数据复现无误**：采集参数复原后得到 `48000 (prev,cur) pairs`，
与 W1.1 记录的「clean+easy 合训（48000 对）」逐位一致。
配方见新增的 `hippoact/configs/stage1_dcs.yaml`，与 `default.yaml` 只差三处，
全部来自 CP6 定型（`alpha_connectivity` / `slot_iters 3` + `slot_dim 128` /
`slot_init_mode shared`）。训练 30000 步，实测 **1.11 steps/s**，耗时 7.4 h。

## R4.1 判据：通过，但明显弱于 W1.1

| | W1.1 | 重训 (ckpt_final) |
|---|---|---|
| clean 富集 / Cohen's d / AUC | 4.15 / +2.262 / 0.943 | **3.51 / +1.463 / 0.724** |
| easy 富集 / Cohen's d / AUC | 4.26 / +1.844 / 0.833 | **3.26 / +1.196 / 0.651** |

判据线 ≥3.0，**两侧都过**。

**对照全部正确**：oracle 9.54、uniform **精确 = 1.00**、walker 占画面 0.0706 ——
三项与 W1.1 逐位一致，故**不是度量出问题，是这个 encoder 确实弱一档**。
（uniform 必须精确等于解析值这条规则，在 W1.1.6 救过一次，这里再次用于
排除"尺子坏了"。）

## R4.2 加训不是解药 —— 判据随步数的走势已测

用中间 checkpoint 逐个测判据（每个 ~3 min）：

| step | clean 富集 | clean d | easy 富集 | easy d |
|---|---|---|---|---|
| 5000 | 2.89 | **+1.773** | 2.49 | +1.008 |
| 10000 | 3.68 | +1.585 | **3.75** | **+1.436** |
| 15000 | 3.13 | +1.325 | 3.62 | +1.350 |
| 20000 | 3.44 | +1.447 | 3.67 | +1.362 |
| 25000 | 3.50 | +1.433 | 3.65 | +1.299 |
| 30000 | **3.51** | +1.463 | 3.26 | +1.196 |

**10000 步即到平台**，此后富集度不再上升；Cohen's d 甚至在 5000 步最高后回落。
→ 「再多训几万步就能追上 W1.1」这条路**已被实测否决**，差距另有原因。

## R4.3 差距原因未知 —— 不编

已排除：度量（对照全对）、训练步数（R4.2）。
无法排除：W1.1 未记录的超参（`num_slots`、`slot_query_mode`、实际步数、
数据采集时的 `distraction_seed`）、单 seed 方差。
**W1.1 的完整配置没有留档，我只能保证 CP6 明确定型的那三项一致。**

顺带一处同形现象：easy 在 30000 步（3.26）低于 10000–25000 的平台（3.62–3.75），
与 §W1.2.11 的「末端略退」同形，但此处只有一条曲线，不作结论。

## R4.4 对 W2.1 E2E-0 的影响（需要 cowork 决定）

判据（TODO 给 E2E-0 设的门）**通过**，可以起 E2E-0。但必须记住：
**这个 encoder 比论文所述的 W1.1 弱一档**。若 E2E-0 不过 0.8× 判据，
**将无法区分是 adapter 的问题还是 encoder 弱一档的问题**——
这个歧义在起跑前就存在，事后无法消除。

两条路：
1. 用 `ckpt_final.pt` 直接起 E2E-0（不按判据挑 checkpoint，与 Table V 口径一致），
   把 3.51/3.26 vs 4.15/4.26 的差距写进 E2E-0 的每一个结论
2. 先查差距（但目前没有可查的假设，见 R4.3）

## R4.5 E2E-0 的集成尚未完成

adapter 代码在（`hippoact/adapters/tdmpc2_adapter.py`，commit ead6d30），但：
- **tdmpc2 侧的 `cfg.encoder_type` 分发还没接**（当前 fork 只改了环境注册）
- adapter 要的观测是 `{rgb: 224×224 ImageNet-normalized, state: qpos/qvel}`，
  而现有 DCS wrapper 只产出 64×64 纯 rgb（TD-MPC2 官方 pixel 规格）
- 需要一个 smoke test 断言 **MPPI 每个 env step 只调用一次 encoder**
  （PHASE2_PLAN §6 风险项：planning 在 latent 空间 rollout，
  若 adapter 被 MPPI 重复调用，开销会放大 512×）

这三项完成前 E2E-0 起不来。

---

# R3 W1.3 DrQ-v2 — 进行中（4/12 完成，8 个在跑）

12 runs = {walker_walk, cheetah_run} × {none, easy} × 3 seeds，
各 `num_train_frames=1_000_000`（= 500K agent steps，与 W1.2 逐点可比）。
gpu1/2/3 各 3 槽。实测 FPS：**none ≈ 48–54，easy ≈ 28–39**（3 并发/卡）。
预计 2026-08-02 20:00 前后全部完成。

## R3.1 一个必须记下的可比性细节

DrQ-v2 的训练循环在 `global_step == 500000` 时先退出、后 eval，
**故其最后一次 eval 落在 475K agent step，不是 500K**（队列日志里
`last_eval_step=475000` 即此）。与 W1.2 的 500K 点对比时要么取 475K 对 475K，
要么用 final checkpoint 重测口径（后者更干净，与 §W1.2.9 一致）。

---

# R5 本轮的四次事故（全部为我的操作失误，已修，记录以免重犯）

## R5.1 装 matplotlib 把 numpy 顶到 2.4.6，炸掉整个环境

Stage-1 的 slot 可视化需要 matplotlib，`pip install matplotlib` 连带升级了 numpy。
后果：**W1.3 的 12 个 job 全部在 `replay_buffer.add` 的断言上崩掉**，
根因是 dm_control 索引层依赖 numpy 1.x 的 `np.array(copy=False)` 语义
（numpy 2 改为报 `ValueError: Unable to avoid copy`）。

**这个坑在 W1.2 装环境时踩过一次**（distracting-control 经由老 gym 顶 numpy），
当时只在文档里记了一句，没固化成规程，于是又踩一次。
现已把 numpy pin **移到 `setup_env.sh` 的最后**并加断言：

```bash
$PY -m pip install "numpy==1.24.4" "opencv-python-headless<4.12" "protobuf==5.29.6"
$PY -c "import numpy; assert numpy.__version__.startswith('1.24')"
```

## R5.2 两个队列跑同一份 job 表 → 同一个 run 被写两次

启动 `chain_w13.sh` 时命令末尾的 `head` 报错让整条命令返回 1，
我误以为没起、又启动一次。结果两个队列各开 3 槽，
`walker_walk easy seed=1` 被**两个进程写同一个 work_dir**。
已杀掉并删除 6.5 GB 污染数据（只跑了几分钟）。

**改文件没用**——队列启动时就把 job 列表读进内存了。
故在 `run_queue.py` 里加了**磁盘抢占锁**：启动前在 `<work_dir>/.claim` 写 pid，
其他队列见到活着的 pid 就 SKIP，持有者死亡则视为过期可接管。
两个队列现在可以安全共用同一份 job 表。

## R5.3 链式脚本等错进程名 → 差点在残缺数据上开训

`chain_stage1.sh` 里写的是 `pgrep -f 'collect_stage1\.sh'`，
实际脚本名是 `collect_stage1_data.sh`，没匹配上，等待循环立即通过。
**被 clip 数量守卫接住**（`train_easy: 0 clips` → FATAL 退出）。
守卫已从"非空"收紧为"必须精确等于 1000/1000/100/100"。

## R5.4 判据脚本没先单独跑过 → gpu0 空转 3 小时

Stage-1 09:42 训完（rc=0），紧接着的判据检查立刻崩在
`from validate_semantics import build_encoder`——该文件在 `tools/diagnostics/`
而非 `tools/dcs/`，链子里没设对 `PYTHONPATH`。
训练产物完好，但 **gpu0 从 09:42 空到 12:35**。

同类问题在同一天出现过两次（另一次是 `python scripts/pretrain_stage1.py` 报
`No module named 'hippoact'`——我之前用 `python -c` / `python -m pytest`
验证"环境没问题"，那两种启动方式都会把 cwd 加进 sys.path，把问题兜住了）。

**教训**：链式脚本里每一个下游步骤，都要先用**真实调用方式**单独跑通一次
再挂进链子。用交互式 `python -c` 验证不算数。

---

# R4.5 E2E-0 集成完成 —— 但实测吞吐说明这条路按现状跑不通

三件集成全部完成，`experiments/scripts/smoke_e2e0.py` 全过。
**然后测吞吐，发现 500K 步要 22.6 天。** 集成是对的，代价不可接受。

## R4.5.1 三件集成（全过）

| 检查 | 结果 |
|---|---|
| env 契约：`obs=hippoact` | `rgb (3,224,224) uint8` + `state (24,) float32`；`cfg.obs_shape` 正确 |
| 物理未被改动 | 与 pixel 臂同 seed 同动作序列，max\|Δreward\| = **0.000e+00** |
| encoder 接线 | 架构从 ckpt 的 `encoder_arch` 快照读出（16 slots / 128 dim / 224），Stage-1 模块冻结 |
| `encode` 形状 | `(B,…) → (1,512)`、`(T,B,…) → (4,1,512)` |
| **MPPI 调用次数** | 3 次 `act()` → encoder **恰好 3 次**（若在 planning 内重编码会是 512×3） |

fork 改动仍然很小：4 文件 +62 −2，**算法零改动**。

## R4.5.2 三个集成期发现的真问题

**(1) TD-MPC2 单任务路径从未走过 dict 观测。** `act()` 调 `obs.to().unsqueeze(0)`，
`update()` 索引时间轴——普通 dict 两样都没有。改为 wrapper 直接产出 **TensorDict**
（replay buffer 本来就是 TensorDict 的），全链路即通。

**(2) DINOv2 会静默降级成随机 CNN。** 集成过程中真的触发了一次：
`Falling back to MockDinoV2Encoder (Remote end closed connection without response)`
——一次网络抖动，backbone 就被换成随机权重，训练照跑、数字全废。
只是碰巧被 ckpt 权重不匹配的 assert 拦住，那是运气不是设计。已修三处：
`torch.hub.load(..., trust_repo=True, skip_validation=True)` 用本地缓存、
`HIPPOACT_STRICT_DINO=1` 禁止回退、adapter 里硬断言（显式 `HIPPOACT_FORCE_MOCK=1`
仍放行，要防的是**静默**回退而非离线测试）。

**(3) `encode_frame` 无条件跑 slot decoder，而 RL 侧根本不用它。**
router 只吃 slot 向量；decoder 产出的 `recon`/`alpha` 只有 Stage-1 的重建与
连通性损失需要。而它会物化 `(B,K,N,D_v)`——B=1024 时 **6 GB**，直接 OOM。
加 `decode=False` 后 OOM 消失、单步 5.4 s → 3.9 s。**这是精确的，不是近似**。

## R4.5.3 吞吐实测：瓶颈是 update()，占 99.6%

`experiments/scripts/bench_e2e0.py`（gpu1，与 3 个 DrQ-v2 共卡，故偏悲观）：

```
act():    1 帧                 14.6 ms   (  69 帧/s)
update(): 256 帧             1008.4 ms   ( 254 帧/s)
update(): 1024 帧 (4×256)    3882.3 ms   ( 264 帧/s)

每个 env step = act(1) + update(1024) = 3897 ms  ->  0.26 SPS
500K 步 = 541 h = 22.6 天
```

原因是结构性的：TD-MPC2 每个 env step 做 1 次 update，而一次 update 要把
`(horizon+1) × batch = 4 × 256 = 1024` 帧 224² 过一遍 DINOv2。
**pixel baseline 的 encoder 是个小 CNN，HippoAct 是 ViT-S** ——
同样的 1024 帧，前者微秒级，后者近 4 秒。

## R4.5.4 解法：缓存 slots（**精确**，非近似）

E2E-0 的 Stage-1 编码器是**冻结**的，因此某一帧的 slots 永不改变：
**在入 buffer 时编码一次，与在采样时编码，数学上完全等价。**

| | 现状 | 缓存 slots |
|---|---|---|
| 每步 DINOv2 前向 | 1024 帧 | **1 帧** |
| 预计 SPS | 0.26 | **~69**（被 act() 卡住） |
| 500K 步 | **22.6 天** | **~2 小时** |
| buffer/帧 | 147 KB（uint8 图） | **8 KB**（16×128 float） |
| buffer @500K | 75 GB | **4 GB** |

**限制**：仅当编码器冻结时成立。E2E-1/E2E-2 若要微调 slot attention 就不能用
（缓存 DINOv2 patch 特征不可行：256×384 fp16 = 197 KB/帧，比存原图还大）。
届时要么冻结 slot attention，要么接受慢速。

**这需要 cowork 拍板**，因为它不再是"只换 encoder"——buffer 里存的东西变了。
其余选项都更差：减 batch（改 TD-MPC2 超参，破坏"算法零改动"）、
降分辨率（DINOv2/14 需要 224，Stage-1 得重训）、减步数（削弱对比）。

**在拍板前 E2E-0 不起跑**——用现状跑等于烧 22 天换一个本可 2 小时得到的相同结果。

---

# R4.6 E2E-0 快路径：把冻结编码器放进 env —— 并因此发现编码器不是确定性函数

按拍板实现「冻结 encoder 前移到环境」。实现过程中断言把两个**比速度更要紧**的
问题抓了出来。

## R4.6.1 实现

`obs=state` + `hippoact_precompute=<ckpt>` → env 直接产出
`[flatten(fast_slots) ⊕ q_t]` = 16×128 + 24 = **2072 维向量**，
**TD-MPC2 本体一行不改**，用它自己的 state encoder 当 z_mlp。

冻结部分抽成 `hippoact/adapters/slot_features.py::SlotFeatureExtractor`，
env 侧与 in-model 侧**共用同一个类** —— 等价性是结构性的，不靠断言。
in-model 路径（`encoder_type=hippoact`）保留，供 E2E-1/2 把编码器放进训练图时用。

## R4.6.2 实测

| | in-model（原方案） | env 前移（本方案） |
|---|---|---|
| 每步 DINOv2 前向 | 1024 帧 | **1 帧** |
| 稳态 SPS | 0.26 | **6.3** |
| 500K 步 | **22.6 天** | **~22 小时** |
| buffer/帧 | 147 KB | **8.75 KB**（实测 0.07 GB / 8000） |
| buffer @500K | 75 GB | **4.4 GB** |
| E2E-0 的 fork 改动 | 4 文件 +62 −2 | **0**（只多一个 config 键） |

smoke test 四项全过，**env 侧与 in-model 的特征 max\|Δ\| = 0.000e+00**。

比预估的 3 小时慢，因为下限不是 update 而是**每个 env step 的 encoder**
（14.6 ms）+ 224² 渲染。6.3 SPS 是这两项的合成。

## R4.6.3 顺带修掉一个我自己引入的架构偏离

TD-MPC2 的两个 encoder（state 与 rgb）**末尾都有 `SimNorm`**——把 latent 切成
8 维一组做 softmax，它的 dynamics / reward / Q 全建立在这个单纯形结构上。
**我手写的 `z_mlp` 末尾是裸 Linear，没有 SimNorm**，等于给 TD-MPC2 喂一个它
从未设计过的潜空间。改用 TD-MPC2 自带 state encoder 后自动修好。

## R4.6.4 **编码器不是确定性函数**（本轮最重要的发现）

写「env 侧 == in-model」这条断言时它 FAIL 了，`max|Δ| = 15.8`。查下去：

```
同一张图，编码两次:  max|Δ| = 8.5
```

原因：`slot_query_mode: sampled` 下 `slots_mu` 形状是 `(1,1,D)`——**16 个 slot
共享同一个 mu，全靠 `sample_init` 里的噪声打破对称**。所以每次 forward 都是
一次新抽样，同一帧编码两次得到不同的 slots。

这**同时**破坏三件事，且与是否预计算无关：
1. buffer 里存的表征与重新编码得到的不一致
2. `act()` 把同一个观测映射到不同的 latent
3. MPC 从一个带噪的 latent 出发做规划

**不能简单去掉噪声**（mu 共享，去噪后 16 个 slot 会完全相同、分解坍缩）。
已实现的解法是**固定一次抽样**：`slot_init_seed` 播种一次、注册为 buffer，
推理时复用。这与 CP6/adapter already 采用的「部署用 argmax 而非 Gumbel」
是同一个理由。

验证：

```
两个独立实例、同 slot_init_seed=0     max|Δ| = 0.00e+00
slot_init_seed = 0  vs  1            max|Δ| = 16.16
```

**这个 init 的选择是任意的，且影响很大**——与 CP5e 早已测到的
「同图换 init → alpha 余弦 0.189」是同一现象。所以 `slot_init_seed`
**是编码器身份的一部分，必须与 checkpoint 一起报告**；换个 seed 就是换个编码器。

论文含义：Stage-1 训出来的编码器**并不定义唯一的分解**。这在 CP5e 已被量化，
但它对 Stage-2 的后果直到现在才显出来。§V 应写明。

---

# R4.7 Stage-1 seed 方差（5 个 seed）—— R4 定论：富集度上没有系统性差距

播种补上后（`c1d5f69`）跑了 5 个独立 seed，同一配方、同一数据。

| seed | clean 富集 | easy 富集 | clean d | easy d |
|---|---|---|---|---|
| 1 | 3.47 | 3.57 | +1.157 | +1.320 |
| 2 | 4.06 | 4.19 | +1.970 | +1.653 |
| 3 | 4.00 | 4.07 | +1.898 | +1.512 |
| 4 | 3.87 | 3.87 | +1.818 | +1.470 |
| 5 | 3.42 | 3.66 | +1.588 | +1.364 |
| *未播种那次* | *3.51* | *3.26* | *+1.463* | *+1.196* |

| | mean | sd | 范围 | **W1.1** | z |
|---|---|---|---|---|---|
| clean 富集 | 3.76 | 0.30 | 3.42–4.06 | **4.15** | **+1.29** |
| easy 富集 | 3.87 | 0.26 | 3.57–4.19 | **4.26** | **+1.48** |
| clean d | 1.69 | 0.33 | 1.16–1.97 | **2.26** | +1.75 |
| easy d | 1.46 | 0.13 | 1.32–1.65 | **1.84** | **+2.90** |

## R4.7.1 结论

**富集度上 W1.1 完全落在我们的分布内**（z = +1.29 / +1.48），只是抽得偏好。
「重训的 encoder 弱一档」这个顾虑**不成立**——它是 Stage-1 的 run-to-run 方差，
与「Stage-1 此前从未播种」（见 `c1d5f69`）完全自洽：W1.1 与重训本来就是
两次独立抽样，从来不存在「同 seed 应当一致」这个前提。

→ **R4 ruling 3a（从 5080 取 W1.1 ckpt）已无必要**，歧义靠方差测量消除了。

## R4.7.2 一处不能用抽样解释的

**easy 上的 Cohen's d，W1.1 是 z = +2.90，5 个 seed 全部低于它。**
富集度（分离的幅度）在分布内，但 d（分离的显著性）不在。原因未知，不编。
可能与 W1.1 那次未记录的超参有关，也可能是 5 个样本估 sd 偏小。
**如实记录**；若论文要引用 easy 的 d，应报本轮 5-seed 的 1.46 ± 0.13
而非 W1.1 的单点 1.844。

## R4.7.3 E2E-0 该用哪个 ckpt —— 不挑，直接测传导

5 个 encoder 的富集度跨度 3.42–4.06。**按判据从中挑最好的去跑 E2E-0 是
cherry-picking**；随便固定一个又无法回答「encoder 方差会不会传导到 return」。

三张卡空着，故**同时跑 3 个 E2E-0，分别用 Stage-1 seed 1 / 3 / 5**
（跨度覆盖 3.42 / 4.00 / 3.47，即分布的低-中-高，规则事先声明、非按结果挑）。
一轮 22 h 同时得到：E2E-0 是否过判据，以及 encoder 方差传导多少。

**判据（TODO W2.1 多点位，0.8× pixel 同点位）**，pixel 值取 W1.2 的
`dcs-easy-walker-walk` 3-seed 均值：

| agent step | pixel | E2E-0 判据 |
|---|---|---|
| 50K | 213.8 | **≥ 171.1** |
| 100K | 391.4 | **≥ 313.1** |
| 250K | 675.5 | **≥ 540.4** |
| 500K | 878.7 | **≥ 703.0** |

**预注册**：早期点位比 final 重要（§W1.2.12：easy 干扰的代价几乎全在样本效率，
50K 时 pixel 只有 clean 的 0.39）。若 HippoAct 的对象中心表征有用，
**应当先在 50K/100K 显出优势**；只在 500K 追平不算赢，那个点位 pixel 已近饱和。

## R4.7.4 E2E-0 已起跑并按指示停下（跑到 ~24K 步）

三个 run 起来后按用户指示停止，用于先归档结果。停止前跑到 23.5K–24.5K 步。

**取得一个重要的修正数据**：独占一张卡时 **~25 SPS**
（24,500 步 / 16.3 min），而不是 §R4.6.2 报的 6.3 SPS —— 那次探针是和
3 个 DrQ-v2 挤在 gpu1 上测的。

→ **500K 步实际约 5.5 小时**，不是 22 小时。相对最初的 22.6 天是 **~100×**。

停止时的 24K 步过早，不足以对判据说任何话（首个判据点位在 50K）。
预注册（§R4.7.3 的多点位表与「早期点位比 final 重要」）保持不变，
重启后从头跑。

---

# R3.2 W1.3 DrQ-v2 walker 补种 —— 方向一致，但**做不出显著性**

首轮 walker_walk 的异常（none 4/6 学会、easy 1/6）无法区分「干扰致命」与
「walker_walk 固有失败率」，故补种到每臂更多 seed。失败形态一律是
**从第 0 步就不动**（walker-walk 随机策略 ≈ 20–24），不是学到一半崩。

| 臂 | 学会 / 总数 | per-seed（475K agent step） |
|---|---|---|
| walker_walk **none** | **4 / 6** | 967, 953, **23**, 881, **22**, 958 |
| walker_walk **easy** | **2 / 9** | 29, 28, 28, 21, 33, **870**, 14*, **860***, 30* |
| cheetah_run none | 3 / 3 | 707, 733, 714 |
| cheetah_run easy | 3 / 3 | 570, 621, 625 |

（`*` = seed 7–9，本文写作时跑到 ~72%，值取当前最后一次 eval）

## R3.2.1 显著性：不成立

```
clean 4/6 vs easy 1/6 (首轮)          Fisher 单侧 p = 0.121
clean 4/6 vs easy 2/9 (补种后当前)     Fisher 单侧 p = 0.119
```

**我此前报过 p = 0.046「显著」，那是基于 easy 0/5 的中间态，已作废** ——
第 6 个 seed（870）和第 8 个（860）都学会了。补到 9 个 seed 后 p 几乎没动。

## R3.2.2 但形态本身是个结果

**DrQ-v2 在带干扰的 walker 上不是学不会，而是大概率起不来。**
学会的那两个拿到 **870 / 860**，与 TD-MPC2 的 853.7 相当。
失败的七个全部停在随机策略水平。这是**双峰**，不是均值下降：

```
easy 臂 9 个 seed:  {14, 21, 28, 28, 29, 30, 33}  和  {860, 870}
```

→ 该写成「优化可靠性」问题而非「能力上限」问题。均值（±sd）在这里是误导性的
统计量，论文应报**学会率 + 学会者的分数**两个数，并画出双峰。

同一格 TD-MPC2 是 3/3 学会、853.7 ± 89.0 —— **model-based 在干扰下不仅更强，
更重要的是更可靠**。这比单纯的分数差更值得写。

## R3.2.3 一个必须披露的基线问题

**DrQ-v2 在干净 walker 上也有 2/6 失败**（23、22 分，同样从第 0 步不动），
而其官方结果是稳定 950+。我没有查出原因。

不披露的话，「easy 2/9 vs clean 4/6」这个对比会被质疑基线本身没调好。
候选方向（均未验证）：`num_train_frames` 我设 1M 而其 walker_walk 配置默认
1.1M；`nstep=1 / batch=512` 的 per-task 覆盖与我们的截断步数交互；
或就是该实现的已知不稳定性。

---

# R3.3 DrQ-v2 clean walker 2/6 失败 —— 排查结束：**是 DrQ-v2 的已知性质，有其官方数据为证**

按拍板第 4 条排查。

## R3.3.1 配置差异：只有一处，且不可能是原因

我们对官方 `cfgs/task/walker_walk.yaml`（`easy` 预设 + `nstep=1` + `batch_size=512`）
的唯一覆盖是 `num_train_frames` = 1M（官方默认 1.1M）。

该参数**只决定何时停**：`stddev_schedule` 按 `global_step` 走、
`num_seed_frames` 独立、`Until(num_train_frames)` 仅是停止条件。
**它不可能导致「从第 0 步就不学」。这条线索排除。**

## R3.3.2 决定性证据：官方自己的曲线里就有同样的失败

`third_party/drqv2/curves/dmc_walker_walk.csv` 是 DrQ-v2 作者随仓库发布的
10-seed 曲线。@1M frames：

```
950  962  949  **24**  960  956  960  **29**  910  968
```

**官方 10 个 seed 里有 2 个卡在 24 / 29 分**，与我们观察到的死平形态完全一致。

| | 失败 / 总数 | 失败率 |
|---|---|---|
| 官方 clean（作者发布） | **2 / 10** | 20% |
| 我们 clean | **2 / 6** | 33% |
| 合并 clean | **4 / 16** | **25%** |
| 我们 easy | **7 / 9** | **78%** |

我们的 2/6 与官方 2/10 完全相容。**排查结论：这是 DrQ-v2 在 walker-walk 上的
已知不稳定性，不是我们的集成或配置问题。** 论文如实披露并引其官方曲线即可，
不必列「已排查项」——根因找到了。

## R3.3.3 用官方数据当基线后，easy 的效应变显著

此前拿我们自己的 clean（4/6 学会，n=6）当基线，Fisher p = 0.119，做不出显著性。
把官方的 10 个 seed 并入基线后（clean 失败 4/16 = 25% vs easy 失败 7/9 = 78%）：

```
Fisher 单侧 p = 0.016   **显著**
```

→ **背景干扰确实把 DrQ-v2 在 walker 上的失败率从 ~25% 推到 ~78%。**
这个结论此前之所以做不出来，是因为基线样本太小（n=6）；
官方发布的 10-seed 曲线是免费的基线扩充。

## R3.3.4 还有一个「晚起飞」的形态

`none s4` 在 200K frames 时只有 19.2 分（看起来与失败者无异），
500–600K 才起飞，最终 879。而真正失败的 s3/s5 **全程 20–26 分、零上升趋势**。

所以「失败」不是「快起飞了被截断」——延长到官方的 1.1M 也救不回来。
但它说明**判定失败需要看完整曲线**，不能只看某个中间点位。


---

# R4.8 E2E-0 的对比是坏的：**proprio 混淆**（本轮最重要的发现）

E2E-0 三个 run 到 350K 步的数字很好看，但**不能当作 HippoAct 赢**。

## R4.8.1 现象

`dcs-easy-walker-walk`，RL seed 均为 1，agent step：

| agent step | E2E-0 (n=3) | pixel 基线 (n=3) | 预注册判据 | **纯 proprio（官方 state-obs）** |
|---:|---:|---:|---:|---:|
| 50K  | 559.9 | 213.8 | 171 | **961.9** |
| 100K | 925.2 | 391.4 | 313 | **973.4** |
| 200K | 968.4 | 572.2 | 458 | **976.4** |
| 250K | 976.1 | 675.5 | 540 | **978.7** |
| 350K | 971.4 | 776.2 | —   | ~980 |

最后一列来自 `third_party/tdmpc2/results/tdmpc2/walker-walk.csv`（作者随仓库发布的
3-seed state-obs 曲线，env step 已除以 action_repeat=2 换算成 agent step）。

**walker-walk 只用 24 维本体感受就能解到 979。**

## R4.8.2 为什么这让整个对比失效

E2E-0 的观测是 `[flatten(fast_slots) ⊕ q_t]` = **2048 + 24** 维。
它的曲线与纯 proprio 基线重合（976 vs 979），50K 处还更差（560 vs 962）。

→ **2048 维 slot 特征的贡献不可测**；能观察到的只有它拖慢了早期学习。

而 pixel 基线**只看像素**。所以这张表测的是「proprio vs 像素」，
不是「slot 表征 vs 像素表征」。判据 171/313/458/540 是照 pixel 基线定的，
对一个拿到 proprio 的 agent 没有意义。

更根本的一点：**背景干扰只作用于视觉通道，proprio 对它免疫。**
只要观测里有 proprio，Q2 想测的「背景鲁棒性」就可以靠无视视觉白拿——
在 walker-walk 上这个实验设计**根本测不到想测的东西**。

这不是实现 bug。`hippoact_include_proprio` 默认 true 是照 E2E-0 契约
`z = MLP(flatten(S_fg) ⊕ q)` 写的，契约本身在这个任务上就有这个洞。

## R4.8.3 补的两组对照（已起跑，2026-08-04 01:55）

| exp_name | 观测 | 目的 |
|---|---|---|
| `e2e0vis_s{1,3,5}` | 2048 维 slot，**无 proprio** | 唯一与 pixel 基线可比的 E2E-0 |
| `proprio_only` | 24 维 proprio，无视觉 | 把上表最后一列钉死在我们自己的环境里，不引官方数字 |

`e2e0vis` 除 `hippoact_include_proprio=false` 外与 `e2e0_s*` 逐字相同
（同 Stage-1 seed 1/3/5、同 RL seed 1、同 500K 步）。
纯视觉路径已 smoke 过：6K 步跑通，观测维度 2048，
banner 记录 `include_proprio=False`。

原来那三个带 proprio 的 run **不停**——它们是混淆存在的证据，留作记录。

## R4.8.4 待定：判据与任务

- 预注册判据（100K≥313 / 250K≥540）是对 pixel 基线定的，
  `e2e0vis` 可以直接沿用；带 proprio 的那三个 run 不适用，其「达标」作废。
- 如果 `proprio_only` 在 easy 上确实到 ~975，则 **walker-walk 不适合做 Q2 主任务**
  ——需要一个 proprio 解不掉的任务。cheetah-run 官方 state-obs 也很高，
  同样要查。这个决定留给用户。

## R4.8.5 顺带：GPU 利用率只有 20% 的原因（不是问题）

单步 37 ms，其中环境侧只占 11.6 ms（实测，GPU1 空载）：

```
env.step（物理 + DCS 背景合成）    5.72 ms
env.render 224x224 (EGL)          1.14 ms
SlotFeatureExtractor 前向 (b=1)   4.71 ms
```

剩下 ~25 ms 是 TD-MPC2 自己的 update + MPPI 规划（6 轮 × 512 条轨迹的小 MLP）。
几百个微秒级 kernel，GPU 大部分时间在等 CPU 发射指令 —— **launch-latency bound**。
每个进程恰好占满 1 个核，机器有 64 核。

→ 单个 run 快不了，但可以并发。已从 3 个并发提到 7 个，四张卡都在用。

---

# R4.9 E2E-0 三臂全部跑完 —— **判据不通过，且根因写在我们自己的代码注释里**

7 个 run × 500K agent step 全部完成（2026-08-04 09:46）。
数据在 `experiments/results/e2e0/`，`export_e2e0.py` 可重生成。

## R4.9.1 结果

`dcs-easy-walker-walk`，RL seed 均为 1，三臂只差观测：

| agent step | slots+proprio (n=3) | **slots only (n=3)** | proprio only (n=1) | pixel 基线 (n=3) | 判据 |
|---:|---:|---:|---:|---:|---:|
| 50K | 559.9 | **121.3** | 953.0 | 213.8 | 171 |
| 100K | 925.2 | **122.1** | 969.7 | 391.4 | 313 |
| 200K | 968.4 | **224.7** | 976.7 | 572.2 | 458 |
| 250K | 976.1 | **225.1** | 979.5 | 675.5 | 540 |
| 500K | 972.7 | **305.5** | 974.3 | **878.7** | — |

三个结论，一个比一个重：

1. **proprio 混淆被我们自己的对照钉死**（不再需要引官方数字）：
   `proprio_only` = 974.3，`slots+proprio` = 972.7。**差 1.6 分。**
   2048 维 slot 特征的边际贡献在测量噪声内。§R4.8 的判断成立。
   而且 proprio_only 在 **50K 就到 953** —— walker-walk 对纯本体感受近乎平凡。

2. **纯视觉 E2E-0 判据全线不通过。** 500K 时 305.5 / 878.7 = **0.35×**，
   判据是 ≥0.8×。四个预注册点位全部不过（121/171、122/313、225/458、225/540）。

3. **冻结的 slot 表征显著劣于原始像素**（305 vs 879）。
   这不是"没有增益"，是**倒退 2.9 倍**。

## R4.9.2 根因：`shared` 模式的 slot 跨帧不跟踪物体，而 E2E-0 的 flatten 要求它跟踪

ckpt 的 `trainer_config` 实读：**`slot_init_mode = shared`**。

`hippoact/training/stage1.py:160` 自己的注释：

> CP5c fix: pair-wide shared init eliminates noise floor but **freezes the
> spatial partition — slots do NOT track objects across frames** (CP5d
> finding: alpha centroid moves 0.4 patch while objects move 2.3 patch).

同文件 `:215`，训练时算 slow loss 前先做**最近邻匹配**：

```python
prev_for_slow = match_slots_nn(slots.detach(), slots_prev.detach())
```

—— 这行代码存在本身就是证据：`shared` 模式下 **slot 下标跨帧没有对应关系**，
所以损失函数必须先匹配再比较。

**而 E2E-0 的契约 `z = MLP(flatten(S_fg) ⊕ q)` 是按固定下标拼接的，
对置换敏感。** 训练目标做到了置换容忍，下游消费方要求置换稳定 ——
两者不兼容。这不是调参问题，是接口不匹配。

## R4.9.3 独立实测（`experiments/scripts/diag_slot_features.py`）

不靠注释，自己在 300 步真实 rollout 上量了一遍
（`experiments/results/e2e0/slot_feature_diagnosis.json`）：

| 量 | 实测 | 含义 |
|---|---|---|
| `same_slot_is_nearest` | **0.490** | 只有一半的 slot 在下一帧仍是自己的最近邻 —— **slot 一直在置换** |
| `corr(Δobs, Δproprio)` | **0.035** | 观测的变化量与物理状态的变化量**几乎零相关** |
| `corr(Δ未门控 slots, Δproprio)` | 0.077 | 去掉门控也一样，**问题在 slot 本身不在路由** |
| gate 每步翻转的 slot 数 | 3.64 / 16 | |
| 每步归零或复活的维度 | **465 / 2048** | 23% 的观测维度每一步跳变 |
| 有翻转的帧占比 | 98.7% | |
| `obs_delta_mean` | 62.2 | 而单个 slot 的模长只有 20.6 —— **每步变化是 slot 自身尺度的 3 倍** |

关键的一条是 `corr(Δobs, Δproprio) = 0.035`。TD-MPC2 的全部机制建立在
**能从 (z_t, a_t) 预测 z_{t+1}** 之上（consistency loss + MPPI 在 latent 里 rollout）。
观测变化与状态变化不相关，等于 world model 的学习目标本身不可学，
MPPI 在一个无意义的空间里规划。**305 分这个数字是可解释的**：
策略从边际统计量里学到了一点东西，但世界模型是废的。

门控翻转（465 维/步）曾是我的首要嫌疑，但 `obs_delta` 在翻转帧(62.2)与
非翻转帧(56.9)几乎一样，且去掉门控后相关性仍是 0.077 ——
**门控不是主因，slot 本身的时间不连续才是。** 记录下来避免后人重查。

## R4.9.4 一个重要推论：这个失败模式**可能只打 E2E-0**

`flatten` 是 E2E-0 独有的读出方式。E2E-1 的 binding transformer 用注意力
处理 slot 集合，**对置换等变**，原理上免疫本失败模式。

所以 build-up ablation 的第一级可能恰好是最差的一级。
**不能从 E2E-0 失败推出方法失败。**

## R4.9.5 三条候选路线（未选，需 cowork 拍板）

| | 做法 | 成本 | 风险 |
|---|---|---|---|
| A | Stage-1 换 `carryover_norm` 重训（该模式存在的目的正是保住 slot 身份，见 `stage1.py:184` CP5g 注释） | **8.4 h/seed**（实测）× 3 + E2E 重跑 5.5h×3 | 假设未验证：carryover_norm 能否真的把 `same_slot_is_nearest` 推高，需先在 100 帧上量一次再决定要不要训 |
| B | 读出改成置换不变（对 slot 求和/池化，或按 canonical key 排序） | 零训练，改 adapter | 改了 E2E-0 契约；池化会丢掉 binding transformer 要用的结构 |
| C | 直接跳到 E2E-1（binding transformer 天然置换等变） | 5.5h×3 | 跳过 build-up ablation 的一级，A5 行会缺 |

**我的建议：先做 A 的前置测量**（零 GPU 训练成本）——
拿现有 ckpt 用 `carryover_norm` 的推理方式（上一帧输出当下一帧 init）
重跑 `diag_slot_features.py`，看 `same_slot_is_nearest` 是否显著上升。
上升 → 走 A 有依据；不上升 → 直接走 C。

## R4.9.6 顺带修掉一个真 bug：`retention_eval.py` 把 HippoAct 臂路由错了

链式 Q2 评测（`chain_e2e0_q2.sh`）在 `difficulty=none` 上崩了：

```
size mismatch for _encoder.state.0.weight:
  ckpt torch.Size([256, 2072]) vs model torch.Size([256, 24])
```

原因：`eval_task_name(base, "none")` 返回裸 `walker-walk`，走的是 **TD-MPC2
自己的 `envs/dmcontrol.py`**，那条路根本不看 `hippoact_precompute` ——
wrapper 被静默丢掉，agent 拿到的是裸 24 维 proprio。

**这次是因为编码器宽度恰好不同才炸出来的**，不是因为有检查。
若两者宽度相同，它会安静地跑完并给出一个错误的数。

修法：HippoAct 臂一律走 `dcs-none-*`（`dcs_env.py` 在 `difficulty=none`
时直接调 `dm_control.suite.load`，与裸名字是同一个环境）。
pixel 基线保持走裸名字不变（与官方逐位一致）。

---

# R4.10 R4.9 拍板的两条前置测量**都不通过**，且我在 §R4.9 给的根因**部分作废**

拍板要求「训练前先量数据」。量完了，结论与拍板的前提不符，所以**没有起跑任何 run**。

## R4.10.1 Ruling 1（E2E-0p 均值池化）：不通过其自己的预注册门槛

预注册：`corr(Δobs, Δproprio)` 应恢复到 **> 0.3**。实测（300 步 rollout）：

| readout | shared（现状） | carryover_norm |
|---|---:|---:|
| `flatten_fast`（E2E-0 现状） | 0.068 | 0.077 |
| **`mean_over_fast`（E2E-0p）** | **0.017** | 0.020 |
| `sum_fast / K` | 0.069 | 0.054 |
| `mean_all_slots`（无门控） | 0.045 | 0.028 |

**池化不但没到 0.3，还比 flatten 更差。** 八种组合全部在 0.02–0.08。

## R4.10.2 Ruling 3（A 线 carryover_norm）：涨幅微小，按拍板应关闭

`same_slot_is_nearest`：**0.527 → 0.584**。不是「显著上升」。
按拍板第 3 条「否则关闭」，**A 线关闭，ICRA 内不重训 Stage-1**。

## R4.10.3 我在 §R4.9 给的根因链条**部分作废**

§R4.9.2 的论证是「slot 跨帧置换 → flatten 被打乱 → world model 不可学」。
两条新证据推翻了它的关键环节：

1. **置换不变的读出反而更差**（池化 0.017 < flatten 0.068）。
   如果置换是主因，去掉置换敏感性应当改善。没有。
2. **状态确实在表征里。** 线性探针（1500 帧、随机切分、λ 扫描）：

| feature | proprio_all(24) | **proprio_pos(15)** | proprio_vel(9) |
|---|---:|---:|---:|
| `flatten_fast`（E2E-0） | 0.001 | **0.716** | −0.005 |
| `mean_over_fast`（E2E-0p） | −0.001 | 0.634 | −0.002 |
| DINOv2 patch mean（对照） | −0.000 | 0.648 | −0.003 |
| `flatten_fast` × 2 帧 | 0.010 | 0.709 | 0.004 |
| `flatten_fast` × 3 帧 | 0.007 | 0.722 | 0.001 |

**单帧 slot 特征把 walker 的位姿编码到 R²=0.716，比 DINOv2 原始特征还好
（0.648）。表征不是垃圾。** 「信息不存在」的说法不成立。

`corr(Δobs,Δproprio)=0.035` 这个数**仍然是真的**，但它不足以支撑当时的推论——
它比的是变化量的模长，而 walker 的 24 维里 9 维是速度，单帧本来就看不到。
**是我的指标钝，不是表征坏。** §R4.9.2/§R4.9.3 的措辞按此更正。

（探针本身也返工过一次：最初按时间切分 + 固定 λ，得到**所有特征包括 DINOv2
对照全部 R²<0**。原因是 1500 步 = 3 个 episode，`easy` 下每个 episode 换一段
DAVIS 背景，时序切分测的是跨背景迁移；而 2048 维对 1050 样本，单一 λ 也不对。
改随机切分 + λ 扫描后才是上表。**负 R² 出现在对照组上时，先查工具。**）

## R4.10.4 剩下的、证据支持的缺陷：**单帧观测不是 Markov 状态**

`Pixels(env, cfg, num_frames=3)` —— **pixel 基线堆 3 帧**（`dmcontrol.py:67`）。
TD-MPC2 的 state obs 直接含 9 维速度。而 `HippoActSlots` 只发**单帧** slots：

| 臂 | 位置 | 速度 |
|---|---|---|
| state obs | ✅ | ✅（直接给） |
| pixel 基线 | ✅ | ✅（3 帧堆叠） |
| **E2E-0 纯视觉** | ✅ | ❌ **两者都没有** |

walker-walk 是运动控制任务，单帧对它**在原理上就不是 Markov 状态**。
这是 E2E-0 接线的确定缺陷，且**与 pixel 基线的对比在这一点上不公平**。

**但要说清楚：上表显示堆 3 帧后速度的线性可解码性仍是 ~0.00。**
所以「堆帧就能修好」**没有实测支持**——只能说单帧确定不足，不能说堆帧确定够。
（线性探针对冻结特征是弱工具；pixel 基线靠的是端到端训练的 CNN，
能提取线性探针提取不到的运动信息。两者不可直接互推。）

## R4.10.5 Q2 零样本网格（21/21 完成，30 episodes/格）

| ckpt | none | easy | hard | invariance (none/easy) | retention (hard/easy) |
|---|---:|---:|---:|---:|---:|
| e2e0_s1（含 proprio） | 976.3 | 976.5 | 974.0 | **1.000** | **0.997** |
| e2e0_s3 | 974.9 | 976.5 | 975.9 | 0.998 | 0.999 |
| e2e0_s5 | 976.2 | 975.1 | 975.6 | 1.001 | 1.001 |
| **proprio_only** | 975.9 | 977.1 | 975.9 | **0.999** | **0.999** |
| e2e0vis_s1（纯视觉） | 172.5 | 247.0 | 213.3 | 0.698 | 0.864 |
| e2e0vis_s3 | 226.9 | 281.8 | 208.2 | 0.805 | 0.739 |
| e2e0vis_s5 | 289.1 | 319.8 | 239.6 | 0.904 | 0.749 |
| *pixel 基线（R1）* | — | — | — | *0.65* | *0.74* |

**这是 §R4.8 那条论证的最后一块：`proprio_only` 拿到 0.999 / 0.999。**
一个完全不看图像的 agent 在背景鲁棒性上拿满分。
→ **walker-walk 上的平坦不变性是平凡的，不构成任何证据。**
带 proprio 的三个 E2E-0 拿到 1.000 同理作废。

纯视觉三个 seed：invariance 0.80±0.10、retention 0.78±0.07，
**方向上确实高于 pixel 的 0.65 / 0.74**。但绝对分只有 247（pixel easy ~854），
**在接近随机策略（~24）的量级上比值**，这个「更鲁棒」不能单独成立为结论。
如实记录，不做强主张。

## R4.10.6 给 cowork 的三条

1. **Ruling 1 按原样不应起跑** —— 它自己的预注册门槛没过，且池化在探针上比
   flatten 差。要改读出，需要新的理由，不是「置换不变」这个理由。
2. **Ruling 3 关闭**（拍板已预写此分支）。
3. **Ruling 2（E2E-1）不受影响，仍是最优先线。** §R4.9.4 的推论要弱化措辞：
   E2E-1 的价值不再是「免疫置换失败模式」（那个失败模式没被证实），
   而是「可训练的 encoder 能学到冻结特征上线性不可得的东西」——
   探针显示单帧位置 R²=0.716 已相当高，说明**瓶颈不在特征信息量，
   在如何把它变成可预测的动力学**，这正是可训练 binding transformer 的用武之地。
   建议同时把**帧堆叠**加进 E2E-1（补上 Markov 性，且与 pixel 基线口径一致）。
