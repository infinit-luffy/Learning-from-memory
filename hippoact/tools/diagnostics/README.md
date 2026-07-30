# 诊断脚本集

CP4 → CP5g → Step M1 这条诊断链上用到的全部 eval-only 脚本。
每个都不改动模型，只读 checkpoint 做测量。

对应论文规划中的 §IV.G "Diagnostic Protocol" / "Measurement Validation" 小节。

---

## 通用用法

```bash
# 从 hippoact/ 目录运行；脚本自身目录会自动进 sys.path，无需设 PYTHONPATH
python tools/diagnostics/<script>.py \
    --config <与 checkpoint 架构匹配的 yaml> \
    --ckpt   outputs/stage1/ckpt_final.pt \
    --data-dir data/frames/synthetic_diag
```

`--config` 只需在 `encoder.*` 段（`slot_dim` / `slot_iters` / `num_slots` /
`image_size` / `dino_model` …）与 checkpoint 一致，否则 `load_state_dict`
会静默丢参数（脚本用 `strict=False`）。

**推荐诊断数据集**（交集掩膜有效样本 98%，见下「度量约定」）：
```bash
python scripts/make_synthetic_data.py --out data/frames/synthetic_diag \
    --min-radius 14 --max-radius 26 --n-clips 400
```

---

## 度量约定（三条硬规则，违反会得到看似合理但错误的数）

### 1. 物体掩膜用**交集**，不用并集

`motion = |cur − prev|` 覆盖物体的**旧位置和新位置的并集**（哑铃形）。
一个停在两个位置中间不动的 slot 也能拿高分，无法区分「跟踪物体」与
「恰好停在运动发生的区域」。位移是亚 patch 时这个混淆无害，位移 >1 patch 就有害。

正确定义（需要 3 帧；4 帧 clip 可让 `obj_t` 与 `obj_{t+1}` 都用交集）：
```
obj_t = motion(t−1, t) ∩ motion(t, t+1)
```
两个差分都包含 t 时刻的物体，交集即物体在 t 的位置。

### 2. 新度量先跑 oracle / uniform 正负对照

任何声称「我们验证了 slot 语义」的度量，都应先证明它在 oracle 上给出正信号。
`validate_semantics.py` 内置这个对照：

| 对照 | motion score | 含义 |
|---|---|---|
| oracle slot（alpha 直接等于 motion mask） | **18.62** | 度量能检出完美跟踪 |
| uniform slot（alpha 均匀） | **1.00** | 定义上的 chance 基线 |

没有这一步，一个负的 Cohen's d 无法区分「模型坏了」和「尺子坏了」。

### 3. 「上界低于实测」是度量 bug 的检出信号

`m1_oracle.py` 初版把代价矩阵写成 `mass2[j] − base_i`，这是**可加分离**的：
任何双射的总和恒等于 `Σmass2 − Σbase`，Hungarian 退化成任意分配，
得到 oracle（0.351）低于实测（0.558）的自相矛盾。
改为「在被选中的行上最大化 gain>0 的计数」后得到正确的 0.837。

---

## 脚本清单

### 语义与分离度

| 脚本 | 测什么 | 参考值 |
|---|---|---|
| `validate_semantics.py` | Cohen's d（router 的 slow/fast 划分 vs 真实运动）+ oracle/uniform 对照 | oracle 18.62 / uniform 1.00 |
| `decomp_check.py` | 场景分割质量：patch 归属熵、slot 两两 IoU、空间铺展、前后景覆盖 | 空间标准差 2.27（均匀铺满 6.5） |
| `redundancy_check.py` | 每个物体上有几个 slot + 它们两两 IoU（判断 on-object 高是否只是冗余） | 理想 1 slot = 1 物体 |
| `inversion_check.py` | 跟踪型 vs 静止型 slot 的 content diff 对比 | — |

### 跨帧绑定

| 脚本 | 测什么 | 参考值 |
|---|---|---|
| `track_check.py` | 无歧义跟踪判据：锁在物体上的 slot，物体移动后 alpha 在新位置的增益 | 0.5 随机 / 0.7 在跟踪 |
| `binding_check.py` | alpha centroid 位移 + NN 匹配为恒等置换的比例 | carryover 下恒等比例应 ≈1.0 |
| `anchor_check.py` | 内容锚定 vs 位置锚定（同 init 换图 vs 换 init 同图） | 比值 ≫1 内容锚定 |
| `savi_probe.py` | random / shared / carryover 三 regime 的 SNR 与相关性对比 | shared 噪声底 = 0 |
| `step1_probes.py` | TODO Step 1 的 P1（fresh init 归因）/ P2（矩匹配）/ P3（iters sweep） | 参照 on-object 0.510 |

### Tracking-by-Matching（Step M1）

| 脚本 | 测什么 | 参考值 |
|---|---|---|
| `m1_matching.py` | Hungarian 匹配 + identity/random 对照 | 经验地板 0.387 |
| `m1_sweep.py` | `w_iou`/`w_feat` × `topk` 一遍数据内全组合评估 | 全表跨度 0.036 |
| `m1_oracle.py` | 双射/非双射 oracle 上界 + 相邻帧分解对齐余弦 | 上界 0.837，对齐余弦 0.689 |

### 其他

| 脚本 | 测什么 |
|---|---|
| `target_compare.py` | 候选 L_slow target（content diff / centroid shift / alpha L1）对「覆盖运动」的预测力 |
| `ab_eval.py` | SNR 分解（相邻帧 diff vs 同图重复 diff）+ Cohen's d + gumbel/argmax gap |
| `viz_slots.py` | 从 checkpoint 事后生成 slot alpha overlay（token 几何自动推断） |

---

## 已建立的参考数值（synthetic_diag，K=16，DINOv2 ViT-S/14 @224 → 16×16=256 token）

```
localization (on-object 交集)
  shared-init 训练              0.510 ~ 0.521
  carryover 训练 (fresh eval)   0.189        <- 权重被 carryover 训练损害
  carryover 训练 (carryover eval) 0.217

tracking
  shared-init                   0.349
  carryover dim128              0.506
  carryover dim192 峰值          0.724 @step4000  (final 0.462, 绑定是训练瞬态)
  Hungarian matching 最佳        0.564  (地板 0.387, 上界 0.837)

SNR 分解 (slot 表征时间方差)
  独立随机 init                  0.15   (87% 是 init 噪声)
  共享 init                      ∞      (噪声底精确 0.000000)

corr(content diff, motion)
  独立随机 init                  +0.085
  共享 init                      +0.309
  carryover 训练后               −0.379  <- 反信号
```

## 已知局限

- **样本瓶颈**：交集掩膜要求 `obj_t` 与 `obj_{t+1}` 各 ≥30 像素。
  `disk_speed=12` + 默认半径 8–22 只有 ~31% clip 有效；用 `--min-radius 14`
  的诊断集可达 98%。旧结果若标注 `clips_used=46` 即受此限制。
- **motion score 的语义**：measures「slot 覆盖运动区域的程度」，不等于
  「slot 跟踪该物体」。后者须用 `track_check.py` 的增益判据。
- 多数脚本按 `batch=1` 逐 clip 前向，400 clips 约 1–2 分钟；未做批处理优化。
