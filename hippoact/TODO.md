# TODO — 服务器 Agent 执行清单

> 本文件是服务器上 Claude Code agent 的**权威任务清单**。按顺序执行，每步有明确判据。
> 规则：判据不过 → **停下来，把数据带回 cowork 会话**，不要自行改 loss / 架构 / 判据。
> 可以自行修的：环境问题、依赖、路径、显存、数据生成参数复跑。
> 最后更新：2026-07-30（TODO_RESULT Step 1 之后 — P1/P2 判决已消化）

---

## 当前状态快照

- 分支：`hippoact`，最新 = cp5g + TODO_RESULT
- **Step 1 判决（TODO_RESULT.md）**：P1 = 权重损害（fresh init 也只有 0.189）；
  P2 = 矩匹配 no-op（on-object 差值 0.000）→ carryover_norm 训练**永久取消**
- **两个关键新事实**：
  1. tracking / localization 在权重层面守恒——三个配置严格反向，同一模块
     无法同时做定位与跨帧绑定（§IV.I negative result 素材）
  2. 绑定是训练瞬态，峰值随容量前移然后被 L_slot 吃掉 → **checkpoint 不能按
     重建 loss 选，必须按 tracking 指标选**
- 手头 checkpoint：shared-init 训练版（定位 0.510，最好的定位性能）+ carryover 系列
- 诊断脚本（scratchpad）：step1_probes / track_check / redundancy_check /
  binding_check / anchor_check / target_compare / validate_semantics /
  savi_probe / decomp_check / inversion_check / ab_eval

---

## 方向变化说明（先读这个）

P1/P2 判决否掉了"init 侧修复"整条路线。新路线基于 TODO_RESULT 的守恒发现：

**不再让 Slot Attention 学跨帧绑定。编码器只做它擅长的定位（shared init，
0.510），跨帧 identity 改为事后计算——Tracking-by-Matching。**

理由：
- 守恒关系说明单模块两任务在此规模是结构冲突，SAVi-lite predictor 仍是
  "让网络学 identity"，有重蹈权重损害的风险，降级为 fallback
- Binding Transformer 是 attention over set，对 slot index 不敏感，Stage 2
  不依赖编码器自身的跨帧绑定
- Matching 是 eval-only 可验证的——又一次"训练前先量"

---

## Step M1 — Tracking-by-Matching probe（eval-only，无训练）

用 **shared-init checkpoint**（定位 0.510 那个）：

1. 相邻两帧独立编码（fresh init 各自收敛——注意**不要**共享 init，那会
   冻结分区；我们现在要的是每帧各自最好的定位）
2. 跨帧 slot 配对用 Hungarian（scipy.optimize.linear_sum_assignment）：
   ```
   cost[i, j] = -(w_iou * IoU(alpha_prev[i], alpha_cur[j])
                  + w_feat * cosine(slot_prev[i], slot_cur[j]))
   先试 w_iou=0.7, w_feat=0.3；alpha IoU 用 top-16 patch 二值化后算
   ```
3. 配对后测两件事：
   - **matched tracking**：配对 slot 的 alpha centroid 位移 vs 物体真实位移
     的方向一致性（交集掩膜，track_check.py 的判据）
   - **localization 保持**：on-object 比例应仍 ≈ 0.510（编码器没动，理论上
     必然保持，测一下当 sanity）

**数据**：重新生成大圆盘诊断集消掉 46/150 的样本瓶颈：
```bash
git pull   # 拿到 --min-radius 参数
python scripts/make_synthetic_data.py --out data/frames/synthetic_diag \
    --min-radius 14 --max-radius 26 --n-clips 400
# 直径 ≥28px > 位移 ~15-20px → 交集掩膜非空，有效样本应 >300 clips
```

### 判据

| matched tracking | localization | 结论 → 行动 |
|---|---|---|
| **≥ 0.60** | ≥ 0.45 | **Matching 路线成立** → Step M2 |
| 0.45–0.60 | ≥ 0.45 | 部分成立 → 调 w_iou/w_feat 和 IoU 二值化阈值，一轮内重测；仍不过 → 停，回 cowork |
| < 0.45 | — | Matching 不足以恢复 identity → 停，回 cowork（SAVi-lite predictor 讨论） |

**顺带记录**：配对的 assignment 稳定性（相邻 pair 之间同一物体是否连续同一
slot 链），这决定 Stage 2 的时间窗 T=4 里 identity 链能不能连起来。

---

## Step M2 — alpha displacement 作为 L_slow target（cowork 出 patch）

M1 过了以后**回 cowork 报数**，cowork 出 patch：
- matching 工具函数正式化（从你的 probe 脚本整理进 hippoact/utils/）
- `slow_signal: alpha_displacement` ——配对 slot 的 centroid 位移经分位数
  归一化后作为 router 的 BCE target
- 训练循环里 matching 在 no_grad 下做，per-batch 开销 ~ms 级

然后重训 shared-init + 新 target（**编码器训练方式完全不变**，只换 router
的监督信号），判据：

- **Cohen's d ≥ 1.0**（validate_semantics.py，oracle/uniform 对照先跑）
- slot alpha 图可见分工
- `slow_ratio` 健康（0.6-0.9 区间，argmax/gumbel gap < 0.05）

| 结果 | 行动 |
|---|---|
| d ≥ 1.0 | **synthetic phase 正式关闭** → Step P2（Phase 2） |
| d < 1.0 | 停，回 cowork 做 synthetic 终审（此时诊断链已完整，直接终审而不是继续迭代） |

---

## Step P2 — Phase 2：DMC + Distracting Control Suite（cowork 主导）

Synthetic 关闭后回 cowork，那边给：
1. TD-MPC2 fork 集成方案（encoder 替换 h_phi；checkpoint 选择规则要按
   tracking 指标不是 L_slot——TODO_RESULT 的瞬态发现）
2. DCS env wrapper
3. walker-walk 单任务跑通判据

**先不要自己搭。**

---

## 挂账清单（不阻塞主线，有空处理）

- [ ] `docs/paper_section_IV_experiments.md` §IV.E.1 "196 patch grid" → 256
      （`docs/paper_section_III_method.md` §III.C 的 "N = 196" 同样要改；
      这个数字错误当初引发过 viz.py 的 AssertionError，不是无害笔误）
- [ ] `docs/paper_section_III_method.md` §III.D.3 还是 pre-CP5 的旧公式，
      等 M2 定型后 cowork 统一重写（会包含守恒发现 + matching 设计）
- [ ] 论文 §IV.I negative result：tracking/localization 权重守恒 + 绑定瞬态，
      素材已在 TODO_RESULT.md §4-5，cowork 写作时直接引用

---

## 常备规则（每一步都适用）

1. **训练前先量数据**：新数据/新 target 先测相关性或 eval-only probe，
   信号不足不启动训练
2. **单变量原则**：一次只改一个 knob，改前记录 baseline
3. **判据前置**：跑之前写下预期数字，跑完对照
4. **度量卫生**：motion mask 用交集定义 `motion(t-1,t) ∩ motion(t,t+1)`；
   新度量先跑 oracle/uniform 正负对照
5. **checkpoint 按 tracking 指标选，不按 L_slot**（绑定是瞬态）
6. **卡住就带数据回 cowork**，不要在服务器上即兴改方法

## 汇报格式（回 cowork 时带上）

```
Step N 结果：
- 判据对照表（预期 vs 实测）
- 关键数字：matched tracking / localization / Cohen's d / assignment 稳定性
- 异常观察（如有）
- 卡在哪一条判据（如卡住）
```
