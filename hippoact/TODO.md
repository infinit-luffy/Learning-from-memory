# TODO — 服务器 Agent 执行清单

> 本文件是服务器上 Claude Code agent 的**权威任务清单**。按顺序执行，每步有明确判据。
> 规则：判据不过 → **停下来，把数据带回 cowork 会话**，不要自行改 loss / 架构 / 判据。
> 可以自行修的：环境问题、依赖、路径、显存、数据生成参数复跑。
> 最后更新：2026-07-30（CP5g 之后）

---

## 当前状态快照

- 分支：`hippoact`，HEAD = `c05a2f3` (cp5g)
- 已完成：CP1-4 ✅；CP5 系列已诊断 6 层 bug（详见 git log cp5 → cp5g）
- 最新结论：raw carryover 保 tracking (0.65) 但毁 localization (0.20)；
  slot_iters 5 和 slot_dim 192 两个 lever 均无效 → 根因是 localization 是
  fresh-init 竞争的涌现产物，carryover 拆掉了该机制
- 手头 checkpoint：carryover-trained（用于下面 P1-P3 probe）
- 现有诊断脚本（scratchpad）：target_compare.py, anchor_check.py,
  savi_probe.py, ab_eval.py, decomp_check.py, validate_semantics.py

---

## Step 1 — 三个 eval-only probe（无需训练，~2 分钟总计）

用 **carryover-trained checkpoint**，全部现有 API（`sample_init()` / `slots_init` 参数）：

### P1：权重 vs init 归因
单帧 eval，用 `slot_attn.sample_init(B)` fresh init，测 on-object 比例。

| 结果 | 含义 |
|---|---|
| on-object 恢复到 ~0.4-0.5 | 权重没坏，问题纯在 init 分布 → 继续 P2 |
| 仍 ~0.2 | carryover 训练损害了编码器 → **停，带 P1/P2/P3 数字回 cowork** |

### P2：矩匹配假设（决定性实验）
Eval 时把 prev 输出 slot 标准化后映回 init 流形再当 init：
```python
z = (prev - prev.mean(-1, keepdim=True)) / (prev.std(-1, keepdim=True) + 1e-6)
init = slot_attn.slots_mu + slot_attn.slots_logsigma.exp() * z
slots_cur = slot_attn(feats_cur, slots_init=init)
```
同时测 on-object **和** tracking consistency（用修正后的交集 motion mask：
`motion(t-1,t) ∩ motion(t,t+1)`）。

| 结果 | 行动 |
|---|---|
| on-object 恢复 ≥0.4 且 tracking ≥0.6 | → **Step 2**（训 carryover_norm） |
| on-object 恢复但 tracking 掉 <0.55 | 矩匹配破坏 identity → 停，回 cowork |
| on-object 不恢复 | init 流形比一二阶矩复杂 → 停，回 cowork（下一步是 SAVi-lite predictor，cowork 出） |

### P3：测试时迭代 sweep
Eval 时 `slot_attn.iters = 3 / 5 / 8` 对比 on-object。
仅作参考数据（区分预算 vs 吸引子），不阻塞流程，跑完记录。

---

## Step 2 — 训 carryover_norm（仅当 P2 通过）

```bash
git pull   # 拿到 cp5g（slot_init_mode: carryover_norm 已实现）
cd hippoact
python -c "
import yaml
c = yaml.safe_load(open('configs/default.yaml'))
c['train']['slot_init_mode'] = 'carryover_norm'
c['loss']['lambda_slow'] = 0.0        # content diff 在 carryover 下是反信号，保持关闭
c['encoder']['slot_iters'] = 3        # 回到 3，iters=5 已证明无效，别混变量
yaml.dump(c, open('configs/carryover_norm.yaml', 'w'))
"
python scripts/pretrain_stage1.py --config configs/carryover_norm.yaml \
    --data-dir data/frames/synthetic --wandb --run-name stage1_cp5g_carryover_norm
```

训练中每 2000 步 checkpoint 跑判据（on-object + tracking，修正版 motion mask）。

### 通过判据（两条同时满足才算过）
- tracking consistency 峰值 ≥ 0.60
- on-object > 2.0 slot 比例 ≥ 0.35

| 结果 | 行动 |
|---|---|
| 两条都过 | → **Step 3** |
| tracking 过、on-object 不过 | 停，回 cowork（该上 SAVi-lite predictor） |
| tracking 不过 | 矩匹配在训练动态下破坏 identity → 停，回 cowork |

---## Step 3 — L_slow target 换成 alpha displacement（cowork 出 patch）

**不要自己实现**。Step 2 过了之后回 cowork 报数，cowork 会出：
- `slow_signal: {content_diff, alpha_displacement}` config
- alpha centroid 位移计算 + 分位数归一化 BCE
- 对应 sanity test

原因：content diff 与 motion 在 carryover 下负相关（-0.379，CP5e 实测），
tracking slot 的特征恰恰稳定。alpha 位移才是正确的"fast"信号。

拿到 patch 后重训 + 判据：
- **Cohen's d ≥ 1.0**（用修正版 motion mask + oracle 对照校准）
- slot alpha 图有可见分工（≥2 slot 追 disk，≥3 slot 稳定覆盖背景）

| 结果 | 行动 |
|---|---|
| d ≥ 1.0 | **synthetic phase 正式关闭** → Step 4 |
| d < 1.0 | 停，回 cowork，带全套数字做 synthetic 终审 |

---

## Step 4 — Phase 2：DMC + Distracting Control Suite（cowork 主导）

Synthetic 关闭后回 cowork，那边会给：
1. TD-MPC2 fork 集成方案（我们的 encoder 替换 h_phi）
2. DCS env wrapper
3. walker-walk 单任务跑通判据

**先不要自己搭**——TD-MPC2 的接口对接有几个坑（replay buffer 格式、
proprio 拼接、eval 协议），cowork 那边有完整设计文档（docs/ 目录）。

---

## 常备规则（每一步都适用）

1. **训练前先量数据**：任何新数据/新 target，先用 target_compare.py 测相关性，
   corr < 0.5 不启动训练
2. **单变量原则**：一次只改一个 knob，改前记录 baseline
3. **判据前置**：跑之前写下预期数字，跑完对照
4. **度量卫生**：motion mask 用交集定义；新度量先跑 oracle/uniform 正负对照
5. **卡住就带数据回 cowork**，不要在服务器上即兴改方法——之前 6 层 bug
   每层都是"看起来能自己修"但根因在别处

## 汇报格式（回 cowork 时带上）

```
Step N 结果：
- 判据对照表（预期 vs 实测）
- 关键数字：on-object / tracking / Cohen's d / corr
- 异常观察（如有）
- 卡在哪一条判据（如卡住）
```
