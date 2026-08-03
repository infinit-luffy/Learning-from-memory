# Core Modification Record

## Modification: `hippoact_include_proprio` —— E2E-0 纯视觉对照开关

- Location: `experiments/dcs/hippoact_slots.py`（构造参数 + `_obs` 提前返回）、
  `experiments/dcs/dcs_env.py`（透传）、`third_party/tdmpc2/tdmpc2/config.yaml`（默认 true）、
  `experiments/scripts/run_queue.py`（`--extra` 透传 hydra override）
- Change: `HippoActSlots` 增加 `include_proprio`；false 时观测只有 2048 维 slot 特征。
  `run_queue.py` 增加 `--extra`，把任意 hydra override 追加到 tdmpc2/e2e0 命令。
- Reason: walker-walk 只用 24 维 proprio 就能解到 979（官方 state-obs 曲线），
  带 proprio 的 E2E-0 与该基线重合 —— slot 特征贡献不可测（TODO_RESULT §R4.8）。
- Problem solved: 让 E2E-0 与只看像素的 pixel 基线可比。
- Broader change avoided: 没有改 E2E-0 契约、没有改 TD-MPC2、没有新增 runner
  类型。开关默认 true，原有三个 run 的语义不变。
- Validation: 6K 步 smoke 跑通（R 52→59）；banner 实测
  `feature_dim=2048 include_proprio=False`；`run_queue --dry-run` 确认
  override 出现在命令末尾。三个 500K run 已起跑。
- Remaining risk: `retention_eval.py` 用 `^e2e0_s(\d+)$` 匹配 E2E-0 checkpoint，
  **匹配不到 `e2e0vis_s*`**，会误按 pixel（obs=rgb）去评测。跑 Q2 前必须先放宽
  该正则并透传 `hippoact_include_proprio=false`。
