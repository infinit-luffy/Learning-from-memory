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

## Modification: `retention_eval.py` 的 HippoAct 臂环境路由

- Location: `experiments/scripts/retention_eval.py`（`eval_task_name` + 调用点）
- Change: HippoAct 臂（obs != rgb）在 `difficulty=none` 时改走 `dcs-none-*`，
  不再走裸 `walker-walk`。pixel 基线保持裸名字不变。
- Reason: 裸名字路由到 TD-MPC2 自己的 `envs/dmcontrol.py`，那条路不看
  `hippoact_precompute` —— wrapper 被**静默丢掉**，agent 拿到裸 24 维 proprio。
- Problem solved: 链式 Q2 评测在 none 上崩（state_dict 2072 vs 24）。
  这次是编码器宽度恰好不同才炸出来，宽度相同时它会安静给出错误的数。
- Broader change avoided: 没有改 dcs_env（`difficulty=none` 本就直接调
  `dm_control.suite.load`，与裸名字是同一环境），没有改 TD-MPC2。
- Validation: 四个臂的 none/hard 路由逐个打印验证；已重新起 Q2 评测。
- Remaining risk: 无。pixel 基线的「与官方逐位一致」性质未变。
