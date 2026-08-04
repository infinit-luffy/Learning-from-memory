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

## Modification: E2E-1 —— 可训练置换等变 binding readout + 3 帧堆叠

- Location: `hippoact/encoders/binding.py`（新增 `VisionBindingEncoder`）、
  `hippoact/adapters/slot_features.py`（新增 `slots_and_mask`）、
  `experiments/dcs/hippoact_slots.py`（帧堆叠 + 发 mask）、
  `experiments/dcs/dcs_env.py`（透传）、
  `third_party/tdmpc2/tdmpc2/common/layers.py`（`enc()` 里一处分支）、
  `config.yaml`、`run_queue.py`（e2e1 runner）
- Change: env 侧发 3 帧堆叠的 slot 集合 + fast/slow mask（6192 维）；
  TD-MPC2 的 state encoder 换成 4 层置换等变 binding transformer
  → masked mean pool → SimNorm。
- Reason: TODO §R4.10 拍板。E2E-0 的冻结 flatten readout 只到 0.35× pixel；
  瓶颈不在特征信息量（位姿探针 R²=0.716 > DINOv2 的 0.648），
  在于把它变成可预测的动力学 + 单帧不是 Markov 状态（§R4.10.4）。
- Problem solved: 同时补上可训练读出与 Markov 性。
- Broader change avoided: **只在 `layers.enc()` 加了一处分支**。
  `obs=state` 不变 → `encode()` / `next()` / MPPI / buffer 全部零改动，
  与 E2E-0 快路径同一个思路。DINOv2 仍每 env step 只跑一次。
- Validation: `smoke_e2e1.py` 7/7 通过 —— slot 置换 z 不变（max|Δ|=2.98e-07）、
  帧序置换 z 改变（1.44e-02）、obs 维度对齐 6192、reset 填窗、
  step 滑窗、mask 二值非平凡、吞吐 82 SPS。
  8000 步真实训练跑通；稳态 24.7 SPS → ~5.6 h/500K。
- Remaining risk: 若 E2E-1 仍远低于 pixel，DCS 上 claim 4 失败
  （拍板已预写 fallback：论文收缩为 claim 1-3 + §IV.I 双 negative，
  claim 4 转移到 Meta-World）。

## Modification: binding 的 NaN 守卫必须无分支

- Location: `hippoact/encoders/binding.py` `VisionBindingEncoder.forward`
- Change: `if all_masked.any(): key_padding[all_masked, 0] = False`
  → `key_padding & ~key_padding.all(dim=1, keepdim=True)`
- Reason: 前者是数据依赖控制流 + 原地索引赋值，在 inductor 的 cudagraph
  backward 里 **segfault**（torch 2.x，`cudagraph_trees.py:_backward_impl`，
  ~2500 步后必现）。
- Problem solved: 保住全掩码行的 NaN 守卫，同时可被 cudagraph 捕获。
- Broader change avoided: **没有退回 `compile=false`** —— 那会掩盖真因
  并损失吞吐。也没有删掉守卫。
- Validation: 同一命令原先 8000 步内必 segfault，改后跑完 8000 步
  （`Training completed successfully`），smoke 7/7 仍通过。
- Remaining risk: 无。语义上全掩码行改为「全部可见」而非「只放行 token 0」，
  在退化输入上更合理。
