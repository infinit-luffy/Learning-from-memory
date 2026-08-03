# Meta-World 视觉版 —— 一页 spec（动手前）

对应 TODO §R4.8/R3.3 拍板第 5 条：*「MW 视觉版环境调研（obs 里物体位姿必须不可见）、
Stage-1 on MW 帧采集计划、TD-MPC2 的 MW 配置确认。先出一页 spec 再动手。」*

本文全部结论来自实读源码（metaworld 2.0.0 wheel、`third_party/tdmpc2/`），
不是回忆。**没有一行代码被写、没有一个包被装。**

---

## 0. 三句话结论

1. **TD-MPC2 的 MW 路径不能直接用**：它 `assert cfg.obs == 'state'`（无像素支持），
   且用的是 goal-observable 变体 —— 39 维 obs 里**明写了物体位姿和目标位置**。
2. **两个官方变体都不满足要求**：goal-hidden 只把目标置零，物体位姿照样在。
   必须自己切片，合法 proprio 只有 **8 维**。
3. **装 metaworld 有把 DCS 管线整个搞坏的风险**：metaworld 2.0.0 要求
   `gymnasium>=1.1`，我们装的是 0.29.1。这是 numpy 那个坑的第三次同类。

---

## 1. Meta-World 观测布局（实读 `metaworld/sawyer_xyz_env.py`）

`_get_obs()` 返回 39 维 = `curr(18) ⊕ prev(18) ⊕ pos_goal(3)`，
其中 `_get_curr_obs_combined_no_goal()` 的 18 维 =
`pos_hand(3) ⊕ gripper_distance_apart(1) ⊕ obs_obj_padded(14)`，
`obs_obj_padded` = 物体1 `pos(3)+quat(4)` ⊕ 物体2 `pos(3)+quat(4)`，不足补零。

| 切片 | 内容 | 本项目协议 |
|---|---|---|
| `[0:3]`, `[18:21]` | 末端执行器 xyz | ✅ 保留 —— 手臂本体感受 |
| `[3]`, `[21]` | 夹爪开合（归一化到 0–1） | ✅ 保留 |
| `[4:18]`, `[22:36]` | **物体 pos + quat** | ❌ 必须切掉 |
| `[36:39]` | **目标位置** | ❌ 必须切掉 |

→ **合法 proprio = 8 维**（当前帧 4 + 上一帧 4）。物体信息全部必须从视觉读。

这正是拍板里 manipulation 与 locomotion 的分界：DCS 的 proprio 是全状态（作弊），
MW 的 proprio 只含手臂（正当），binding transformer + proprio 融合在这里才被真检验。

## 2. 官方变体为什么都不够

- `ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE`（TD-MPC2 用的）：目标可见，物体位姿可见。
- `ALL_V2_ENVIRONMENTS_GOAL_HIDDEN`：`_partially_observable=True` 只让
  `pos_goal = np.zeros_like(pos_goal)` —— **物体位姿仍在 `[4:18]`/`[22:36]`**。

两者都不行。要自己写切片 wrapper，取 `obs[[0,1,2,3,18,19,20,21]]`。

## 3. 切掉目标后任务会不会 ill-posed？—— 需渲染确认

切掉 `pos_goal` 只有在**目标能从图像看出来**时才是合法的部分可观测。
查了 XML，目标以 site 形式存在且不透明：

```
sawyer_push_v3.xml:27   <site name="goal" pos="0.1 0.8 0.02" size="0.02"
                              rgba="0 0.8 0 1"/>          <- alpha=1，会渲染
sawyer_drawer.xml:12    <site name="goal" pos="0. 0.74 0.05" size="0.02"
```

drawer 的目标由抽屉自身几何决定，天然视觉可读。

> **⚠️ 这一条只是 XML 证据，不是观测证据。**
> XML 里有 site ≠ `corner2` 相机拍得到（可能被遮挡、可能在画面外、
> MuJoCo 也可能因 group 设置不渲染 site）。
> **动手前必须渲染一帧肉眼看**（常备规则 4：先渲染数据再信度量）。
> 目标球看不见的任务一律不选。

## 4. 依赖风险 —— 本项目已被同类坑咬过两次

| | 我们已装 | metaworld 2.0.0 要求 |
|---|---|---|
| gymnasium | **0.29.1** | **>= 1.1** ⚠️ |
| mujoco | 3.1.2 | >= 3.0.0 ✅ |
| numpy | **1.24.4（已 pin）** | >= 1.18（无上界）⚠️ |
| python | 3.11 | >= 3.10 ✅ |

`pip install metaworld` 会**升级 gymnasium**，而 dm_control 1.0.16、TD-MPC2、
`experiments/dcs/*.py` 全部依赖当前版本。numpy 无上界也意味着可能被顺带顶到 2.x
—— 那正是 W1.3 十二个 DrQ-v2 run 全灭的原因。

**另外 TD-MPC2 的 MW 代码根本不兼容 2.0.0**：它 `from metaworld.envs import
ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE`，该符号在 2.0.0 中已不存在。
TD-MPC2 pin 的是 git 版：

```
third_party/tdmpc2/docker/environment.yaml:49
  # - git+https://github.com/Farama-Foundation/Metaworld.git@04be337a12305e393c0caf0cbf5ec7755c7c8feb
```

（在其 environment.yaml 里是**注释掉的**，即官方基础镜像也不含 MW。）

### 硬性要求

**metaworld 装进独立 venv，绝不装进跑 DCS 的 `hippoact` 环境。**
装完必须重跑 `experiments/setup_env.sh` 的 numpy/gymnasium 断言确认 DCS 侧未受影响。

## 5. 需要新写 vs 可复用

| | 状态 |
|---|---|
| MW 视觉 wrapper（render 224² + obs 切片） | ❌ 新写。TD-MPC2 无像素 MW 路径 |
| MW 帧采集脚本（Stage-1 用） | ❌ 新写。可照搬 DCS 采集的结构 |
| `SlotFeatureExtractor` | ✅ 直接复用，与域无关 |
| `HippoActSlots` 的 env 前移快路径 | ✅ 复用（§R4.6，180× 提速） |
| `run_queue.py` / `export_*.py` | ✅ 复用 |

## 6. 工作量（基于实测，不是估计）

| 步骤 | 成本 | 依据 |
|---|---|---|
| Stage-1 on MW 帧，每 seed | **8.4 h** | walker 30K 步实测 30366 s（`logs/stage1_var_s3.log`） |
| 帧采集 25K 帧 | ~0.5 h | 照 DCS 口径 |
| E2E 500K 步（快路径） | ~5.5 h/run | §R4.6 |
| 判据（TODO W4.1） | 成功率 ≥ 0.9× pixel | —— |

MW 有 100 步 episode（`Timeout(env, max_episode_steps=100)`），比 DCS 的 500 短，
单步成本应低于 walker，但**渲染分辨率 384² vs 224²** 要重新量，不要直接外推。

## 7. 建议的最小起步（三件事，零 GPU，不碰现有环境）

1. **隔离 venv 装 metaworld**，确认 `hippoact` 环境的 gymnasium/numpy 未被动。
   同时决定用 2.0.0 还是 TD-MPC2 pin 的 `04be337`。
2. **渲染 `pick-place-v2` / `drawer-open-v2` 的 `corner2` 帧**存成 PNG 肉眼看：
   目标球是否可见、物体是否清晰、384² 下 DINOv2 patch 够不够。
   —— 这一步决定 §3 的假设成不成立，**不做这步就动手是赌**。
3. **写 obs 切片 wrapper + smoke**：确认切片后 obs 是 8 维、
   物体位姿确实不可见（对同一状态改变物体位置，obs 应不变）。

三件做完再谈训练。

---

## 8. 尚未决定 / 需 cowork 拍板

1. **选哪两个任务**。TODO W4.1 写的是 `pick-place-v2` + `drawer-open-v2`，
   但要等 §7.2 的渲染结果 —— 目标球在 `corner2` 看不见的任务必须换掉。
2. **metaworld 版本**：2.0.0（新、需自己写 env 接线）vs TD-MPC2 pin 的 `04be337`
   （旧、TD-MPC2 的 `envs/metaworld.py` 能直接用，但那条路是 state-obs，
   我们本来也要重写）。倾向 2.0.0，因为反正要自己写视觉 wrapper。
3. **Stage-1 是否需要 per-domain 重训**。DCS 侧目前只有 walker-walk 上训的
   encoder；cheetah 的 E2E（拍板第 4 条）也面临同一问题 —— 见下。

## 9. 附：拍板第 4 条（cheetah 走纯视觉）有一个未言明的依赖

现有 Stage-1 checkpoint **只在 DCS walker-walk 帧上训过**
（`experiments/results/stage1/stage1_dcs.yaml` 第 1 行）。cheetah 的 E2E 要么
(a) 复用 walker 上训的 encoder —— 跨域迁移，本身是另一个变量，必须声明；
要么 (b) 在 cheetah 帧上重训 Stage-1 —— **8.4 h/seed × 3 seed**。

两条路都可以，但不能默认。**需 cowork 指定。**
