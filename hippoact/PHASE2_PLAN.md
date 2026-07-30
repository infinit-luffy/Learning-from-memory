# PHASE2_PLAN — TD-MPC2 × DCS 集成方案

> cowork 出的设计文档。服务器 agent 按此执行 P2.1-P2.4（步骤级判据见 TODO.md）。
> 原则：**TD-MPC2 本体不改一行算法**，只换 encoder。这是论文公平对比的根基。

---

## 1. 目录布局

```
Learning-from-memory/
├── hippoact/                    # 现有包（Stage-1 已定型）
│   ├── hippoact/
│   ├── configs/
│   └── tools/diagnostics/
├── third_party/
│   └── tdmpc2/                  # fork of nicklashansen/tdmpc2 @ pinned commit
└── experiments/
    ├── dcs/                     # DCS wrapper + 配置
    └── logs/
```

Fork 方式：`git clone` 后记录 commit hash 到 `third_party/tdmpc2/PINNED`，
不用 submodule（服务器网络环境下 submodule 常出问题）。

## 2. 环境安装

```bash
pip install dm_control==1.0.14 gymnasium
# distracting_control: 用 Google research 官方或 pip 安装的社区包
pip install distracting-control  # 若无则 vendored 到 experiments/dcs/
# DAVIS 2017 背景视频 (~800MB)
wget https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip
unzip -d experiments/dcs/davis/
```

坑预告：
- dm_control 需要 EGL 渲染：`export MUJOCO_GL=egl`（服务器无显示器必需）
- distracting_control 的 background 视频路径要绝对路径
- torch 2.11 + dm_control 1.0.14 的 numpy 版本冲突：pin numpy<2.0 如果报错

## 3. Encoder 接入设计

### 3.1 TD-MPC2 侧的接口点

TD-MPC2 的 `common/world_model.py` 里 `WorldModel._encoder` 是一个
`h(obs) → z` 的模块字典（按 obs 模态分）。**唯一改动点**：

```python
# third_party/tdmpc2/tdmpc2/common/world_model.py
# 原: self._encoder = layers.enc(cfg)
# 改: 由 cfg.encoder_type 分发
if cfg.encoder_type == "hippoact":
    from hippoact_adapter import HippoActAdapter
    self._encoder = HippoActAdapter(cfg)
else:
    self._encoder = layers.enc(cfg)      # 原路径，pixel baseline 不受影响
```

### 3.2 Adapter（新文件，放 hippoact 包里）

```python
# hippoact/hippoact/adapters/tdmpc2_adapter.py
class HippoActAdapter(nn.Module):
    """obs dict {'rgb': (B,T,3,H,W), 'state': (B,T,dq)} → z (B, latent_dim)

    - 加载 Stage-1 checkpoint（slot_attn + decoder + router 冻结或 lr*0.1）
    - forward: encode_frame 每帧 → fast slots → binding → concat → MLP → z
    - DCS 没有 proprio? 有 — dm_control 的 state 观测（qpos/qvel）作 q_t
    """
```

关键决定（已按 CP6 定型更新）：
- **slot_init_mode = shared**（pair 内；推理时单帧无所谓）
- **router 用 argmax 不用 Gumbel**（部署一致性；CP6 的 gumbel/argmax gap
  在 connectivity 信号下待测，先保守用 argmax）
- z 的构成：`concat[flatten(S_fg), q_t, c_t] → MLP → 256`（论文 §III.F 不变）
- Binding Transformer 的 T=4 时间窗：TD-MPC2 默认 obs 是单帧 + frame-stack；
  改用 wrapper 提供 4 帧历史（DCS wrapper 里做，TD-MPC2 本体不动）

### 3.3 Stage-1 → Stage-2 的 checkpoint 流

```
1. DCS random policy 采 50K 帧（clean+easy 混合, 含 state 向量, 存 clip 结构）
2. Stage-1 训练（定型配方, slow_signal=alpha_connectivity）→ ckpt
3. Stage-2: HippoActAdapter 加载 ckpt; slot_attn/decoder lr = 3e-5,
   binding + z-MLP + TD-MPC2 head lr = 3e-4
```

## 4. DCS wrapper 要点

```python
# experiments/dcs/make_env.py
# - distracting_control suite, difficulty in {none, easy, hard}
# - action_repeat=2 (TD-MPC2 官方 walker 配置)
# - obs: {'rgb': 224x224 render, 'state': qpos+qvel}  ← 注意 224 不是官方 84
# - 4-frame history buffer for binding window
```

**分辨率决定**：TD-MPC2-pixel baseline 用官方 84×84（保持它的最优配置，
公平）；HippoAct 用 224×224（DINOv2 要求）。这是 encoder 自带的输入规格差异，
论文里如实披露，并在附录补一个 TD-MPC2-pixel@224 的对照（预期更慢更差，
证明我们没有靠分辨率赢）。

## 5. 判据与里程碑（对照 TODO.md P2.1-P2.4）

| 里程碑 | 判据 | 失败时 |
|---|---|---|
| P2.1 管线通 | pixel walker-walk 100K return ≥ 500 | 查渲染/action_repeat/reward scale |
| P2.2 baseline 对齐 | clean 500K return 650-750 | 停，回 cowork 核对超参 |
| P2.3 Stage-1 on DCS | walker 身体有专属 slot + router 判视频背景为 slow | 停，带 slot 图回 cowork |
| P2.4 端到端 | HippoAct ≥ 0.9× pixel (easy)；retention(hard/none) 显著更高 | 部分失败可接受，带全数字回 cowork 定叙事 |

## 6. 风险与预案

- **DINOv2 每步 forward 的开销**（224² vs 84²）：~1ms/帧 on 5080，
  500K steps 增加 ~10min，可忽略。但 planning 中 TD-MPC2 只在 latent 空间
  rollout，encoder 每步只跑一次——确认 adapter 没有被 MPPI 重复调用
- **connectivity 信号在 DCS 的未验证性**：合成数据的"紧凑物体+弥散背景"
  假设在 walker（关节体，多个部件）上如何表现未知。P2.3 的前置检查
  就是为此设的；如果 walker 部件被判 slow，回 cowork——可能需要
  connectivity 阈值或 top-k 比例调整（这是超参不是架构）
- **DCS hard 的视频背景紧凑区域**（比如视频里的人/车）可能被误判 fast：
  这正是论文要测的 failure mode，如实报告，不预先修
