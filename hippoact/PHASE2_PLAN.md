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
# 注意: 不要 pin dm_control==1.0.14 —— 见下方坑 1
pip install dm_control gymnasium "numpy<2.0"
pip install distracting-control
pip install opencv-python-headless          # distracting_control 依赖 cv2 但未声明
python tools/dcs/patch_distracting_control.py   # 修 tex_rgb -> tex_data, 见坑 3

# DAVIS 2017 背景视频 (~795MB, 解压 819MB, 90 个序列)
wget https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip
unzip DAVIS-2017-trainval-480p.zip -d /var/tmp/hippoact_dcs/davis/
# background_dataset_path 要给到含视频序列的那一层:
#   /var/tmp/hippoact_dcs/davis/DAVIS/JPEGImages/480p
```

**A5000 服务器（Phase-2 RL 用）的实际安装脚本见 `experiments/setup_env.sh`，
版本与上面的 5080 配方不同，见坑 3 与坑 7。**

坑预告（1-3 为 W1.1 实测新增，7-8 为 W1.2 实测新增；EGL / 绝对路径 / numpy 均已证实）：

1. **`dm_control==1.0.14` 与任何 mujoco 版本都配不上。** 它需要
   `MjModel.bvh_geomid`；实测 mujoco 3.1.6 / 3.0.1 / 3.0.0 均无该字段。
   解法：升级 dm_control 到最新版，保留 mujoco 3.11.0。
2. **`distracting_control` 依赖 `cv2` 但未在依赖里声明。** 装
   `opencv-python-headless`，不要装 `opencv-python`（无显示器机器上会拉 GUI 依赖）。
3. **`distracting_control` 用 `model.tex_rgb` 写天空盒，该字段在新版 mujoco
   改名为 `tex_data`。** 实测天空纹理 `nchannel=3, adr=0`，布局与旧版一致，
   故为纯改名。用 `tools/dcs/patch_distracting_control.py`（幂等）。
   **改名发生在 mujoco 3.2**：停在 **mujoco 3.1.2 + dm_control 1.0.16**
   （tdmpc2 官方 docker pin）则该补丁不需要 —— W1.2 的 A5000 环境即如此，
   easy 背景实测正常渲染。
4. EGL 渲染：`export MUJOCO_GL=egl`（服务器无显示器必需）—— 已证实必需。
5. DAVIS 路径要绝对路径，且要给到 `DAVIS/JPEGImages/480p` 这一层。
6. `numpy<2.0`（distracting_control 间接依赖老 gym）；实测不影响
   torch 2.11+cu128 的 CUDA 可用性。
7. **装 `distracting-control` 会把 numpy 顶到 2.x**（它依赖老 `gym`）。
   必须在它之后再 `pip install numpy==1.24.4` 压回去。
8. **`opencv-python-headless` 5.0.x 强制 numpy≥2**，与坑 7 互斥。
   固定 `opencv-python-headless<4.12`（实测 4.11.0 与 numpy 1.24.4 共存）。

**分割真值**：`env.physics.render(..., segmentation=True)` 返回 `(H,W,2)` 的
`(geom_id, type)`。walker-walk 中 `geom 0=floor`、`1-7=walker 部件`、`-1=天空`，
故 walker 掩膜 = `geom_id >= 1`，**精确且在 easy 上同样有效**（天空盒被 DAVIS
替换后分割仍返回 −1）。优先用它，不要用帧差 + 闭运算的 proxy —— 后者会把
地板倒影一并圈入，且在 easy 上完全失效。

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

已实现：`experiments/dcs/dcs_env.py`（版本控制内）+ `third_party/tdmpc2/tdmpc2/envs/dcs.py`
（17 行 shim）。任务名 `dcs-<difficulty>-<domain>-<task>`。

```python
# - distracting_control suite, difficulty in {none, easy, medium, hard}
#   difficulty=none 绕开 distracting_control 直接走 dm_control -> 与官方逐位相同
# - distraction_types=("background",), dynamic=True   # 仅背景, 与 W1.1 一致
# - action_repeat=2  # 硬编码在 tdmpc2 envs/dmcontrol.py 的 DMControlWrapper.step
# - baseline obs: 3 帧 stack x 64x64  # 复用 tdmpc2 自己的 Pixels wrapper
```

**分辨率决定（已按实测更正）**：TD-MPC2-pixel 官方 pixel 配置是 **3×64×64**，
不是先前写的 84×84（见 `envs/dmcontrol.py` 的 `Pixels(num_frames=3, size=64)`）。
baseline 保持官方 64×64（保持它的最优配置才叫公平）；HippoAct 用 224×224
（DINOv2 要求）。这是 encoder 自带的输入规格差异，论文里如实披露，
并在附录补一个 TD-MPC2-pixel@224 的对照（预期更慢更差，证明我们没有靠分辨率赢）。

**单位约定（关键，曾踩坑）**：`cfg.steps` 与日志的 `step` 是 **agent step**；
论文与仓库自带 `results/*.csv` 报的是 **env step = 2 × agent step**
（论文 Table 6：DMControl episode length 1000 / action repeat 2 / effective length 500）。
故 `steps=500000` 应在论文里写作 **1M environment steps**。
**拿 agent step 直接对官方 CSV 会凭空多出 1.8 倍的假差距**——推导与撤回见
`TODO_RESULT.md` §W1.2.0 / §W1.2.7。DrQ-v2 系论文同样报 env step。

## 5. 判据与里程碑（对照 TODO.md P2.1-P2.4）

判据按 **agent step** 表述，阈值取对应 env step 处官方均值的 0.9×
（该值落在官方最差 seed 附近）。推导见 `TODO_RESULT.md` §W1.2.0。

| 里程碑 | 判据（agent step） | 官方对照（env step） | 失败时 |
|---|---|---|---|
| P2.1 管线通 | walker-walk clean 100K ≥ **700** | @200K env = 836.1（784/834/890） | 查渲染/action_repeat/reward scale |
| P2.2 baseline 对齐 | walker-walk clean 500K ≥ **850** | @1M env = 939.6（929/942/949） | 停，回 cowork 核对超参 |
| P2.2b | cheetah-run clean 500K ≥ **480** | @1M env = 537.3（453/570/590） | 同上 |

原判据（100K ≥ 500 / 500K 650–750）在两种单位约定下都对不上官方数字，来源不明。
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
