# W2.1 E2E-0 —— 数据与 **proprio 混淆**

`dcs-easy-walker-walk`，RL seed 全部固定为 1，500K agent steps。
三个臂只差观测：

| 臂 | 观测 | exp_name |
|---|---|---|
| slots+proprio | 2048 维 slot ⊕ 24 维本体感受 | `e2e0_s{1,3,5}` |
| slots only | 2048 维 slot | `e2e0vis_s{1,3,5}` |
| proprio only | 24 维本体感受 | `proprio_only` |

`_sN` 后缀是 **Stage-1 encoder 的 seed**（1/3/5，覆盖 5-seed 富集度分布的
低-中-高段，事先声明、不按判据挑），不是 RL seed。

完整推导见 `hippoact/TODO_RESULT.md` §R4.6–§R4.8。

## 最终结果（2026-08-04，7 个 run 全部 500K 完成）

| agent step | slots+proprio (n=3) | **slots only (n=3)** | proprio only (n=1) | pixel 基线 (n=3) | 判据 |
|---:|---:|---:|---:|---:|---:|
| 50K | 559.9 | **121.3** | 953.0 | 213.8 | 171 |
| 100K | 925.2 | **122.1** | 969.7 | 391.4 | 313 |
| 250K | 976.1 | **225.1** | 979.5 | 675.5 | 540 |
| 500K | 972.7 | **305.5** | 974.3 | **878.7** | — |

1. **proprio 混淆被自建对照钉死**：`proprio_only` 974.3 vs `slots+proprio` 972.7，
   **差 1.6 分**。2048 维 slot 特征的边际贡献在噪声内。不再需要引官方数字。
2. **纯视觉 E2E-0 判据全线不通过**：500K 时 0.35×，判据 ≥0.8×，四个点位全不过。
3. **冻结 slot 表征显著劣于原始像素**（305 vs 879），是倒退 2.9 倍，不是无增益。

### 根因（详见 `TODO_RESULT.md` §R4.9）

ckpt 实读 `slot_init_mode = shared`。`stage1.py:160` 自己的注释就写了该模式
**「slots do NOT track objects across frames」**，且训练时算 slow loss 前要先
`match_slots_nn` 做最近邻匹配 —— 说明 slot 下标跨帧无对应关系。

**而 E2E-0 的 `flatten(S_fg)` 是按固定下标拼接的，对置换敏感。接口不匹配。**

`slot_feature_diagnosis.json`（300 步实测，`diag_slot_features.py` 可重跑）：

```
same_slot_is_nearest      0.490    只有一半 slot 在下一帧仍是自己的最近邻
corr(Δobs, Δproprio)      0.035    观测变化与物理状态变化几乎零相关
corr(Δ未门控, Δproprio)    0.077    去掉门控也一样 → 问题在 slot 不在路由
每步跳变维度              465/2048
```

TD-MPC2 全部机制建立在「能从 (z_t,a_t) 预测 z_{t+1}」上。相关性 0.035 意味着
world model 的学习目标本身不可学。**305 分是可解释的。**

**重要推论**：`flatten` 是 E2E-0 独有的读出方式。E2E-1 的 binding transformer
用注意力处理 slot 集合，**对置换等变**，原理上免疫本失败模式。
**不能从 E2E-0 失败推出方法失败。**

---

## 关于 `e2e0_s*` 那三条曲线

**不能读作 HippoAct 赢。**

`proprio_reference.csv` 是 TD-MPC2 官方发布的 state-obs walker-walk 曲线
（`third_party/tdmpc2/results/tdmpc2/walker-walk.csv`，3 seed，env step 已
除以 action_repeat=2 换算成 agent step）：

```
agent step   50K    100K   200K   250K   500K
纯 proprio   961.9  973.4  976.4  978.7  979.7
```

**walker-walk 只用 24 维本体感受就能解到 979。**
而 `e2e0_s*` 在 250K 是 976 —— 与纯 proprio 基线重合，50K 处还更差
（559.9 vs 961.9）。→ **2048 维 slot 特征的贡献不可测**，能观察到的只有
它拖慢了早期学习。

同表里的 pixel 基线**只看像素**。所以那张对比测的是「proprio vs 像素」，
不是「slot 表征 vs 像素表征」。判据 171/313/458/540 是照 pixel 基线定的，
对一个拿到 proprio 的 agent 没有意义 —— **`e2e0_s*` 的「达标」作废**。

更根本的一点：**背景干扰只作用于视觉通道，proprio 对它免疫。**
只要观测里有 proprio，Q2 想测的「背景鲁棒性」就可以靠无视视觉白拿。

这不是实现 bug。E2E-0 契约本身就是 `z = MLP(flatten(S_fg) ⊕ q)`，
`hippoact_include_proprio` 默认 true 忠实于该契约；洞在契约与任务的组合上。

`e2e0_s*` 三个 run **没有停**，留作混淆存在的证据。
`e2e0vis_s*` 是唯一与 pixel 基线可比的 E2E-0 配置。

## 待定：walker-walk 可能不适合当 Q2 主任务

如果 `proprio_only` 在 easy 上确实到 ~975（预期如此，proprio 对背景免疫），
则这个任务测不到 Q2 想测的东西。cheetah-run 的官方 state-obs 也很高，
同样有嫌疑。**换任务需 cowork 拍板**，本仓库不擅自改主任务。

## 文件

| 文件 | 内容 |
|---|---|
| `curves/<exp_name>.csv` | 学习曲线，`step, episode_reward`，每 25K 一点，10 eval episodes |
| `proprio_reference.csv` | 官方 state-obs walker-walk，**已换算成 agent step** |
| `comparison.md` | 三臂 + pixel 基线 + 判据 + 官方 proprio 的对照表 |
| `progress.json` | 每个 run 导出时跑到第几步（训练中导出的凭据） |

重新生成：`python experiments/scripts/export_e2e0.py`（训练途中可随时重跑）。

## 读数注意

1. **步数单位**：这些 CSV 的 `step` 是 agent step；官方 CSV 是 env step（2×）。
   `proprio_reference.csv` 已经换算过，其余官方数据没有。
2. **`e2e0vis` / `proprio_only` 仍在训练**（起跑于 2026-08-04 01:55），
   `progress.json` 记录导出时各自到了第几步。曲线未跑完时表里显示 `—`。
3. **slot init seed 是编码器身份的一部分**：seed 0 vs 1 让特征差 max|Δ|=16.2
   （§R4.6.4）。全部 run 用 `hippoact_slot_init_seed=0`，记在每个 run 的
   hydra config 和 console banner 里。

---

## 更正（R4.10）：上面「置换是根因」的说法**部分作废**

R4.9 拍板要求起跑前先量数据。量完之后，**两条前置测量都不通过，
而且我原来的根因链条被自己的新数据推翻**。没有起跑任何 run。

### 1. 均值池化（E2E-0p）不通过预注册门槛

`corr(Δreadout, Δproprio)`，预注册目标 >0.3：

| readout | shared | carryover_norm |
|---|---:|---:|
| `flatten_fast`（现状） | 0.068 | 0.077 |
| **`mean_over_fast`（E2E-0p）** | **0.017** | 0.020 |
| `mean_all_slots` | 0.045 | 0.028 |

**置换不变的读出反而更差。** 若置换是主因，去掉置换敏感性应当改善。没有。

### 2. 状态确实在表征里（`linear_probe.json`）

1500 帧、随机切分、λ 扫描，held-out R²：

| feature | proprio_all(24) | **proprio_pos(15)** | proprio_vel(9) |
|---|---:|---:|---:|
| `flatten_fast`（E2E-0） | 0.001 | **0.716** | −0.005 |
| `mean_over_fast` | −0.001 | 0.634 | −0.002 |
| DINOv2 patch mean（对照） | −0.000 | 0.648 | −0.003 |
| `flatten_fast` × 3 帧 | 0.007 | 0.722 | 0.001 |

**单帧 slot 特征把位姿编码到 R²=0.716，比 DINOv2 原始特征还好。表征不是垃圾。**

`corr = 0.035` 仍然是真的，但它比的是变化量模长，而 24 维里 9 维是速度、
单帧本来看不到 —— **是指标钝，不是表征坏**。

> 探针本身返工过一次：最初按**时间**切分 + 固定 λ，所有特征**包括 DINOv2 对照**
> R² 全负。原因是 3 个 episode 各换一段 DAVIS 背景，时序切分测的是跨背景迁移。
> **负 R² 出现在对照组上时，先查工具。**

### 3. 剩下的、有证据的缺陷：单帧不是 Markov 状态

`dmcontrol.py:67` — pixel 基线是 `Pixels(env, cfg, num_frames=3)`，**堆 3 帧**。
state obs 直接含 9 维速度。而 `HippoActSlots` 只发单帧：

| 臂 | 位置 | 速度 |
|---|---|---|
| state obs | ✅ | ✅ |
| pixel 基线 | ✅ | ✅（3 帧） |
| **E2E-0 纯视觉** | ✅ | ❌ |

walker-walk 是运动控制，单帧在原理上不是 Markov 状态，**且这一点上与
pixel 基线的对比不公平**。但上表也显示堆 3 帧后速度的线性可解码性仍 ~0.00，
所以「堆帧就能修好」**没有实测支持**——只能说单帧确定不足。

## Q2 零样本网格（21/21，30 ep/格）—— §R4.8 论证的最后一块

| ckpt | none | easy | hard | invariance | retention |
|---|---:|---:|---:|---:|---:|
| e2e0_s1（含 proprio） | 976.3 | 976.5 | 974.0 | **1.000** | 0.997 |
| e2e0_s3 | 974.9 | 976.5 | 975.9 | 0.998 | 0.999 |
| e2e0_s5 | 976.2 | 975.1 | 975.6 | 1.001 | 1.001 |
| **proprio_only** | 975.9 | 977.1 | 975.9 | **0.999** | **0.999** |
| e2e0vis_s1（纯视觉） | 172.5 | 247.0 | 213.3 | 0.698 | 0.864 |
| e2e0vis_s3 | 226.9 | 281.8 | 208.2 | 0.805 | 0.739 |
| e2e0vis_s5 | 289.1 | 319.8 | 239.6 | 0.904 | 0.749 |
| *pixel 基线（R1）* | | | | *0.65* | *0.74* |

**一个完全不看图像的 agent（`proprio_only`）在背景鲁棒性上拿 0.999 / 0.999。**
→ walker-walk 上的平坦不变性是平凡的，不构成任何证据。

纯视觉三个 seed 方向上高于 pixel（0.80 vs 0.65、0.78 vs 0.74），
但绝对分只有 247（pixel easy ~854），**是在接近随机策略（~24）的量级上比值**。
如实记录，不作强主张。
