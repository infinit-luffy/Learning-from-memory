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

## 这批数据要说的一件事

**`e2e0_s*`（带 proprio）那三条曲线不能读作 HippoAct 赢。**

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
