# W1.2 结果数据 — TD-MPC2-pixel baseline 矩阵

12 runs = {walker-walk, cheetah-run} × {clean, distracting-easy} × 3 seeds，
各 500K agent steps（= **1M environment steps**，action repeat 2）。

完整推导、判据、异常与撤回记录见 `hippoact/TODO_RESULT.md` §W1.2。
这里只放数据本身，便于离线分析（运行产物 ~500 MB 在 `experiments/logs/`，未入库）。

## 文件

| 文件 | 内容 |
|---|---|
| `curves/<task>_s<seed>.csv` | 学习曲线，`step, episode_reward`，每 25K 一点，10 eval episodes |
| `final_eval.json` | 每个 run 的 final checkpoint 用 **30** eval episodes 重测的结果 |
| `summary.csv` | 每个 run 一行，关键 checkpoint + final_ckpt_30ep |
| `table_v.md` | 渲染好的 Table V / 官方对照 / 判据裁决 |

重新生成：`python experiments/scripts/export_results.py`

## 读数时必须注意的三件事

### 1. 步数单位：`step` 是 agent step，官方数字是 env step（2×）

论文 Table 6：DMControl `episode length 1000` / `action repeat 2` /
`effective length 500`。代码里 `cfg.steps` 与这些 CSV 的 `step` 都是 **agent step**。
拿它直接对 `third_party/tdmpc2/results/tdmpc2-pixels/*.csv`（env step）会
**凭空多出 1.8 倍的假差距**。对照时用 `2 × step`。

### 2. Table V 用 `final_ckpt_30ep`，不用 `step_500000`

12 个 run 里有 4 个（walker s1/s2、cheetah s1/s2）在最后一次 eval 出现骤降，
z 分数 −3.6 到 −29.5。独立进程加载 `final.pt` 重测复现了它，所以**末端权重确实略差**，
不是测量假象；但单次 10-episode eval 又在上面叠了向下的噪声。
`final_ckpt_30ep` 用 3 倍样本量评的是实际交付的策略，两组口径一致。

**这 4 个 run 为何特殊尚无解释**（详见 §W1.2.11，已排除 eval 假象、
buffer 写满、波次、clean/easy 之分）。

### 3. `dcs-easy-*` 是「easy 上训练 + easy 上评测」

**不是**论文 Q2 的零样本 retention（easy 训练 → hard 评测），后者是 Week 3-4 的内容。
本轮数据显示 easy 干扰的代价几乎全在样本效率：50K 时 easy/clean 仅 0.31–0.39，
到 500K 收敛到 0.93–0.97，且 easy 组**尚未收敛**（末段仍单调上升）。

## 判据结果

```
P2.1  walker-walk @100K  869.1 vs >= 700  -> PASS
P2.2  walker-walk @500K  915.3 vs >= 850  -> PASS
P2.2b cheetah-run @500K  442.0 vs >= 480  -> FAIL  (差 38)
```

P2.2b 不通过如实保留。cheetah 在 100K/250K/375K 三点对官方是 1.03/1.08/0.98，
把读数点挪到 375K 即可"通过"，但那是挑点，未做。

## 复现

TD-MPC2 fork：`nicklashansen/tdmpc2` @ `e9f59321933cbc8e11a002b842adc7d4ffae8ff1`，
改动仅 2 文件 +22 −1 行（环境注册），见 `experiments/dcs/tdmpc2_fork.patch`，
**算法零改动**。环境安装见 `experiments/setup_env.sh`，其余见 `experiments/README.md`。
