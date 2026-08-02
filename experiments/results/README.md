# W1.2 结果数据 — TD-MPC2-pixel baseline 矩阵

12 runs = {walker-walk, cheetah-run} × {clean, distracting-easy} × 3 seeds，
各 500K agent steps（= **1M environment steps**，action repeat 2）。

完整推导、判据、异常与撤回记录见 `hippoact/TODO_RESULT.md` §W1.2。
这里只放数据本身，便于离线分析（运行产物 ~500 MB 在 `experiments/logs/`，未入库）。

## 文件

| 文件 | 内容 |
|---|---|
| `curves/<task>_s<seed>.csv` | 学习曲线，`step, episode_reward`，每 25K 一点，10 eval episodes |
| `curves/*_rerun.csv` | cheetah clean s1/s2 的重跑（R2，测末端退化复现性） |
| `final_eval.json` | 每个 run 的 final checkpoint 用 **30** eval episodes 重测（14 条，含 2 条重跑） |
| `summary.csv` | 每个 run 一行，关键 checkpoint + final_ckpt_30ep（不含重跑） |
| `table_v.md` | 渲染好的 Table V / 官方对照 / 判据裁决 |
| `retention_eval.json` / `retention.md` | **R1 Q2 零样本网格**：12 ckpt × {none, easy, hard} × 30 ep |
| `stage1/criterion_by_step.csv` | Stage-1 重训的判据随步数走势（证明加训无用） |
| `stage1/stage1_dcs.yaml` | Stage-1 重训用的配方 |
| `stage1/slots_*.png` | slot alpha 叠加图 |

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

### 3. `summary.csv` 的 `dcs-easy-*` 是「easy 训练 + easy 评测」

零样本那部分在 `retention.md` / `retention_eval.json`。本轮数据显示 easy 干扰
的代价几乎全在样本效率：50K 时 easy/clean 仅 0.31–0.39，到 500K 收敛到
0.93–0.97，且 easy 组**尚未收敛**（末段仍单调上升）。

### 4. 论文 §IV.C 的 `retention = hard/none` 在这组数据上失效

easy 上训练的策略在 **none（干净背景）上比在 easy 上更差**
（walker 556.6 vs 853.7；cheetah 118.6 vs 389.7）——去掉背景同样是分布外，
分母本身带偏移，于是 hard/none 算出 1.137 / 1.618 这种 >1 的值。
用 `hard/easy`（相对训练分布）才有意义：walker **0.741**、cheetah **0.492**。
**指标定义变更需 cowork 拍板**，两种口径的数都在 `retention.md` 里。

### 5. cheetah clean 固定 seed 也不可复现，方差极大

同一 seed 2 两次运行的平台是 **470 vs 647**（差 38%）。5 个 cheetah clean run 的
final-ckpt 值极差 377（260.9–638.3）。用 ±0 的阈值线（P2.2b 的 480）去卡一个
sd≈140 的量在统计上没有意义，见 `TODO_RESULT.md` §R2。

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
