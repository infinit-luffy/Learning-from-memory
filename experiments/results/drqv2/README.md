# W1.3 — DrQ-v2 baseline（model-free 代表）

12 + 9 补种 runs，`num_train_frames=1_000_000`（= 500K agent steps，与 W1.2 逐点可比）。
fork 改动 4 文件 +29 −8，算法零改动（`experiments/dcs/drqv2_fork.patch`）。

| cell | 学会 / 总数 | per-seed |
|---|---|---|
| cheetah_run none | **3/3** | 707, 733, 714 |
| cheetah_run easy | **3/3** | 570, 621, 625 |
| walker_walk none | **4/6** | 967, 953, **23**, 881, **22**, 958 |
| walker_walk easy | **2/9** | 29, 28, 28, 21, 33, **870**, 28, **789**, 30 |

「学会」= 最后一次 eval > 200；walker-walk 的随机策略约 20–24 分。

## 读这张表必须注意的三件事

### 1. 最后一次 eval 在 475K agent step，不是 500K

DrQ-v2 的训练循环在 `global_step == 500000` 时先退出、后 eval。与 W1.2 的
500K 点比较时要么取 475K 对 475K，要么改用 final-checkpoint 重测口径。

### 2. walker_walk 的失败是**双峰**，均值是误导性统计量

失败形态一律为**从第 0 步就不动**，不是学到一半崩。easy 臂 9 个 seed 分成
`{21,28,28,28,29,30,33}` 与 `{789, 870}` 两簇 —— 学会的那两个与 TD-MPC2 的
853.7 相当。

→ **DrQ-v2 在带干扰的 walker 上不是学不会，而是大概率起不来。**
论文应报「学会率 + 学会者的分数」两个数并画双峰，不要报 mean ± sd。

同一格 TD-MPC2 是 3/3、853.7 ± 89.0 —— model-based 在干扰下**不仅更强，
更重要的是更可靠**。

### 3. 显著性做不出来，且基线本身有问题

```
clean 4/6 vs easy 1/6 (首轮 6 seed)   Fisher 单侧 p = 0.121
clean 4/6 vs easy 2/9 (补到 9 seed)    Fisher 单侧 p = 0.119
```

补种没有改变结论。（过程中曾在 easy 0/5 的中间态得到 p = 0.046，
那个数**已作废** —— 第 6 和第 8 个 seed 都学会了。）

**且 DrQ-v2 在干净 walker 上也有 2/6 失败**，而其官方结果是稳定 950+。
原因未查出。不披露这一点的话，easy/clean 的对比会被质疑基线没调好。
候选方向（均未验证）：`num_train_frames` 设为 1M 而其 walker_walk 默认 1.1M；
`nstep=1 / batch=512` 的 per-task 覆盖与截断步数的交互；或该实现的已知不稳定。

## 文件

| 文件 | 内容 |
|---|---|
| `summary.csv` | 每个 run 一行：cell / seed / 末点 step 与 reward / 是否学会 |
| `curves/<cell>_s<seed>.csv` | 学习曲线 |

分辨率差异需在论文披露：**DrQ-v2 84×84 / TD-MPC2 64×64**，各自的已发表配置，
未做统一（统一会让某一方偏离其最优设置）。
