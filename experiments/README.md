# experiments/ — Phase 2 (TD-MPC2 × DCS) 运行目录

W1.2 的 TD-MPC2-pixel baseline 矩阵在这里跑。设计文档见 `hippoact/PHASE2_PLAN.md`，
判据与预注册见 `hippoact/TODO_RESULT.md` §W1.2。

## 布局

```
experiments/
├── setup_env.sh              # A5000 服务器的 conda 环境安装（版本选择理由写在脚本里）
├── dcs/
│   ├── dcs_env.py            # DCS 环境构造（版本控制内的真实实现，两个 baseline 共用）
│   ├── drqv2_dcs.py          # DrQ-v2 侧接入，复用上面同一个 make_dm_env
│   ├── tdmpc2_fork.patch     # 对 third_party/tdmpc2 的改动：2 文件 +22 −1，算法零改动
│   ├── drqv2_fork.patch      # 对 third_party/drqv2 的改动：4 文件 +29 −8
│   └── __init__.py
├── scripts/
│   ├── smoke_env.py          # 集成正确性检查（跑长 run 前必过）
│   ├── egl_probe.py          # 测 MUJOCO_EGL_DEVICE_ID → 物理 GPU 的映射（见坑 4）
│   ├── train_dbg.py          # tdmpc2 train.py 的透明包装：SIGUSR1 → 栈转储
│   ├── run_queue.py          # 多 GPU 作业队列（可重入：已完成的 run 会跳过）
│   ├── sps.py                # 吞吐 / 进度 / ETA
│   ├── smoke_drqv2.py        # DrQ-v2 侧的同款集成检查
│   ├── final_eval.py         # final checkpoint × 30 episodes 重测（Table V 的读数）
│   ├── retention_eval.py     # Q2 零样本：每个 ckpt × {none, easy, hard}
│   ├── w12_report.py         # 结果表 + 与官方曲线对照 + 判据裁决
│   ├── export_results.py     # 把结果从 logs/ 导出到 results/（入库的那份）
│   ├── chain_w13.sh          # 等 retention 跑完自动接 W1.3
│   └── w1{2,3}_*.txt         # 作业清单
├── results/                  # 48 KB，**入库**：曲线 / final_eval / summary / table_v
└── logs/                     # ~500 MB 运行产物（.gitignore）
    ├── <task>/<seed>/<exp>/{eval.csv, models/final.pt}
    ├── final_eval.json
    └── console/<task>_s<seed>_<exp>.log
```

## 常用操作

```bash
PY=/usr1/home/s125mdg56_03/miniconda3/envs/hippoact/bin/python

# 看进度 / ETA
$PY experiments/scripts/sps.py

# 看结果表与判据裁决
$PY experiments/scripts/w12_report.py

# 起一批 run（已完成的会自动跳过，可反复执行）
nohup $PY experiments/scripts/run_queue.py \
    --jobs experiments/scripts/w12_jobs.txt --gpus 0,1 --slots-per-gpu 3 \
    > experiments/logs/queue.log 2>&1 &

# run 跑完后：重测 final checkpoint（Table V 的读数），再导出入库
$PY experiments/scripts/final_eval.py --episodes 30 --gpus 0,1
$PY experiments/scripts/export_results.py

# 某个 run 疑似卡住时取 Python 栈（服务器 ptrace_scope=1，py-spy 挂不上现有进程）
kill -USR1 <pid>        # 栈会打进该 run 的 console log
```

## 五个容易踩的点

1. **冷 `torch.compile` 缓存会让 run 看起来死了。** 首个 run 的 seed pretraining
   可能停 15 分钟、`I: 3,500` 后再停 13 分钟，GPU 0% 而单核 100%。缓存预热后
   同样的 run 只要 2 分钟。**不要用"inductor 缓存目录不增长"判断"没在编译"——
   dynamo 追踪阶段是纯 Python，不写缓存。**
2. **replay buffer 在 CPU 内存，18.45 GB/run**（rgb 9×64×64 × 500K）。
   并发数受主机内存而非显存限制（显存只用 ~1.3 GB/run）。6 并发 ≈ 111 GB。
3. **`obs=rgb` 是 3 帧 × 64×64，不是 84×84。** 见 `third_party/tdmpc2/tdmpc2/envs/dmcontrol.py`
   的 `Pixels(num_frames=3, size=64)`。
4. **`MUJOCO_EGL_DEVICE_ID` 的编号不等于 CUDA 编号。** 本机实测
   `0→GPU2, 1→GPU3, 2→GPU0, 3→GPU1`（整体转两位）。写成 `=<cuda id>` 会把
   MuJoCo 渲染上下文放到**别的卡**上，且不报错、不明显变慢，只有看
   `nvidia-smi` 的**完整进程表**（`C`/`G` 类型列）才能发现——只看
   `--query-gpu` 的显存/利用率汇总是看不出来的。换机器/换驱动用
   `egl_probe.py` 重测。
5. **步数单位：`cfg.steps` 是 agent step，论文与官方 CSV 是 env step（2×）。**
   DMControl `action_repeat=2`。对照官方曲线时必须查 `2 × step`，
   否则会凭空多出 1.8 倍的假差距。DrQ-v2 侧的 `num_train_frames` 与
   `eval.csv` 的 `frame` 列都是 env step，同一 csv 的 `step` 列才是 agent step。
6. **DrQ-v2 README 的 `task=walker_walk` 是 hydra 1.1 写法。** hydra 1.3 下
   必须写 `task@_global_=walker_walk`，否则启动即报错。
7. **DrQ-v2 的 `replay_buffer._worker_init_fn` 与新 numpy/Python 不兼容**
   （`np.uint32` 传给 `random.seed()`，Python 3.11 拒收），已在 fork 里
   改成 `int(...)`，播种语义不变。

## W1.3 DrQ-v2 起跑

```bash
$PY experiments/scripts/smoke_drqv2.py          # 先过集成检查
nohup $PY experiments/scripts/run_queue.py \
    --jobs experiments/scripts/w13_jobs.txt --runner drqv2 \
    --gpus 0,1 --slots-per-gpu 3 > experiments/logs/queue_w13.log 2>&1 &
```

作业清单里 task 字段写作 `<drqv2 task>__<distraction>`，队列拆成
`task@_global_=<task> distraction=<distraction>`；`steps` 一律用 agent step，
队列自动 ×2 换算成 `num_train_frames`。**DrQ-v2 的 replay buffer 落磁盘**
（~21 GB/run），不占内存。

## 环境

conda env `hippoact`：python 3.11 / torch 2.7.1+cu126 / dm_control 1.0.16 /
mujoco 3.1.2 / numpy 1.24.4 / opencv-python-headless 4.11。
**mujoco 停在 3.1.2 是有意的**：`tex_rgb → tex_data` 的改名发生在 3.2，
停在 3.1.2 则 `distracting_control` 无需打补丁。

DAVIS 2017：`/usr1/home/s125mdg56_03/datasets/davis/DAVIS/JPEGImages/480p`（90 序列，819 MB）。
用 `$DAVIS_PATH` 覆盖。**不放 /var/tmp**——多天的 run 不能依赖可能被清理的临时目录。
