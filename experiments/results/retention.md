# R1 — zero-shot retention grid

See `hippoact/TODO_RESULT.md` §R1: the paper's `hard/none` ratio is
misleading here because easy-trained policies score *worse* on clean
backgrounds than on their training distribution, so the denominator is
itself out-of-distribution. `hard/easy` is the meaningful ratio.

```
zero-shot evaluation of trained checkpoints (30 episodes each)
trained on                      eval:none       eval:easy       eval:hard   retention
----------------------------------------------------------------------------------------
dcs-easy-walker-walk        556.6±65.9    853.7±89.0    633.0±85.6        1.137
walker-walk                 916.0±59.6     75.8±7.4      97.1±10.9        0.106
dcs-easy-cheetah-run        118.6±13.6    389.7±84.9    191.8±47.4        1.618
cheetah-run                 443.2±27.9     25.4±4.8      28.6±6.2         0.064

retention = mean(eval on hard) / mean(eval on none), per paper §IV.C.
Rows starting with `dcs-easy-` are the protocol rows (trained on Easy);
clean-trained rows are the contrast (a larger distribution shift).
```

---

## W2.1 三臂（2026-08-04，21/21 格，30 episodes 每格）

同一网格，加上 E2E-0 的三个观测臂。原始数据在 `retention_eval.json`，
完整分析在 `TODO_RESULT.md` §R4.10.5、`e2e0/README.md`。

| ckpt | 观测 | eval:none | eval:easy | eval:hard | invariance<br>none/easy | retention<br>hard/easy |
|---|---|---:|---:|---:|---:|---:|
| `proprio_only` | 24 维本体感受，**无视觉** | 975.9 | 977.1 | 975.9 | **0.999** | **0.999** |
| `e2e0_s1` | 2048 slot ⊕ 24 proprio | 976.3 | 976.5 | 974.0 | 1.000 | 0.997 |
| `e2e0_s3` | 同上 | 974.9 | 976.5 | 975.9 | 0.998 | 0.999 |
| `e2e0_s5` | 同上 | 976.2 | 975.1 | 975.6 | 1.001 | 1.001 |
| `e2e0vis_s1` | 2048 slot，**纯视觉** | 172.5 | 247.0 | 213.3 | 0.698 | 0.864 |
| `e2e0vis_s3` | 同上 | 226.9 | 281.8 | 208.2 | 0.805 | 0.739 |
| `e2e0vis_s5` | 同上 | 289.1 | 319.8 | 239.6 | 0.904 | 0.749 |
| *pixel 基线（上表）* | *3×64×64 ×3 帧* | *556.6* | *853.7* | *633.0* | *0.652* | *0.742* |

### 读这张表的两件事

**1. `proprio_only` 的 0.999 / 0.999 说明这个指标在 walker-walk 上是平凡的。**

一个**完全不看图像**的 agent 在「背景鲁棒性」上拿满分 —— 因为背景干扰只作用于
视觉通道，本体感受对它免疫。带 proprio 的三个 `e2e0_s*` 拿到 1.000 同理。

→ **在 walker-walk 上、观测里含 proprio 时，平坦不变性不构成任何证据。**
这是 §R4.8 proprio 混淆论证的最后一块，也是为什么 Q2 协议改成全部纯视觉
（TODO §R4.8 拍板）。

**2. `e2e0vis` 的比值高于 pixel，但不能单独作为结论。**

纯视觉三个 seed：invariance 0.80±0.10（pixel 0.65）、retention 0.78±0.07
（pixel 0.74）。方向一致，但**绝对分只有 247，pixel 是 854** ——
这是在接近随机策略（walker 约 20–24 分）的量级上算比值。
一个学得差的策略天然更「不变」。如实记录，不作强主张。

`e2e0vis` 的训练判据本身是不通过的（500K 时 0.35× pixel，判据 ≥0.8×），
见 `e2e0/README.md`。
