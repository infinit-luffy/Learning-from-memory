# HippoAct — 测试策略

> 研究代码 vs. 生产代码的测试哲学不一样：研究代码变得快、正确性靠"能不能复现结果"来验证。**过度单元测试是研究项目的常见坑**。这份策略按 6 个 level 展开，你按需选，不用全做。

---

## 测试金字塔（从下到上，密度递减）

```
                    ▲ 少
                   ┌──┐
                   │L6│ 真机安全测试     — 上真机前必做
                   ├──┤
                   │L5│ 行为验证        — 方法真的成立吗（1/周）
                   ├──┤
                   │L4│ 集成测试        — 全 pipeline（1/天）
                   ├──┤
                   │L3│ Sanity 测试     — 关键，能过滤 80% 的 bug（每次改架构）
                   ├──┤
                   │L2│ 模块单元测试     — 只测硬约束（shape/梯度）
                   └──┤
                   │L1│ Smoke 测试      — 每次 commit（<10 秒）
                   ▼ 多
```

**研究代码的黄金比例**：L1 + L3 + L4 是主力；L2 只测硬约束；L5 每周跑一次；L6 上真机前一次性搞完。**不要花时间写 L2 里的边界情况测试**——那是生产代码的事。

---

## L1 · Smoke 测试（每次 commit 前 10 秒）

**目标**：至少让 import 和 forward 不炸。

```python
# tests/test_smoke.py
import torch
import pytest
from hippoact import HippoActEncoder, TDMPC2Wrapper

@pytest.fixture(scope="module")
def encoder():
    return HippoActEncoder(num_slots=16, slot_dim=128, proprio_dim=32).cuda()

def test_import():
    """能 import 就行"""
    from hippoact import HippoActEncoder, SlotAttention, BindingTransformer

def test_encoder_forward_no_nan(encoder):
    imgs = torch.randn(2, 4, 3, 224, 224).cuda()
    proprio = torch.randn(2, 4, 32).cuda()
    with torch.no_grad():
        out = encoder(imgs, proprio)
    assert not torch.isnan(out.c).any()
    assert not torch.isinf(out.c).any()
    assert out.c.shape == (2, 128)

def test_encoder_backward(encoder):
    imgs = torch.randn(2, 4, 3, 224, 224).cuda()
    proprio = torch.randn(2, 4, 32).cuda()
    out = encoder(imgs, proprio)
    loss = out.c.sum() + out.slot_recon.sum()
    loss.backward()
    # slot_attn 应该有梯度；dino 不应该
    assert encoder.slot_attn.to_q.weight.grad is not None
    assert encoder.dino.backbone.patch_embed.proj.weight.grad is None
```

**用法**：`pytest tests/test_smoke.py -x`，加进 pre-commit hook。**只有这一层可以强制**；其他层根据研究进度灵活跑。

---

## L2 · 模块单元测试（只测硬约束）

**原则**：只测三件事——**shape、frozen 状态、梯度流**。别测数值精度，研究代码没这个必要。

```python
# tests/test_modules.py

def test_dinov2_frozen():
    """DINOv2 参数必须冻结"""
    enc = DinoV2Encoder()
    assert all(not p.requires_grad for p in enc.backbone.parameters())

def test_slot_attention_shape():
    """输入 (B, N, d_v)，输出 (B, K, d_s)"""
    sa = SlotAttention(num_slots=16, slot_dim=128, input_dim=384)
    x = torch.randn(4, 196, 384)
    out = sa(x)
    assert out.shape == (4, 16, 128)

def test_router_gumbel_hard():
    """router 输出必须是 one-hot"""
    router = SlotRouter(slot_dim=128)
    slots = torch.randn(4, 16, 128)
    g, _ = router(slots, hard=True)
    # 每个 slot 的 g 只有一个 1
    assert torch.allclose(g.sum(-1), torch.ones(4, 16))
    assert ((g == 0) | (g == 1)).all()

def test_binding_mask_effective():
    """slow slot 被 mask 后不应影响输出"""
    bt = BindingTransformer(slot_dim=128, proprio_dim=32)
    fast_seq = torch.randn(2, 4, 16, 128)
    proprio = torch.randn(2, 4, 32)
    # 场景 A：正常 mask
    mask_A = torch.ones(2, 4, 16); mask_A[:, :, 8:] = 0
    c_A = bt(fast_seq, proprio, mask_A)
    # 场景 B：mask 位置换任意值
    fast_seq_B = fast_seq.clone()
    fast_seq_B[:, :, 8:] = 999.0
    c_B = bt(fast_seq_B, proprio, mask_A)
    # 输出应完全一致
    assert torch.allclose(c_A, c_B, atol=1e-5), "mask 没起作用！"
```

**最后一个测试是杀手锏**——它检查 P2 里最容易出 bug 的 mask 逻辑。如果这个失败，slot-swap augmentation (P3) 就没意义。

---

## L3 · Sanity 测试 ⭐ **最重要**

**这层能过滤 80% 的架构 bug**。每次改架构都跑一次，2 小时内出结果。

### S1 · Slot Attention 单张图 overfit

**假设**：给一张图片跑 500 步 Stage-1 loss，L_slot 应该降到 <0.01。

```python
def test_overfit_single_image():
    enc = HippoActEncoder().cuda()
    img = load_one_image("test_scene.png").cuda()   # (1, 3, 224, 224)
    
    optimizer = torch.optim.AdamW(
        list(enc.slot_attn.parameters()) + list(enc.slot_decoder.parameters()),
        lr=3e-4
    )
    
    losses = []
    for step in range(500):
        with torch.no_grad():
            P = enc.dino(img)
        S = enc.slot_attn(P)
        P_hat, alpha = enc.slot_decoder(S)
        L = F.mse_loss(P_hat, P.detach())
        optimizer.zero_grad(); L.backward(); optimizer.step()
        losses.append(L.item())
    
    assert losses[-1] < 0.01, f"Slot attention 无法 overfit！最终 loss = {losses[-1]}"
    assert losses[0] > 10 * losses[-1], "loss 没有明显下降"
```

**如果失败**：说明 Slot Attention 架构或 loss 有 bug。**立刻停下来查**，别继续跑真实数据。

### S2 · Router 二值化验证

**假设**：给一段"背景不变、物体运动"的合成视频（10 帧），跑 Stage-1 100 步，router 应把背景 slot 稳定路到 slow。

```python
def test_router_learns_slow_fast():
    # 合成数据：黑色背景静止 + 白色圆点移动
    frames = synthetic_moving_ball_video(T=10)   # (10, 3, 224, 224)
    
    enc = HippoActEncoder().cuda()
    train_stage1_briefly(enc, frames, steps=100)
    
    # 检查慢/快路由稳定性
    with torch.no_grad():
        slot_slow_freq = []
        for i in range(9):
            _, logits = enc.router(enc.slot_attn(enc.dino(frames[i:i+1])))
            slot_slow_freq.append(logits.softmax(-1)[..., 0])
    
    slow_stability = torch.stack(slot_slow_freq, dim=0).std(dim=0)  # (K,)
    # 至少一半 slot 应稳定（std < 0.1）
    stable_slots = (slow_stability < 0.1).sum().item()
    assert stable_slots >= 8, f"Router 不稳定，只有 {stable_slots} 个 slot 稳定分类"
```

### S3 · c_t 线性探针能预测本体感

**假设**：Stage-2 训练 5000 步后，冻结 encoder，从 c_t 线性预测 ee 位置，R² > 0.5。

```python
def test_c_predicts_ee_position():
    # 从 replay buffer 拿 1000 个样本
    dataset = load_replay_samples(n=1000)
    encoder = load_stage2_checkpoint("step_5000.pt")
    
    with torch.no_grad():
        cs = [encoder(x.imgs_seq, x.q_seq).c for x in dataset]
        cs = torch.cat(cs)                          # (1000, 128)
        ee_positions = torch.stack([x.q_seq[-1, :3] for x in dataset])  # (1000, 3)
    
    # 简单 linear regression
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    reg = LinearRegression().fit(cs[:800].cpu(), ee_positions[:800].cpu())
    pred = reg.predict(cs[800:].cpu())
    r2 = r2_score(ee_positions[800:].cpu(), pred)
    
    assert r2 > 0.5, f"c_t 无法编码 ee 位置，R² = {r2}"
```

### S4 · Slot-swap consistency

**假设**：随机交换 2 个 batch 里的 slow slot，动作应几乎不变。

```python
def test_slot_swap_action_invariance():
    encoder = load_stage2_checkpoint("step_100000.pt")
    policy = load_policy("step_100000.pt")
    
    batch = load_batch(size=64)
    
    with torch.no_grad():
        # 原始
        z_orig = encode_to_z(encoder, batch)
        a_orig = policy(z_orig)
        
        # slot-swap
        z_swap = encode_with_swap(encoder, batch, perm=torch.randperm(64))
        a_swap = policy(z_swap)
        
        cos_sim = F.cosine_similarity(a_orig, a_swap, dim=-1).mean()
    
    assert cos_sim > 0.85, f"Swap 后动作变化太大，cos = {cos_sim}"
```

如果这个不过，说明 P3 训练不到位，需要延长训练或提高 λ_swap。

**Sanity 层的核心哲学**：**用最小可能的数据规模验证方法性质**。合成视频 10 帧比真实数据 10 万帧快 1000×，还更能定位问题。

---

## L4 · 集成测试（1/天）

**目标**：全 pipeline 端到端能跑，reward 曲线合理。

### I1 · TD-MPC2 单任务 Sanity（能否复现 TD-MPC2 论文数字）

跑 TD-MPC2-pixel 在 DMC walker-walk 上 100K 步。**期待**：return > 500（TD-MPC2 论文报告 ~700）。这一步是流水线整体通路验证，也是 §IV Table IV 里 TD-MPC2 那一行的实测值来源。

```bash
# 一次性跑，记录到 wandb
python train.py \
    method=tdmpc2 \
    env=dmc_walker_walk \
    steps=100000 \
    seed=0 \
    wandb.project=hippoact-sanity
```

### I2 · HippoAct 单任务能不能训

同环境跑 HippoAct 100K 步。**期待**：
- 每一项 loss 都平稳（不 NaN、不爆炸）
- Return 曲线上升（不至少匹配 baseline）
- Router 分布稳定在 60/40 到 80/20 之间

**如果 I2 return 曲线不升**：立刻停下，回头检查 Sanity 层 S1-S4 是不是都过。90% 的 bug 会在 Sanity 层暴露。

### I3 · 存 / 读 checkpoint 一致性

```python
def test_checkpoint_roundtrip():
    encoder = HippoActEncoder().cuda()
    imgs, proprio = get_test_input()
    
    with torch.no_grad():
        out_before = encoder(imgs, proprio)
    
    torch.save(encoder.state_dict(), "/tmp/ckpt.pt")
    encoder2 = HippoActEncoder().cuda()
    encoder2.load_state_dict(torch.load("/tmp/ckpt.pt"))
    
    with torch.no_grad():
        out_after = encoder2(imgs, proprio)
    
    assert torch.allclose(out_before.c, out_after.c, atol=1e-5)
```

**这个测试拯救过我很多次**——checkpoint 兼容性 bug 在真机部署时最麻烦。

---

## L5 · 行为验证（1/周）

**目标**：验证论文核心 claim，不是 bug，而是**方法真的成立**。

### B1 · 背景鲁棒性（对应 §IV.C, Q2）

训完 DMC walker-walk（clean），zero-shot 评测 DCS-Hard。**期待**：HippoAct retention > 0.75，TD-MPC2-pixel retention < 0.5。

### B2 · Sim-to-Sim transfer（对应 §IV.D，无真机时的替代）

Robosuite (MuJoCo) 训 → ManiSkill3 (SAPIEN) 测。**期待**：HippoAct 成功率降幅 < 20%，pixel baseline 降幅 > 40%。

### B3 · 表征质量（对应 §IV.E）

t-SNE 可视化 c_t，按任务/背景分色。**期待**：按任务聚类，不按背景聚类。这是一张附录图。

### B4 · OOD detection AUROC（对应 §IV.F）

用 T-safety 协议测。**期待**：AUROC > 0.9。

**L5 每周跑一次**是因为这些实验单次 1-4 小时；一周节奏能保证在 12 周 timeline 内有 12 个数据点。

---

## L6 · 真机安全测试（上真机**每次**都做）

**顺序不能乱**。任何一步失败都不能进行下一步。

### R1 · Emergency stop 独立测试

- 按下物理 E-stop → 机械臂在 0.5 秒内停止
- 通过 ROS/rospy 发 SIGKILL → 机械臂进 hold mode
- **不接你的策略代码，也不接 HippoAct**——这层要在系统级独立验证

### R2 · Joint limit + workspace bounding box

策略输出被 clip 到关节 limit（安全比默认 ROM 保守 10°）+ workspace 立方体（保守 5 cm）。跑 10 分钟 random policy，检查：
- 没有 joint 超限报警
- 没有触碰桌面
- 没有超过 workspace box

### R3 · Fallback controller 冷启动

**不接策略**，直接测 safety gate 的 fallback：手动触发 gate → 机械臂应立即进入 "hold current joint config" 状态并停留。

```python
# 单独脚本
def test_fallback():
    robot.move_to_home()
    time.sleep(2)
    initial_q = robot.get_joint_positions()
    
    # 模拟 gate 触发
    for _ in range(30):  # 1 秒 × 30 Hz
        robot.hold_position()
        time.sleep(1/30)
    
    final_q = robot.get_joint_positions()
    assert np.max(np.abs(final_q - initial_q)) < 0.01, "hold_position 不稳定！"
```

### R4 · Camera → policy 延迟测试

测 image → DINOv2 → slot → binding → policy → action → robot 一环 latency。**期待** < 50 ms。如果 > 100 ms，控制不稳。

### R5 · Sim-to-real one-shot（先跑最简单版本）

不带 slot swap augmentation 直接跑 sim policy 上真机。**先看 baseline sim-to-real gap 有多大**——这是你评估 augmentation 效果的对照。

**顺序**：R1 → R2 → R3 → R4 → R5 → 才能跑完整 policy。**跳步 = 事故**。

---

## 关键：Dashboard / 监控（研究代码的 CI 替代品）

单元测试解决"代码有没有 bug"，但 **RL 代码的很多问题是"学习动态出问题"**，只有可视化才能发现。

### 必装的 dashboard

用 W&B 或 TensorBoard，每 step 记录：

**A. 数值健康**
- 各 loss 项（`L_slot`, `L_slow`, `L_route`, `L_predict`, `L_align`, `L_swap`, `L_TDMPC2`）
- grad_norm 分模块（encoder / router / binding / TDMPC2）
- 参数范数分模块

**B. 学习动态**
- Gumbel τ 当前值
- Slow slot 比例（应稳定 60-80%）
- Slot 平均 pairwise cosine（应稳定 <0.5）
- c_t norm 均值方差
- Actor / Q 值

**C. 环境指标**
- Return / success rate（每 eval）
- Episode length

**D. 表征质量（每 10K 步一次）**
- Slot alpha 可视化（8 张图 × 16 slot 的 mask heatmap）
- c_t t-SNE
- Reconstruction P vs P_hat 对比图

**没有这个 dashboard，出问题你只能猜**。

---

## 什么不要测（避免过度工程）

研究项目里，**下面这些别写测试**：

- 边界情况（K=1, T=1, B=1 之类）——研究代码从不这么用
- 特定输入的精确数值输出——随机初始化下没意义
- 每个内部函数的每个分支——研究代码变得快，测试维护成本高
- 视觉表现（"这张图 slot attention 应该是这样"）——用 dashboard 目视审阅代替

---

## 推荐工具栈

```
pytest              # 单元测试
pytest-xdist        # 并行跑 test
wandb               # 主 dashboard （比 TensorBoard 好，团队协作）
hydra-core          # 配置管理（避免 hard-code 超参）
rich                # 好看的终端 log
mlflow              # checkpoint & artifact 追踪
```

单元测试用 pytest 就够了；行为测试用 W&B 记录，不要写成 pytest 断言（因为不 deterministic）。

---

## 我的建议节奏

| 阶段 | 每天 | 每周 | 每次改架构 |
|---|---|---|---|
| **Week 1-2**（架构初期） | L1 + L2 | L3 全套 | L3 全套 |
| **Week 3-6**（训练主体） | L1 | L4 (I1/I2) | L3 (S1/S2) |
| **Week 7-9**（实验冲刺） | L1 | L5 一项 | L4 |
| **Week 10-13**（真机 + 写作） | L1 + L6 检查 | 修 bug | — |

**最重要的一句**：**L3 (Sanity) 是研究代码的 CI**。跑通 S1-S4 后再跑真实数据，能省 80% 的调试时间。剩余 20% 靠 dashboard 目视审阅。

---

## 自检 checklist

- [ ] L1 smoke 已经加进 pre-commit hook 了吗？
- [ ] L2 里的 `test_binding_mask_effective` 你实现出来了吗？（这是 P2 的核心正确性保障）
- [ ] L3 的 S1-S4 你能在 30 分钟内跑完吗？（如果不能，说明你的最小数据 pipeline 太慢）
- [ ] Dashboard 里有没有 slow slot 比例 + c_t 线性探针 R² 这两个指标？（缺一不可）
- [ ] 上真机前 R1-R4 有没有做？（如果没有，别上）

这五个都能勾上，你的开发效率会比"改代码 → 跑实验 → 出错猜 → 改回去"的循环快 3-5 倍。
