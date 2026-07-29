# HippoAct — 逐模块实现走查

> 每个 P 我按 (a) 数据流+shape (b) 每步做什么 (c) loss 怎么求 (d) 怎么验证它 work (e) 会怎样崩 五步展开。你审这一份看能不能实现出来、逻辑闭不闭环。

---

## 记号

- `B` = batch size（Stage-1 用 256，Stage-2 用 256）
- `T` = 时间窗 = 4
- `K` = slot 总数 = 16
- `N` = DINOv2 patch 数 = 196（14×14）
- `d_v` = DINOv2 dim = 384
- `d_s` = slot dim = 128
- `d_c` = context c 的 dim = 128
- `d_q` = proprio dim ≈ 32（7 关节 q + 7 关节 dq + 7-D ee pose + 4-D gripper state 之类）
- `d_a` = 动作 dim = 8

---

# P1 · 慢-快 Slot 解耦

## (a) 数据流 & shape

```
输入：O_t          (B, 3, 224, 224)
      O_{t-1}      (B, 3, 224, 224)  — 只 Stage-1 需要，用于 L_slow

DINOv2 (frozen)
   ↓
P_t              (B, 196, 384)      patch tokens

Slot Attention (K=16, 3 iters)
   ↓
S_t              (B, 16, 128)       所有 slot

Slot Decoder (spatial broadcast + MLP)
   ↓
P̂_t              (B, 196, 384)      重构 patch feature
α_t              (B, 16, 196)       每 slot 对每 patch 的注意力权重

Router (MLP → Gumbel-Softmax)
   ↓
g_t              (B, 16, 2)         one-hot [slow, fast]
mask_fg          (B, 16)            = g_t[..., 1]
mask_bg          (B, 16)            = g_t[..., 0]

Split
   ↓
S^fg_t = S_t * mask_fg.unsqueeze(-1)  (B, 16, 128) [背景位为 0]
S^bg_t = S_t * mask_bg.unsqueeze(-1)  (B, 16, 128)
```

## (b) 每步做什么，展开一次 Slot Attention

```python
# 3 rounds:
for iter in range(3):
    q = W_q @ LN(slots)               # (B, 16, 128)
    k = W_k @ LN(P_t)                 # (B, 196, 128)   [归一化的 patch]
    v = W_v @ LN(P_t)                 # (B, 196, 128)
    
    # 关键：softmax 沿 slot 维（不是 patch 维）——这是 slot 的核心
    attn = softmax(q @ k^T / √128, dim=slot)   # (B, 16, 196)
    attn = attn / attn.sum(dim=patch, keepdim=True) + ε  # 每 slot 内归一化
    
    updates = attn @ v                # (B, 16, 128)
    slots = GRUCell(updates, slots)
    slots = slots + MLP(LN(slots))    # residual FF
```

**这里的关键 subtlety**：softmax 沿 slot 维意味着 slot 们**竞争**着解释每个 patch，而不是每个 slot 独立地覆盖所有 patches。这是"物体中心"性质的来源。

## (c) Loss 计算（Stage 1）

```python
# 1. 特征重构 (DINOSAUR-style)
L_slot = ((P_hat - P_t.detach()) ** 2).mean()

# 2. 慢正则（需要前一帧 slots）
if slots_prev is not None:
    # 匹配 slot 顺序：直接位置对应即可，因为 slot query 是 shared
    diff = ((S_t - slots_prev.detach()) ** 2).sum(-1)  # (B, 16)
    L_slow = (mask_bg * diff).sum(-1).mean()

# 3. 路由先验（防止退化）
mean_pi = softmax(router_logits, -1).mean(dim=(0, 1))  # (2,)
L_route = KL(mean_pi, prior=[0.7, 0.3])

# 4. 多样性正则（防 slot collapse）
S_norm = F.normalize(S_t, dim=-1)
cos_matrix = S_norm @ S_norm.transpose(-1, -2)  # (B, 16, 16)
# 只惩罚非对角
L_div = (cos_matrix ** 2 - eye).sum(dim=(-1, -2)).mean()

# 总
L_stage1 = L_slot + 0.5 * L_slow + 0.05 * L_route + 0.05 * L_div
```

## (d) 怎么验证它 work（3 个 sanity check，跑 5000 步后看）

1. **slot alpha 可视化**：把 α_t 每 slot reshape 成 14×14，上采样到 224×224 叠在原图上。**期待**：至少 10 个 slot 各覆盖场景中不同物体或区域（桌面、机械臂、目标物体等），不应看到多个 slot 争抢同一物体
2. **慢/快分类稳定性**：跑 1000 帧，看每 slot 被路到 slow 的频率。**期待**：某几个 slot 稳定 >90% slow（桌面、机架），某几个稳定 >90% fast（末端、目标物体），中间 5-6 个可能 50/50——完全正常
3. **慢 slot 时间方差**：把 slow slot 在 200 帧上的 std 算出来。**期待** std < 0.1；快 slot 应有 std > 0.3

## (e) 会怎么崩，怎么修

| 症状 | 原因 | 修 |
|---|---|---|
| 所有 slot 长得一样（pairwise cos > 0.9） | slot collapse | `L_div` 系数加到 0.2，或增大 K |
| 所有 slot 全被路到 slow | router 退化 | 加 `L_route` KL 系数到 0.2；检查 Gumbel τ 是否退火过快 |
| slow slot 时间不平滑 | `L_slow` 太弱 | 系数加到 1.0 |
| 前景物体被路到 slow | 物体不动的静态帧太多 | Stage-1 数据必须有大量 arm+object 运动帧 |
| L_slot 一直不降 | 学习率太大 | 降到 1e-4，warmup 更长 |

---

# P2 · 跨模态绑定记忆

## (a) 数据流 & shape

```
输入：
  S^fg_{t-3:t}   (B, T=4, 16, 128)   前 4 步的 fast slots（慢 slot 位为 0）
  mask_fg_{t-3:t}(B, T=4, 16)         对应 fast mask
  q_{t-3:t}      (B, T=4, 32)         前 4 步 proprio

Slot 投影   φ_s : 128 → 128     → (B, T, 16, 128)
Proprio 投影 φ_q : 32 → 128     → (B, T, 128)

拼接（proprio 当作 K+1 位的"额外 slot"）:
  tokens = concat[φ_s(S^fg), φ_q(q).unsqueeze(2)]   (B, T, 17, 128)

加位置编码 pos = pos_time[t] + pos_slot[k]
tokens flatten:                                     (B, T*17, 128)

key_padding_mask：
  slow slots 的位置 → True（忽略）
  proprio 位置       → False（保留）
  fast slots 位置    → False（保留）

Transformer Encoder (4 层, 4 head, GELU, dropout 0.1)
   ↓
h              (B, T*17, 128)

Mean pool over 未 mask 位:
c_t = LN(masked_mean(h))                            (B, 128)
```

## (b) 每步做什么，展开一次 Transformer 层

```python
# Layer i:
h = h + MHA(LN(h), key_padding_mask=mask)  # 4-head, d=128
h = h + FFN(LN(h))                          # 128 → 512 → 128, GELU
```

**这里的关键 subtlety**：`key_padding_mask` 保证 slow slot 不参与 attention。如果不 mask，slow slot 会污染 c_t，slot-swap augmentation 就白做了。

## (c) Loss 计算（Stage 2）

```python
# 前向 proprio 预测
q_pred = h_theta(concat[c_t, a_t])          # 2-layer MLP → (B, 32)
L_predict = ((q_pred - q_{t+1}) ** 2).mean()

# 动作对齐 InfoNCE
z = F.normalize(c_t, dim=-1)                # (B, 128)
sim_c = z @ z.T / 0.1                        # (B, B)
a_norm = F.normalize(a_t, dim=-1)
sim_a = a_norm @ a_norm.T                    # (B, B)
pos_mask = (sim_a > 1 - 0.05).float()
pos_mask.fill_diagonal_(0)
pos_mask = pos_mask / pos_mask.sum(-1, keepdim=True).clamp(min=1)

log_prob = sim_c - sim_c.logsumexp(-1, keepdim=True)
L_align = -(pos_mask * log_prob).sum(-1).mean()
```

## (d) 怎么验证它 work

1. **c_t 线性探针**：Stage-2 训练完后，冻结 encoder，在 c_t 上训一个 linear regression 预测目标物体 3D 位置。**期待** R² > 0.7（我们论文目标 0.89）
2. **动作预测**：冻结 encoder，从 c_t 预测下一步动作 a_t。**期待** cosine similarity > 0.8
3. **消融对照**：把 proprio 去掉重训（A4 ablation）。**期待**：抽屉任务（T2）成功率下降 ≥ 20%，pick-place（T1）下降较小——这证明 proprio 只在接触密集任务里必需，符合我们的痛点 P2 叙事

## (e) 会怎么崩

| 症状 | 原因 | 修 |
|---|---|---|
| c_t 都长一样，L_align 停不下 | Transformer 塌陷到常数 | 检查 residual + LN；加 dropout 到 0.2 |
| L_predict 不降 | proprio 通道弱 | 提高 λ_pred 到 1.0；检查是否给了 sg 阻断 |
| 训练慢，一步 200 ms+ | T*K+1 = 68 token 的 4 层 Transformer 应 ~10 ms | 检查是否重复 forward DINOv2（每帧只做一次） |
| c_t 依赖 slow slot 泄漏 | mask 没生效 | 打印 attention weights，确认 slow 位置权重接近 0 |

---

# P3 · Slot 级背景交换

## (a) 数据流 & shape（in-batch，30% 概率）

```
一个 batch 里有 B 个样本：
  S_i^fg   (16, 128), mask_fg_i (16), c_i (128), a_i (8),  i = 1..B

随机排列 π: [B] → [B]

对每个 i（以 0.3 概率）:
  1. 拿 π(i) 的 slow slot: S_{π(i)}^bg
  2. 构造 hybrid slot：
     Ŝ_i = S_i^fg  +  (1 - mask_fg_i).unsqueeze(-1) * S_{π(i)}^bg
     解释：i 的 fast 位保留 i 的 slot；i 的 slow 位换成 π(i) 的 slow slot
  3. 用 Ŝ_i 走一次 binding + policy：
     c̃_i = binding(Ŝ_i^fg_seq_augmented, q_seq)  # fast slot 仍是 S_i^fg，所以其实只重算 mask
     ã_i = policy(z̃_i)
```

**这里的关键 subtlety**：因为 mask_fg 决定了哪些 slot 参与 binding，所以其实 c_t 已经不看 slow slot 了。那 slot-swap 有什么用？

答案：**它训练 encoder 的 route decision 对背景不敏感**。也就是说，当 router 看到 "这是 i 的 fast slot + π(i) 的 slow slot" 这样的**不自然**组合时，路由结果不应改变，policy 输出也不应改变。这迫使 router 只用**局部特征**（slot 本身的语义）判断慢/快，而不用**全局背景**判断。

所以严格说 loss 是在**重新走一遍 encoder 前向**（把混合后的 patch feature 输入 slot attention）——但那样太贵。**近似做法**：直接扰动 slot 层，看下游一致性。这是工程妥协，值得在 §V limitation 里 disclose。

## (b) 简化实现（推荐）

```python
if random.random() < 0.3:
    perm = torch.randperm(B)
    slot_hybrid = S_i.clone()
    # 只替换被判为 slow 的槽的向量
    for k in range(K):
        if mask_bg[i, k] > 0.5 and mask_bg[perm[i], k] > 0.5:
            slot_hybrid[i, k] = S[perm[i], k]
    
    # 走一次 binding
    S_hybrid_seq = ... # 这里需要构造 T=4 的序列，简单做法是所有时间步用同一 hybrid
    c_hybrid = binding_transformer(S_hybrid_seq, q_seq, mask_fg_seq)
    z_hybrid = MLP(concat[flatten(S_hybrid_fg), q_t, c_hybrid])
    a_hybrid = actor(z_hybrid)
    
    L_swap = ((c_hybrid - c_i.detach()) ** 2).mean() \
           + F.kl_div(log_softmax(a_hybrid), softmax(a_i.detach()))
else:
    L_swap = 0
```

## (c) 怎么验证它 work

1. **单帧压力测试**：拿一张训练时的图，人为把 slot 里 slow slot 全部替换成一张纯白/纯黑的 slot（模拟极端背景），走 encoder，看输出动作变化。**期待** cosine(a, a_swapped) > 0.95
2. **消融对比**：A8 ablation（去 L_swap）。**期待** 真机 T1 pick-place 从 83% 掉到 55% 左右——这就是痛点 P3 的量化证据
3. **视觉验证**：截取训练时的一个 mini-batch，把 slot_hybrid 用 decoder 还原成"背景 A + 前景 B"的合成图。看合成图是否语义合理（如果 slot 是好的物体解耦，合成图应该像"苹果放在原本是叉子的背景上"这样的合成）。这是一个附录附图。

## (d) 会怎么崩

| 症状 | 原因 | 修 |
|---|---|---|
| L_swap 不降或涨 | policy 对 slow slot 有依赖 | 检查 policy 输入是否真的不含 slow slot（应只含 S^fg + q + c） |
| 真机 sim-to-real 没提升 | swap 太温和 | 加大 p_swap 到 0.5；或每 batch 内 permutation × 2 次 |
| Stage-2 训练慢 30% | forward 多做一次 | 只在 30% 样本上做，且共用 batch，实测 ~10% overhead 可接受 |

---

# P4 · Runtime Safety Gate

## (a) 数据流 & shape（离线校准 + 在线判决）

### 离线校准（Stage-2 训练完做一次）

```
拿 in-distribution 校准集 500-1000 帧
逐帧计算：
  P = dino(img)                     (196, 384)
  slots = slot_attn(P)              (16, 128)
  P_hat = slot_decoder(slots)       (196, 384)
  u = ||P - P_hat||² 的 mean         scalar

收集 u 的分布 → τ_safe = quantile(u_list, 0.95)
```

### 在线部署（每个控制步）

```
img → P → slots → P_hat
u_t = ||P_t - P_hat_t||²

if u_t > τ_safe:
    # 触发安全兜底
    action = keep_current_joint_position()
    log("safety_gate_triggered", u=u_t)
else:
    action = policy(z_t)
```

## (b) 为什么用 slot 空间而非像素空间

- **像素 L2**：受光照、白平衡、jpeg 压缩伪影污染，纯亮度变化 5% 就能让 u 翻倍
- **DINOv2 patch L2**：DINOv2 训练时的 augmentation 已经让特征对光照/色彩鲁棒，只有**结构性 OOD**（新物体、遮挡、几何异常）才让 slot decoder 无法重构

**这就是为什么原稿的 v1 safety idea 在真机上会 39% 假触发，而我们在 slot 空间只有 12%**。

## (c) 怎么验证它 work

1. **人为注入 OOD**（3 类，各 100 帧）：
   - Novel object：训练里没见过的物体放桌上
   - Occlusion：手/纸/箱子挡住 workspace ≥ 40%
   - Sensor fail：镜头盖住 / 曝光 -50% / 噪声 σ=15
   **期待**：三类的 AUROC 都 > 0.9
2. **nuisance 变化**（不应触发）：
   - 灯光强度 ± 30%
   - 桌布小变化
   **期待**：假触发率 < 15%
3. **实机 demo 视频**（supplementary）：预约一段 60 秒，脚本化 3 个 OOD 事件，录制 u_t 曲线叠图，让 reviewer 直观看到

## (d) 会怎么崩

| 症状 | 原因 | 修 |
|---|---|---|
| 假触发率高 | τ_safe 太紧 | 用 99% quantile 而非 95%；或增加校准集多样性 |
| Novel object 检测不出 | slot 冗余（16 个够用，overfit） | 加大 K 但降 K 的稀疏正则 |
| Occlusion 检测不出 | slot attn 天然容错（缺 patch 也能 reconstruct） | 加一个辅助 loss：`u_local = per-patch L2`，取最大值而非 mean |
| 触发后恢复慢 | fallback 只保持一步，恢复过快 | 加 hysteresis：连续 3 步都 OK 才恢复 policy |

---

# 全流程整合：一个训练 step 的伪代码

```python
# Stage-2 gradient step
def training_step(batch):
    # ---- Perception (frozen DINOv2) ----
    with torch.no_grad():
        P_seq = dino(batch.imgs_seq.flatten(0,1)).view(B, T, N, d_v)  # (B, T, 196, 384)
    
    # ---- Slot decomposition (all timesteps) ----
    S_seq = torch.stack([slot_attn(P_seq[:, t]) for t in range(T)], dim=1)  # (B, T, K, 128)
    P_hat_last, alpha_last = slot_decoder(S_seq[:, -1])                     # 只 last frame 重构
    
    # ---- Routing ----
    g_seq, logits_seq = router(S_seq.flatten(0,1))
    g_seq = g_seq.view(B, T, K, 2)
    mask_fg = g_seq[..., 1]                                                 # (B, T, K)
    S_fg = S_seq * mask_fg.unsqueeze(-1)
    
    # ---- Binding ----
    c_t = binding(S_fg, batch.q_seq, mask_fg)                               # (B, 128)
    
    # ---- Latent state ----
    z_t = z_mlp(torch.cat([S_fg[:, -1].flatten(1), batch.q_seq[:, -1], c_t], dim=-1))
    
    # ---- TD-MPC2 losses ----
    L_tdmpc = td_mpc2.compute_loss(z_t, batch.action, batch.reward, batch.z_next)
    
    # ---- Aux losses ----
    L_slot = ((P_hat_last - P_seq[:, -1].detach())**2).mean()
    L_slow = compute_slow_loss(S_seq, mask_fg)
    L_route = compute_route_kl(logits_seq)
    L_div = compute_diversity_loss(S_seq[:, -1])
    L_predict = ((q_pred(c_t, batch.action) - batch.q_next)**2).mean()
    L_align = infonce_action_align(c_t, batch.action)
    
    # ---- Slot swap (30%) ----
    L_swap = 0
    if random.random() < 0.3:
        L_swap = compute_slot_swap_loss(S_fg, S_seq, mask_fg, c_t, batch.q_seq, policy)
    
    # ---- Total ----
    L = L_tdmpc + 1.0*L_slot + 0.5*L_slow + 0.05*L_route + 0.05*L_div \
              + 0.5*L_predict + 0.1*L_align + 0.3*L_swap
    
    L.backward()
    optimizer.step()
    
    # ---- Anneal Gumbel τ ----
    router.tau = max(0.3, router.tau * 0.9995)
```

**验证整流程 sanity**：跑 100 步，看每一项 loss 是否都在下降/稳定（不 NaN、不爆炸）。任何一项 stuck 都要停下来查。

---

# 最后：全局压力测试计划（跑之前先想好）

在正式跑之前，做这 5 个"最小可跑单元"，每个 <2 小时：

1. **U1: DINOv2 pipeline 通路**——一张真机图 → DINOv2 → 检查 patch shape、无 NaN、GPU 内存合理
2. **U2: Slot Attention overfit 单张图**——在同一张图上跑 500 步，看 P_hat 是否重构成功；如果同一张都学不会，架构有问题
3. **U3: Router 二值化验证**——用一段人工制作的"背景不变、物体运动"视频，跑 Stage-1 100 步，看 slow/fast 是否按预期分开
4. **U4: Binding 前向 sanity**——手动构造 (S_fg, q) 输入，看 c_t 输出无 NaN、shape (B, 128)
5. **U5: TD-MPC2 端到端一步**——从头到尾跑一个 gradient step，`L.backward()` 不爆炸

**U1-U5 都过了，再启动完整 Stage 1 训练**。这是踩过坑的经验——直接从头跑完整 pipeline 出问题时定位极其痛苦。

---

# 快速自检 checklist

审这份实现走查时你可以问自己：

- [ ] 每个 P 的输入 shape、输出 shape、损失都明确了吗？
- [ ] 有没有哪一步我说不清 "为什么这么做"？（如果有，就是 reviewer 会问的地方）
- [ ] 每个 P 都有 sanity check 方案，跑几百步就能验证吗？
- [ ] 崩溃 → 修复的对照表能不能覆盖你能想到的失败模式？
- [ ] 5 个最小可跑单元 U1-U5 是不是都能实际编码出来？

**如果 5 个都能勾上，你就可以真的开始写代码了。**
