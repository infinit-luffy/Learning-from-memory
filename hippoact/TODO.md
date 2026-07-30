# TODO — 服务器 Agent 执行清单

> 权威任务清单。判据不过 → 停下带数据回 cowork。
> 最后更新：2026-07-30（CP6 之后 — **synthetic phase 正式关闭**）

---

## 🎉 Synthetic Phase 关闭声明

CP6 达成全部 exit criteria：
- Cohen's d = **+1.889**（判据 ≥ 1.0）、秩 AUC 0.925
- FAST objectness 4.44 / SLOW 0.09（几乎纯净分离）
- 定位富集 18.78×、coverage 1.000、identity 0.466、exclusivity 0.430——**四项历次最好，无取舍**
- Router 泛化超过训练代理（learned 1.889 > proxy 阈值路由 1.503）

**方法定型（Stage-1 配方）**：
```
encoder:  DINOv2 frozen + SlotAttention(iters=3, dim=128) + shared init
routing:  slow_signal = alpha_connectivity (neighbor coherence)
loss:     L_slot + λ_slow·BCE(connectivity target) + L_route + L_div
不需要:   carryover / 跨帧匹配 / learned queries / 任何时间类信号
```

已取消的计划：M1b / M1c / M2 alpha-displacement patch（GT.4 否掉整类时间信号）。

**遗留决策（Phase 2 前处理）**：
- [ ] `configs/default.yaml` 把 `loss.slow_signal` 默认切到 `alpha_connectivity`
      （CP6 为了可回退默认还是 content_diff；现在 CP6 已验证，切默认）
- [ ] 尺寸混淆 limitation（大物体被弱化，corr = −0.247）记入论文 §V

---

## Phase 2 — DCS + TD-MPC2 集成

详细步骤见 `PHASE2_PLAN.md`（cowork 出）。摘要：

### P2.1 环境搭建（1 天）
- fork `github.com/nicklashansen/tdmpc2` 到 `third_party/tdmpc2`
- 装 dm_control + distracting_control（DAVIS-2017 背景视频）
- 判据：TD-MPC2 官方 pixel config 在 walker-walk (clean) 跑 100K steps，
  return ≥ 500（官方 ~700@500K，100K 打半即可确认管线）

### P2.2 Baseline 数字（2-3 天，挂机）
- TD-MPC2-pixel 在 {walker-walk, cheetah-run} × {clean, distracting-easy}
  各 500K steps × 3 seeds
- 这些数字直接进论文 Table IV/V 的 TD-MPC2-pixel 行（替换 ○ 估算值）
- 判据：clean walker-walk final return 650-750（对齐官方），不对齐先停

### P2.3 HippoAct encoder 接入（2-3 天）
- 按 PHASE2_PLAN.md 的接口方案替换 h_phi
- Stage-1 预训练：用 DCS 环境 random policy 采 50K 帧（clean + easy 混合），
  跑定型配方
- **训练前置检查（沿用 CP6 纪律）**：在 DCS 帧上测 connectivity 信号对
  walker 身体部件的富集度；DCS 没有物体 annotations，用 walker 的
  几何 proxy（躯干/四肢在画面中的已知运动区域）或人工标 50 帧
- 判据：Stage-1 slot alpha 图上 walker 身体有专属 slot，背景（视频）被
  弥散 slot 覆盖且 router 判 slow

### P2.4 端到端对比（1 周挂机）
- HippoAct vs TD-MPC2-pixel，walker-walk distracting-easy，500K × 3 seeds
- 判据（软）：HippoAct ≥ pixel 的 90%（sample efficiency 打平即可，
  主故事在 Q2 robustness）
- 判据（硬，Q2 preview）：easy 训练 → hard zero-shot，HippoAct retention
  显著高于 pixel（这是论文核心 claim 的第一个真实数据点）

### 常备规则（不变）
1. 训练前先量数据（CP6 前置检查两条是范本）
2. 单变量原则
3. 判据前置
4. **先看数据本身**——新度量投产前渲染出来目视（GT.7 规则 1）
5. **真值用生成过程的已知量，不从观测反推**（GT.7 规则 2）
6. 卡住带数据回 cowork

---

## 论文侧任务（cowork 负责，记录在此供对照）

- [ ] §III.D.3 完全重写：路由信号从时间不变性 → alpha 空间连通性；
      叙事从 "slow/fast temporal" 调整为 "object/background via spatial
      compactness"，保留 slow/fast 术语但依据改写
- [ ] §III 设计前提修改：GT.3 的机制发现（时间不变性 = 跟踪成功的标志）
      作为 motivation 写进去——"为什么不用时间信号"现在有完整的实证回答
- [ ] §IV.G Diagnostic Protocol 成型：GT.7 两条规则 + oracle 对照 +
      "上界低于实测=度量bug" + SNR 分解，素材全在 TODO_RESULT.md
- [ ] §IV.I negative results 重写：删守恒/瞬态（GT.2 推翻），
      换成「时间类路由信号的系统性失效」（GT.3/GT.4，更强的结果）
- [ ] §V limitation：尺寸混淆（大物体弱化）、连通性在 cluttered 场景的
      未验证性
- [ ] Table VII：exclusivity 0.43 替换旧的 purity 目标 0.86；
      coverage 1.000 + 富集 18.78× 作为主定位指标

## 汇报格式（不变）

```
Step N 结果：判据对照表 / 关键数字 / 异常观察 / 卡点
```
