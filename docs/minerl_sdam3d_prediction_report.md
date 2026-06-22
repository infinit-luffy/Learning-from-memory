# MineRL SDAM-3D 场景预测代码说明

## 目标

这部分代码用于回应审稿人关于 3D 场景预测的要求。第一版不做完整 Minecraft 强化学习，而是做离线 scene prediction：

```text
MineRL-style RGB/action sequence
-> static scene memory
-> dynamic/action-conditioned state
-> z_assoc
-> predict next frame, next latent, changed region
```

这样可以验证 SDAM 的静态-动态关联记忆是否能从 2D Atari 扩展到第一人称 3D 场景。

## 数据接口

代码位置：

```text
src/sdam/data/minerl_sequence.py
```

当前使用预处理后的 `.pt` shard，不直接依赖 MineRL 包。每个 shard 是一个字典：

```python
{
    "obs": Tensor[N, T, C, H, W],
    "actions": Tensor[N, T - 1, action_dim],
}
```

dataset 输出：

```text
obs         = 前 T-1 帧
actions     = 前 T-1 个动作
next_obs    = 第 T 帧
change_mask = abs(next_obs - last_obs) > threshold
```

这个设计让本地测试和服务器 smoke test 不依赖 MineRL 安装；后续可以单独添加 MineRL 原始轨迹到 `.pt` shard 的转换脚本。

## 模型

代码位置：

```text
src/sdam/models/sdam_3d.py
```

核心模块：

- `frame_encoder`: 把每帧 RGB 图像编码成视觉特征。
- `static_head`: 对上下文帧特征做时间平均，得到 static scene memory。
- `dynamic_gru`: 融合 frame feature 和 action，得到 action-conditioned dynamic sequence。
- `association`: 融合 static memory、当前 dynamic state 和当前 action，得到 `z_assoc`。
- `next_latent_head`: 根据 `z_assoc`、dynamic state 和 action 预测下一 latent。
- `decoder`: 预测下一帧 RGB。
- `change_head`: 预测变化区域 mask。

输出包括：

```text
static
dynamic_seq
z_assoc
pred_next_latent
target_next_latent
pred_next_frame
pred_change_mask
recon_frame
```

## 损失函数

代码中的损失为：

```text
frame_loss  = MSE(pred_next_frame, next_obs)
latent_loss = MSE(pred_next_latent, target_next_latent.detach())
change_loss = BCE(pred_change_mask, change_mask)
recon_loss  = MSE(recon_frame, current_frame)

total =
    frame_weight  * frame_loss
  + latent_weight * latent_loss
  + change_weight * change_loss
  + recon_weight  * recon_loss
```

其中 `target_next_latent` 使用 detach，避免预测器和目标编码器同时漂移。

## 训练入口

配置：

```text
configs/minerl/navigate_sdam_prediction.yaml
```

脚本：

```bash
python scripts/train_minerl_sdam_prediction.py \
  --config configs/minerl/navigate_sdam_prediction.yaml \
  --train-steps 1000 \
  --output-dir runs/minerl/navigate_sdam_prediction_smoke \
  --device cuda
```

输出 checkpoint：

```text
runs/minerl/navigate_sdam_prediction_smoke/sdam_3d_prediction.pt
```

## 下一步实验

建议实验顺序：

1. 用 synthetic `.pt` shard 跑通 smoke test。
2. 从 MineRL NavigateDense 或 Treechop 轨迹预处理出 `.pt` shard。
3. 跑 SDAM-3D 与 ConvVAE+GRU / ConvLSTM / RSSM-style baseline 的 next-frame prediction 对比。
4. 做 ablation：去掉 static memory、去掉 action、去掉 `z_assoc`。
5. 保存 current / target next / predicted next / error map 可视化，用于论文图。
