# Atari EnvModel RSSM-like 改动报告

## 背景

上一版 `main_vector_dqn` pipeline 更接近 main 分支旧实现：DQN 输入是
`dynamic_features(4 * 32) + background_latent(32)`，总共 160 维。这个版本的问题是：

- `ENV_MODEL_V2.connection_recog` 训练出的静态-动态关联表示 `z_assoc` 没有进入 RL 输入。
- env model 的原图重建损失使用 BCE，连续灰度图上容易出现较高平台。
- dynamic feature 只做当前前景重建，没有显式学习下一步动态预测。

本次改动把 env model 训练目标改成更接近 RSSM 的形式，并把 `z_assoc` 接入 RL vector。

## 核心代码位置

- `src/sdam/experiments/main_vector_dqn.py`
  - `MainEnvModelV2`
  - `collect_main_env_model_dataset`
  - `train_main_env_model_from_dataset`
  - `MainVectorObservationWrapper`
  - `MainVectorVecEnvWrapper`
- `scripts/pretrain_main_env_model.py`
- `scripts/run_main_atari_pipeline.py`

## EnvModel 训练目标

新的 env model 输入是 5 帧 foreground binary sequence：

```text
feature_seq = [binary_{t-3}, binary_{t-2}, binary_{t-1}, binary_t, binary_{t+1}]
```

前 4 帧作为上下文，第 5 帧作为 next feature prediction target。

训练过程：

```text
context_features = feature_recog.encode(binary_{t-3:t})
next_feature     = feature_recog.encode(binary_{t+1})
z_assoc          = connection_recog.encode(context_features, background_low)
org_hat          = connection_recog.decode(z_assoc)
pred_next        = rssm_predictor(concat(z_assoc, context_features[-1]))
rec_feature      = feature_recog.decode(context_features[-1])
```

新的损失函数：

```text
feature_recon_loss = BCE(rec_feature, binary_t)
org_recon_loss     = MSE(org_hat, original_frame_t)
next_pred_loss     = MSE(pred_next, next_feature)
total              = feature_recon_loss + org_recon_loss + pred_weight * next_pred_loss
```

其中 `pred_weight` 默认是 `1.0`，可以通过 CLI 调整：

```bash
--prediction-weight
--env-prediction-weight
```

实现里 `next_feature` 在 prediction loss 中作为 detached target 使用，避免预测器和目标编码器同时漂移。dynamic feature encoder 仍然通过当前帧 foreground reconstruction 训练；同一帧在滑动窗口推进后也会成为 reconstruction target。

## RSSM-like 部分

这不是完整 Dreamer/RSSM，但采用了类似思想：

- `context_features` 类似 posterior dynamic states。
- `connection_recog` 用 GRU 汇聚动态序列，并用 static background 初始化 hidden state。
- `z_assoc` 是静态背景条件下的关联动态状态。
- `rssm_predictor` 根据 `z_assoc` 和当前 dynamic feature 预测下一步 dynamic feature。

也就是说，模型不只重建当前帧，还要学习“当前关联状态能否预测下一步动态状态”。

## RL Vector 改动

上一版 DQN 输入：

```text
dynamic_features(128) + background_latent(32) = 160
```

新版 DQN 输入：

```text
dynamic_features(128) + z_assoc(32) + background_latent(32) = 192
```

含义：

- `dynamic_features`: 最近 4 帧 foreground binary 的 latent features。
- `z_assoc`: env model 学到的 static-dynamic associative state。
- `background_latent`: VAE 背景/static latent。

这让 RL 真正使用 `connection_recog` 训练出的核心关联表示，而不是只使用 foreground feature stack。

## 兼容性

`env_model.load_state_dict(..., strict=False)` 允许旧 checkpoint 被加载用于调试。不过旧 checkpoint 没有 `rssm_predictor` 参数，正式实验应重新训练 env model。

## 建议实验

先跑短检查：

```bash
python scripts/run_main_atari_pipeline.py \
  --config configs/atari/alien_sdam_ppo.yaml \
  --output-dir runs/alien/rssm_vector_check \
  --vae-episodes 20 \
  --vae-train-steps 5 \
  --env-episodes 20 \
  --env-train-steps 5 \
  --rl-algo dqn \
  --timesteps 1000 \
  --eval-episodes 1 \
  --device cuda
```

正式对比：

```text
NatureCNN DQN 5M
Old 160-D vector DQN 5M
New 192-D z_assoc vector DQN 5M
```

如果新 192-D 版本仍明显低于 NatureCNN DQN，需要进一步保留 foreground spatial map，避免把关键目标/子弹位置过早压缩成 32 维。
