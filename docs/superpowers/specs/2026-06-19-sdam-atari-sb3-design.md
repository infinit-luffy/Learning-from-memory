# SDAM Atari Stable-Baselines3 Design

## Goal

Add the first Atari reinforcement-learning integration for SDAM by using Stable-Baselines3's policy extension points. The milestone should make SB3's Atari PPO pipeline consume SDAM memory as its image feature representation while keeping the existing synthetic SDAM framework intact.

## Background

The first SDAM milestone implemented the core static-dynamic-associative encoder and a synthetic prediction task. The paper's next experimental step is Atari-style visual decision making. Stable-Baselines3 already provides PPO, Atari preprocessing helpers, vectorized environments, rollout collection, optimization, logging, and checkpointing. This milestone should reuse those pieces and replace only the image feature extractor used by `CnnPolicy`.

The SB3 integration point is `policy_kwargs["features_extractor_class"]`. SB3's Atari example path uses `make_atari_env` plus `VecFrameStack`, which produces a stacked image observation suitable for `CnnPolicy`. SDAM will interpret that frame stack as a short temporal window.

## Scope

This milestone will implement:

- An SB3-compatible SDAM Atari feature extractor.
- A small Atari experiment builder that creates Atari envs, frame stacking, PPO, and SDAM policy kwargs.
- A YAML config for an Atari PPO run.
- A training script that reads the config, builds the SB3 model, trains for a configurable number of timesteps, and saves the model.
- Tests for observation reshaping, extractor output shape, config validation, optional dependency behavior, and script argument wiring.

This milestone will not implement:

- A custom PPO algorithm or custom rollout buffer.
- A custom policy/value loss.
- Automatic NatureCNN versus SDAM benchmark sweeps.
- Atari ROM download logic.
- Default tests that require a real Atari ROM.

## Architecture

The integration should be additive:

```text
src/sdam/
  config.py
  policies/
    __init__.py
    encoder_adapter.py
    sb3_atari.py
  experiments/
    __init__.py
    atari.py
configs/
  atari/
    sdam_ppo.yaml
scripts/
  train_atari_sdam.py
tests/
  test_atari_config.py
  test_sb3_atari_policy.py
  test_atari_experiment.py
```

`sdam.policies.sb3_atari` owns SB3 feature-extractor compatibility and tensor conversion. `sdam.experiments.atari` owns SB3 model and environment construction. The script stays thin and delegates to the experiment package.

## Feature Extractor Design

`SDAMAtariFeaturesExtractor` should subclass SB3's `BaseFeaturesExtractor` when SB3 is installed. It will accept an Atari image observation space and create an `SDAMEncoder`.

Expected observation format:

```text
observations: [B, T, H, W]
```

For the default SB3 Atari path, `T` is `n_stack`, normally 4, and each frame is grayscale 84x84. The extractor converts observations to:

```text
sdam_obs: [B, T, 1, H, W]
```

The extractor should:

- Cast observations to `float32`.
- Normalize uint-like image values to `[0, 1]` by dividing by 255 when the maximum value is greater than 1.
- Validate rank and sequence length with clear `ValueError`s.
- Run `SDAMEncoder`.
- Project the SDAM memory to `features_dim` using a linear layer and `ReLU`.
- Return `[B, features_dim]` for SB3 policy/value heads.

The extractor should not require proprioception or action tensors in the first Atari milestone. `q_dim` and `action_dim` should remain 0 for Atari config.

## Policy Construction

The Atari experiment should expose a helper:

```python
def build_sdam_atari_policy_kwargs(config: AtariSDAMConfig) -> dict:
    ...
```

The returned dictionary should be suitable for:

```python
PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, ...)
```

The helper should include:

- `features_extractor_class`: `SDAMAtariFeaturesExtractor`
- `features_extractor_kwargs`: SDAM dimensions, sequence length, and output `features_dim`
- optional `net_arch` from config if present

The first implementation does not need a new `ActorCriticPolicy` subclass. SB3's `CnnPolicy` remains the policy class, and SDAM replaces the default CNN feature extractor.

## Atari Environment Construction

`build_atari_env(config)` should:

- Import SB3 and Atari dependencies lazily.
- Call `make_atari_env(config.env.env_id, n_envs=config.env.n_envs, seed=config.env.seed, wrapper_kwargs=...)`.
- Wrap the result with `VecFrameStack(env, n_stack=config.env.n_stack)`.
- Pass `terminal_on_life_loss` from config through `wrapper_kwargs`.

The config should default to `terminal_on_life_loss=False` for more intuitive reset behavior during early experiments.

## Training Script

`scripts/train_atari_sdam.py` should:

- Accept `--config`, defaulting to `configs/atari/sdam_ppo.yaml`.
- Accept optional `--timesteps` to override config training timesteps.
- Accept optional `--save-path` to override the configured output path.
- Load config.
- Build env and PPO model.
- Call `model.learn(total_timesteps=...)`.
- Save the model.
- Close the environment if it exposes `close()`.

The script should fail with a clear message if Stable-Baselines3 or Atari dependencies are missing, explaining that the user should install the `atari` extra.

## Configuration

`configs/atari/sdam_ppo.yaml` should include:

```yaml
env:
  env_id: PongNoFrameskip-v4
  n_envs: 1
  n_stack: 4
  seed: 0
  terminal_on_life_loss: false

model:
  static_dim: 64
  dynamic_dim: 64
  assoc_dim: 128
  hidden_channels: 32
  features_dim: 256

ppo:
  learning_rate: 0.00025
  n_steps: 128
  batch_size: 64
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.1

training:
  total_timesteps: 10000
  save_path: runs/atari/sdam_ppo
```

The existing `SDAMConfig` is synthetic-specific. This milestone should add separate Atari dataclasses rather than overload synthetic config fields.

## Dependency Strategy

Stable-Baselines3 and Atari packages should be optional:

```toml
[project.optional-dependencies]
atari = [
  "stable-baselines3[extra]>=2.3",
  "gymnasium[atari,accept-rom-license]>=0.29",
  "ale-py>=0.8",
]
```

Core imports should not import SB3 at package import time. Modules that require SB3 should raise clear `ImportError`s only when the Atari functionality is used.

## Testing Strategy

Default tests should run without Stable-Baselines3, Gymnasium, ALE, or Atari ROMs.

Tests should cover:

- Atari config loading and validation.
- Missing/unknown config keys produce clear errors.
- A pure tensor conversion helper maps `[B, T, H, W]` to `[B, T, 1, H, W]`.
- The conversion helper rejects invalid rank and wrong sequence length.
- With a local fake `BaseFeaturesExtractor`, `SDAMAtariFeaturesExtractor` can be instantiated and returns `[B, features_dim]`.
- `build_sdam_atari_policy_kwargs` returns a dict containing the expected extractor class and kwargs.
- The training script parses arguments and calls the experiment runner with overrides, using monkeypatching rather than launching a real Atari environment.

Optional integration tests may be skipped when SB3 or Atari dependencies are absent. They should not be required for the default test suite.

## Error Handling

Errors should identify the missing component or invalid shape:

- Missing SB3 dependency: `Stable-Baselines3 is required for Atari experiments. Install with: pip install -e ".[atari]"`
- Invalid Atari observation rank: `observations must have shape [B, T, H, W]`
- Wrong frame stack length: `observations frame stack must be <sequence_length>`
- Invalid config fields should use `ValueError` and include the section/key path.

## Success Criteria

The milestone is complete when:

- `SDAMAtariFeaturesExtractor` can consume stacked Atari-style tensors and return fixed-size SB3 features.
- Atari config loads and validates independently of synthetic config.
- `train_atari_sdam.py` can build an SB3 PPO model when optional dependencies are installed.
- Default tests pass without Atari ROMs.
- Existing synthetic tests continue to pass.

## Future Extensions

After this milestone, future work can add:

- NatureCNN versus SDAM benchmark scripts.
- Dynamic-only and static-dynamic ablations.
- Evaluation scripts using SB3's `evaluate_policy`.
- TensorBoard logging of SDAM auxiliary statistics.
- A full custom `ActorCriticPolicy` if actor and critic need different SDAM memory projections.
