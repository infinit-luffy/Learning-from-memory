# SDAM Experimental Framework Design

## Goal

Build the first implementation of the paper idea "Static-Dynamic Associative Memory for Efficient Visual Decision Making" as a PyTorch research framework. The first milestone should provide the core representation-learning modules, a synthetic moving-object video environment, lightweight training and evaluation entry points, and tests that prove the static-dynamic-associative memory path is usable.

## Background

The paper proposes that visual decision agents do not need to store every high-dimensional frame. Many decision tasks contain slow static context and sparse local dynamic events. The framework should encode these factors separately:

- `b_t`: static context memory.
- `z_{t-K:t}`: recent dynamic event memory.
- `c_t`: temporal associative memory conditioned on the static context and optional proprioception/actions.
- `m_t`: compact decision state consumed by downstream policy or prediction heads.

This project starts from an empty repository, so the first implementation should establish a clean project layout rather than patch an existing codebase.

## Scope

The first version will implement a complete experimental skeleton, not only a single module file. It will include:

- PyTorch modules for static encoding, dynamic encoding, associative memory, and combined SDAM encoding.
- A synthetic moving-object video dataset with static background, optional clutter, a single moving target, dynamic masks, and position/velocity labels.
- Configuration files for reproducible synthetic experiments.
- Training and evaluation scripts for a lightweight predictive task.
- Tests for data generation, model interfaces, configuration loading, and shape validation.
- A reserved policy encoder interface for later Atari, robotics, PPO, SAC, or Dreamer integration.

The first version will not implement full Atari experiments, robot grasping environments, high-fidelity image reconstruction, long-horizon world-model rollouts, or an online RL algorithm.

## Architecture

The repository should use a small research-library layout:

```text
src/sdam/
  __init__.py
  config.py
  models/
    __init__.py
    static_encoder.py
    dynamic_encoder.py
    associative_memory.py
    sdam_encoder.py
    prediction_heads.py
  data/
    __init__.py
    synthetic_video.py
  losses/
    __init__.py
    predictive.py
    flow_matching.py
  policies/
    __init__.py
    encoder_adapter.py
  experiments/
    __init__.py
    synthetic.py
configs/
  synthetic/
    sdam.yaml
scripts/
  train_synthetic.py
  eval_synthetic.py
tests/
  test_models.py
  test_synthetic_data.py
  test_experiment_config.py
```

Each file should have one clear responsibility. The model files define reusable neural modules. The data file defines the synthetic dataset and batch format. The experiments package binds config, data, model, loss, and optimizer together. The scripts should stay thin and call experiment functions.

## Core Data Flow

The core model consumes a short video window:

```text
obs:     [B, T, C, H, W]
q:       [B, T, Q]      optional proprioceptive state
actions: [B, T - 1, A]  optional previous actions
```

The SDAM path is:

```text
obs window
  -> StaticEncoder(background estimate or current frame)
       -> b_t

  -> DynamicEncoder(frame differences)
       -> z_{t-K:t}

  -> AssociativeMemory(z_{t-K:t}, b_t, q_t, actions)
       -> c_t

  -> SDAMEncoder
       -> memory = concat(b_t, flatten(z_{t-K:t}), c_t, q_t)
```

`SDAMEncoder.forward` should return a structured dictionary:

```python
{
    "b": static_latent,
    "z_seq": dynamic_latents,
    "c": associative_latent,
    "memory": compact_memory_state,
    "aux": auxiliary_outputs,
}
```

The compact `memory` tensor is the downstream decision representation. The separate fields make ablations straightforward.

## Module Design

### StaticEncoder

`StaticEncoder` encodes slow-changing visual context. For the first version, it should estimate background from the input window by taking a temporal mean and passing the result through a compact CNN projection. This is simple, deterministic, and appropriate for the synthetic environment. The interface should allow later replacement with a learned background branch.

Input:

```text
obs: [B, T, C, H, W]
```

Output:

```text
b: [B, static_dim]
```

### DynamicEncoder

`DynamicEncoder` encodes motion-related local changes. It should compute adjacent-frame absolute differences and pass each difference frame through a shared compact CNN projection.

Input:

```text
obs: [B, T, C, H, W]
```

Output:

```text
z_seq: [B, T - 1, dynamic_dim]
```

### AssociativeMemory

`AssociativeMemory` encodes short-term relationships between dynamic events and static context. The first implementation should use a GRU over dynamic latents, with static latent, optional current proprioception, and optional summarized actions fused into the GRU output by an MLP.

Input:

```text
z_seq: [B, K, dynamic_dim]
b: [B, static_dim]
q: [B, Q] optional
actions: [B, K, A] optional
```

Output:

```text
c: [B, assoc_dim]
```

The class should expose enough metadata for `SDAMEncoder` to compute the final memory dimension.

### SDAMEncoder

`SDAMEncoder` wires the three modules together. It should validate shapes, call each component, and concatenate `b`, flattened `z_seq`, `c`, and optional current `q_t` into `memory`.

It should be usable as a generic observation encoder for downstream decision policies.

### Prediction Heads

The first training task should attach a small MLP head to `memory` to predict the next target position and velocity in the synthetic environment. This verifies that the compact memory state carries short-term dynamic trend information.

## Synthetic Video Dataset

The synthetic dataset should generate controlled sequences:

- Static background: random low-frequency pattern or geometric texture.
- Moving target: a small square or disk with position and velocity.
- Motion rule: linear motion with boundary reflection.
- Static clutter: optional same-colored or similar-looking distractors.
- Labels: target position, velocity, background image, dynamic mask, and final-frame target center.

Batch format:

```python
{
    "obs": Tensor[B, T, C, H, W],
    "target_position": Tensor[B, 2],
    "target_velocity": Tensor[B, 2],
    "dynamic_mask": Tensor[B, T, 1, H, W],
    "background": Tensor[B, C, H, W],
}
```

The target should be defined by motion rather than by a unique appearance cue, matching the paper's central task motivation.

## Training And Evaluation

`scripts/train_synthetic.py` should train SDAM plus a prediction head on the synthetic task. The main supervised objective is:

```text
L = MSE(predicted_position, target_position) + velocity_weight * MSE(predicted_velocity, target_velocity)
```

`scripts/eval_synthetic.py` should report:

- Position MSE.
- Velocity MSE.
- Compact memory dimension.
- Forward-pass latency on a small batch.

The first version does not need image reconstruction, RL return, or grasping success metrics.

## Losses

`losses/predictive.py` should implement the supervised position/velocity prediction loss used by the synthetic experiment.

`losses/flow_matching.py` should provide an explicit future-extension interface for latent flow association, but it should not be part of the default training path. It should raise a clear `NotImplementedError` if called before a real implementation is added.

## Configuration

`configs/synthetic/sdam.yaml` should define:

- Image shape and sequence length.
- Static, dynamic, associative, and memory dimensions.
- Dataset size, clutter count, object size, and motion speed range.
- Batch size, learning rate, training steps, and velocity loss weight.
- Optional proprioception and action dimensions, defaulting to disabled.

Configuration loading should validate required fields and report missing or inconsistent values with clear messages.

## Error Handling And Validation

The first implementation should validate common shape errors at module boundaries:

- `obs` must be rank 5: `[B, T, C, H, W]`.
- `T >= 2`.
- `q`, when provided to `SDAMEncoder`, must align with batch and time dimensions.
- `actions`, when provided, must align with batch and `T - 1`.
- `AssociativeMemory` should reject dynamic sequences with the wrong latent dimension.
- Configuration loading should fail early when required sections or values are absent.

Errors should be raised as `ValueError` with messages that identify the offending tensor or config field.

## Testing Strategy

The first version should include focused tests:

- Synthetic data tests:
  - generated observation tensors have expected shape and dtype;
  - target position changes over time;
  - dynamic masks are non-empty;
  - background is stable across a sequence.
- Model tests:
  - each encoder returns expected tensor shapes;
  - `SDAMEncoder` returns the expected dictionary keys;
  - optional `q` and `actions` can be omitted;
  - invalid shapes raise clear `ValueError`s.
- Config tests:
  - default YAML config loads successfully;
  - loaded config can instantiate the synthetic experiment;
  - missing required fields raise clear errors.

Tests should run quickly on CPU and should not require downloading datasets.

## Future Extensions

After this first milestone, the framework can grow in three directions:

- Add latent flow association as a real alternative to GRU association.
- Add Atari wrappers and compare raw image, whole-image latent, dynamic-only, static-dynamic, and SDAM variants.
- Add a motion-defined dynamic grasping simulator or robotics interface.

These extensions should reuse the same `SDAMEncoder` API and avoid changing the core representation contract unless tests and downstream adapters are updated together.

## Open Decisions Resolved

- Start with a full experimental framework rather than a single-file prototype.
- Use synthetic moving-object video as the first validation environment.
- Use GRU associative memory as the default first implementation.
- Keep latent flow association as a future extension behind an explicit interface.
- Predefine policy encoder adapters but do not implement a complete RL algorithm in the first milestone.
