# MineRL SDAM-3D Prediction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a minimal offline MineRL-style 3D scene prediction pipeline for SDAM.

**Architecture:** Use preprocessed trajectory shards saved as `.pt` dictionaries so tests and smoke runs do not require the MineRL package. Train an action-conditioned SDAM-3D model that predicts next frame, next latent, and changed region from a short RGB/action sequence.

**Tech Stack:** PyTorch, PyYAML, existing `sdam.config`, `sdam.models`, and thin script entrypoints.

---

### Task 1: Config And Dataset Contract

**Files:**
- Modify: `src/sdam/config.py`
- Create: `src/sdam/data/minerl_sequence.py`
- Test: `tests/test_minerl_prediction.py`

- [ ] Write tests for loading `configs/minerl/navigate_sdam_prediction.yaml`.
- [ ] Write tests for a `.pt` shard dataset returning `obs`, `actions`, `next_obs`, and `change_mask`.
- [ ] Implement dataclasses and validation for MineRL prediction config.
- [ ] Implement `MineRLSequenceDataset` with shape normalization and frame-difference masks.

### Task 2: SDAM-3D Model

**Files:**
- Create: `src/sdam/models/sdam_3d.py`
- Modify: `src/sdam/models/__init__.py`
- Test: `tests/test_minerl_prediction.py`

- [ ] Write tests for forward output shapes.
- [ ] Write tests that loss is scalar and backpropagates.
- [ ] Implement frame encoder, static memory, dynamic GRU, action-conditioned association, next-latent predictor, frame decoder, and change-mask head.

### Task 3: Training Experiment And CLI

**Files:**
- Create: `src/sdam/experiments/minerl_prediction.py`
- Create: `scripts/train_minerl_sdam_prediction.py`
- Create: `configs/minerl/navigate_sdam_prediction.yaml`
- Modify: `README.md`
- Test: `tests/test_minerl_prediction.py`

- [ ] Write tests for one train step over a synthetic shard.
- [ ] Write tests for CLI `--help`.
- [ ] Implement build/train/evaluate helpers.
- [ ] Add server smoke commands to docs.
