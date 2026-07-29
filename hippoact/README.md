# HippoAct

Object-centric, hippocampus-inspired visuomotor representations for robot learning.

- **Frozen DINOv2** patch features → **Slot Attention** → **Gumbel slow/fast router** → **Cross-modal Binding Transformer** → **TD-MPC2** latent planner.
- **Slot-space background swap** augmentation for label-free sim-to-real.
- **Reconstruction residual** as a runtime OOD safety gate.

Paper draft: `../paper_section_III_method.md`, `../paper_section_IV_experiments.md`.

---

## 1. Repo layout

```
hippoact/
├── pyproject.toml               # dependencies
├── configs/
│   └── default.yaml             # Table II hyperparameters
├── hippoact/                    # python package
│   ├── encoders/                # DINOv2 wrapper, Slot Attention, Router, Binding, top-level
│   ├── losses/                  # every auxiliary loss from §III
│   ├── augmentation/            # slot-swap augmentation
│   ├── safety/                  # runtime OOD gate
│   ├── training/                # Stage 1 pretrain loop
│   └── utils/                   # config, slot alpha visualization
├── scripts/
│   ├── pretrain_stage1.py       # entry point for Stage 1
│   └── overfit_test.py          # L3-S1 sanity: overfit one synthetic image
└── tests/
    ├── test_smoke.py            # L1: <10 seconds, run on every commit
    ├── test_modules.py          # L2: shape/frozen/mask correctness
    └── test_sanity.py           # L3: verify the method actually learns
```

Design docs (kept outside the code repo): `paper_section_III_method.md`,
`implementation_walkthrough.md`, `testing_strategy.md`.

---

## 2. Server setup

Tested target: **4 × NVIDIA A5000 (24 GB), CUDA 12.1, Python 3.10**.

```bash
# 1. Clone / rsync to server
scp -r hippoact/ user@server:~/hippoact/
ssh user@server
cd ~/hippoact

# 2. Create env
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# 3. Install PyTorch matched to your CUDA
pip install torch==2.3.0 torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Install the package + dev deps
pip install -e ".[dev]"

# 5. Sanity: unit tests should all pass in <1 min (CPU is fine)
pytest tests/test_smoke.py tests/test_modules.py -v
pytest tests/test_sanity.py -v -m sanity      # slower, ~2 min on CPU
```

The DINOv2 wrapper falls back to a mock CNN if `torch.hub` cannot reach the
network — tests use `HIPPOACT_FORCE_MOCK=1` and require no downloads.

For actual training, DINOv2 will download on first use to `~/.cache/torch/hub/`
(~90 MB). Pre-download on a machine that has internet:
```bash
python -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')"
```

---

## 3. First run — 40-second overfit test

Before spending GPU-days on real training, prove the architecture converges
on a single synthetic scene:

```bash
python scripts/overfit_test.py                # default: 8000 steps, ~40s on A5000
```

Success criterion (matches `testing_strategy.md` §S1): final `L_slot < 0.01`,
initial/final ratio > 10×.

**Expected loss trajectory** — loss starts near patch-feature variance (~6),
plateaus at ~2.0 (the "predict mean" solution) for the first 100-500 steps,
then breaks symmetry and descends to <0.01 by step ~5000-8000. The plateau is
normal Slot Attention warmup, not a bug.

**If final loss > 0.01 after 8000 steps, come back to me** — architecture
has a real problem. Do not continue to Stage 1.

---

## 4. Stage 1 — unsupervised representation pretraining

### 4.1 Data

Any RGB image loader works. Recommended source mix (~50-100 K frames):

- 30-50 K frames from **teleop demos** (all your T1/T2/T3 sessions)
- 30 K frames from **random arm motion** — 30 min per background × 4 backgrounds
- 5-10 K frames from **static scene sweeps** (multiple object arrangements, arm static)

Point the trainer at a directory of images:
```
data/frames/
├── teleop_T1/*.png
├── teleop_T2/*.png
├── random_motion_bg1/*.png
├── random_motion_bg2/*.png
└── static_scenes/*.png
```

### 4.1.1 Layout choices — clip layout (preferred) vs. flat layout (legacy)

**L_slow is a temporal-variance signal.** For it to be semantically meaningful,
consecutive samples must be *time-adjacent* frames of the *same scene*. The
loader auto-detects the layout of `--data-dir`:

- **Clip layout** *(preferred)*: `data-dir/clip_XXX/frame_YYYY.png`. Loader
  yields `(img_prev, img_t)` pairs from the same clip; slots at the two frames
  are matched via nearest-neighbor cosine so that per-index temporal diff
  reflects content change, not permutation drift.
- **Flat layout** *(legacy)*: `data-dir/*.png`. Loader falls back to using the
  previous training-iteration's slots as `slots_prev`. Because iterations see
  unrelated shuffled frames, this signal is semantically weak. A warning is
  printed. Use only for smoke-testing.

**No real data yet? Generate synthetic clips:**
```bash
python scripts/make_synthetic_data.py --out data/frames/synthetic --n-clips 1250 --clip-len 4
```
1250 clips × 4 frames = 5000 total frames in ~90s. Each clip has a fixed
background + rectangle (should be routed slow) and 2-5 moving disks (should
be routed fast) — a controlled ground truth for L_slow.

Flat mode is still available for compatibility:
```bash
python scripts/make_synthetic_data.py --out data/frames/synthetic_flat --flat
```

### 4.2 Run

```bash
python scripts/pretrain_stage1.py \
    --config configs/default.yaml \
    --data-dir data/frames \
    --num-workers 4 \
    --wandb \
    --run-name stage1_v1
```

Runtime: ~2 hours on any modern discrete GPU (RTX 4090 / 5080 / A5000 / L40)
at batch 256, 100 K steps. Run multiple seeds in parallel (one per GPU) if
you want variance bounds.

### 4.3 What to watch during training

Track in W&B:

- `L_slot` should drop from ~5.0 to <0.5 in the first 10 K steps.
- `L_slow` should decrease steadily after step ~5 K.
- `L_route` should stay in [0.001, 0.05] (already close to prior).
- `gumbel_tau` anneals 1.0 → 0.3 over ~10 K steps.
- **Slot alpha visualizations** (dumped every 10 K steps to `outputs/stage1/`)
  should show visually distinct entity groupings by step ~30 K.

Common failure signals:

| Symptom | Diagnosis | Fix |
|---|---|---|
| `L_slot` plateaus > 2.0 | slot collapse | raise `lambda_div` to 0.2 |
| Router locks to one class within first ~500 steps | you are on the pre-fix `slow_temporal_loss` (mask-weighted variance sum, has a trivial all-fast minimum) | ``git pull`` — router-supervision CE fixes this |
| `NaN` in any loss | learning rate spike | lower `stage1_lr` to 1e-4, longer warmup |
| `slow_ratio` stuck at 0 or 1 after ~2 K steps | `lambda_slow` too weak relative to `lambda_slot` | raise `lambda_slow` to 1.0 |

---

## 5. What Phase 1 does NOT include (yet)

Phase 1 gives you a fully working Stage-1 pretraining pipeline and the entire
encoder used at Stage-2. The following components are **Phase 2** and depend
on choices you still need to make:

- **TD-MPC2 fork with our encoder plugged in** (Stage-2 online RL). Need to
  fork `github.com/nicklashansen/tdmpc2` and replace `WorldModel.encoder`
  with `HippoActEncoder`.
- **Env wrappers** for DMC + Distracting Suite (chosen as our first sim).
  Easy — DMC's OpenGL rendering is well-supported.
- **Real robot deployment code** — depends on which arm you're using. Skeleton
  will target ROS 2 + a `RobotAbstract` interface, so swapping Franka /
  UR5 / xArm is a driver change only.

Ping me once you've got Stage 1 pretraining running and know your arm model,
and I'll cut Phase 2.

---

## 6. Troubleshooting quick table

| Problem | Command / check |
|---|---|
| `ImportError: hippoact` | `pip install -e .` was skipped |
| `RuntimeError: input_size != patch grid` | image_size must be multiple of 14 |
| DINOv2 download fails | pre-download on a machine with internet, then rsync `~/.cache/torch/hub/` |
| CUDA OOM at batch 256 | drop to 128, `L_slot` will still converge |
| Loss all zeros | `HIPPOACT_FORCE_MOCK=1` accidentally set in shell |
| Tests fail on `test_binding_mask_effectively_hides_slow_slots` | The mask/attention wiring is broken — don't ignore, this is what makes P3 sound |

---

## 7. Repro of paper claims

Every claim in the paper maps to a specific test / experiment:

| Paper claim | Code entry point | Expected artifact |
|---|---|---|
| P1: DINOv2 + Slot Attention decompose scenes | `scripts/pretrain_stage1.py` + slot alpha viz | `outputs/stage1/slots_step*.png` |
| P2: Binding produces actionable c_t | (Phase 2, needs env) | Linear probe R² > 0.7 |
| P3: Slot swap improves sim-to-real | (Phase 2, needs real robot) | Table VI |
| P4: Slot residual detects OOD | `hippoact.safety.SafetyGate` | Table VIII |

---

## License

TBD (research code, not for redistribution yet).
