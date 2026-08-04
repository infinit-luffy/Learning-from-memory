"""E2E-0 observation: the frozen HippoAct encoder, applied in the environment.

    obs = [ flatten(fast_slots) ⊕ q_t ]        (16*128 + 24 = 2072 floats)

which is exactly the E2E-0 contract's input. TD-MPC2 then runs with
`obs=state` and **no modification at all** — its own state encoder
(`mlp(..., act=SimNorm)`) is the trainable MLP of `z = MLP(flatten(S_fg) ⊕ q)`.

Why this rather than encoding inside the world model
----------------------------------------------------
Identical function, ~180x cheaper, and it removes an architecture deviation:

* **Identical.** Stage-1 is frozen at E2E-0, so a frame's slots never change.
  Encoding at env-step time and encoding at sample time are the same map; the
  shared `SlotFeatureExtractor` is literally the same class both ways.
* **Cheaper.** TD-MPC2 runs one update per env step over (horizon+1) x batch =
  1024 frames. Encoding inside the model re-runs DINOv2 on all 1024 every step
  (measured 3882 ms/step -> 22.6 days for 500K). Here DINOv2 runs once per env
  step. The buffer also drops from 147 KB/frame to 8.3 KB (75 GB -> 4 GB).
* **Architecturally faithful.** Both of TD-MPC2's own encoders end in SimNorm,
  and its dynamics/reward/Q are built on that simplicial latent. A hand-written
  z_mlp without SimNorm feeds the model a latent it was never designed for.
  Using the stock state encoder keeps that exactly right.

The in-model path (`encoder_type=hippoact`) is still needed when the encoder
joins the training graph — i.e. E2E-1/E2E-2 if Stage-1 is unfrozen. Keeping
Stage-1 frozen across all three levels keeps this fast path valid *and* keeps
the build-up ablation single-variable (each level adds a module rather than
adding a module and unfreezing the encoder at the same time).
"""
from __future__ import annotations

from collections import deque

import gymnasium as gym
import numpy as np
import torch

HIPPOACT_IMAGE_SIZE = 224


class HippoActSlots(gym.Wrapper):
    """DMControlWrapper -> flat [fast_slots ⊕ proprio] float32 vector."""

    def __init__(self, env, stage1_ckpt: str, device: str = "cuda",
                 size: int = HIPPOACT_IMAGE_SIZE, slot_init_seed: int = 0,
                 include_proprio: bool = True, num_frames: int = 1,
                 emit_mask: bool = False):
        super().__init__(env)
        import sys
        from pathlib import Path
        root = Path(__file__).resolve().parents[2]
        for p in (str(root / "hippoact"), str(root)):
            if p not in sys.path:
                sys.path.insert(0, p)
        from hippoact.adapters.slot_features import SlotFeatureExtractor

        self.env = env
        self._size = size
        self._device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.extractor = SlotFeatureExtractor(
            stage1_ckpt, slot_init_seed=slot_init_seed).to(self._device).eval()
        # Print it: the init draw is part of the encoder's identity (seed 0 vs 1
        # moves the features by max|Δ| = 16.2), so every run's log must say
        # which one it used, not just which checkpoint.
        print(f"[HippoActSlots] ckpt={stage1_ckpt} slot_init_seed={slot_init_seed} "
              f"feature_dim={self.extractor.feature_dim} "
              f"include_proprio={include_proprio}", flush=True)

        # walker-walk is solvable from proprio alone (TD-MPC2 state obs reaches
        # 979), so including it makes the visual representation unmeasurable —
        # E2E-0 then tracks the proprio-only baseline exactly. `include_proprio
        # = False` gives the vision-only control this comparison needs.
        self._include_proprio = include_proprio
        # E2E-1: a single frame is not a Markov state for locomotion — the
        # pixel baseline is `Pixels(env, cfg, num_frames=3)` and the state obs
        # carries 9 velocity dims, while E2E-0 vision-only had neither
        # (TODO §R4.10.4). Stacking here matches the pixel baseline's window.
        self._num_frames = int(num_frames)
        # E2E-1 also needs the routing decision itself, so the binding
        # transformer can *mask* slow slots out of attention rather than see
        # them as zero tokens. E2E-0 only ever needed the gated vector.
        self._emit_mask = bool(emit_mask)
        self._frames = deque(maxlen=self._num_frames)

        K = self.extractor.num_slots
        proprio_dim = int(env.observation_space.shape[0]) if include_proprio else 0
        per_frame = self.extractor.feature_dim + (K if emit_mask else 0)
        self._dim = self._num_frames * per_frame + proprio_dim
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._dim,), dtype=np.float32)

    def _encode(self) -> torch.Tensor:
        """One frame -> feature vector (slots [+ fast mask])."""
        frame = self.env.render(width=self._size, height=self._size)   # (S,S,3) uint8
        rgb = torch.from_numpy(np.ascontiguousarray(frame.transpose(2, 0, 1)))
        x = rgb.unsqueeze(0).to(self._device)
        if not self._emit_mask:
            return self.extractor(x).squeeze(0).cpu()
        slots, mask = self.extractor.slots_and_mask(x)
        return torch.cat([slots.squeeze(0).flatten().cpu(),
                          mask.squeeze(0).float().cpu()])

    def _obs(self, state, is_reset=False) -> torch.Tensor:
        feat = self._encode()
        # On reset the window is filled with the first frame, exactly as
        # TD-MPC2's own `Pixels` wrapper does.
        for _ in range(self._frames.maxlen if is_reset else 1):
            self._frames.append(feat)
        out = torch.cat(list(self._frames))
        if not self._include_proprio:
            return out
        state = state if torch.is_tensor(state) else torch.from_numpy(state)
        return torch.cat([out, state.float()])

    def reset(self):
        return self._obs(self.env.reset(), is_reset=True)

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        return self._obs(state), reward, done, info
