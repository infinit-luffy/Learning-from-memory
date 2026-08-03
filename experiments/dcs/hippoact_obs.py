"""Observation wrapper for the HippoAct encoder (W2.1 E2E-0).

TD-MPC2's pixel baseline sees a 3-frame stack of 64x64 RGB. The HippoAct
encoder instead needs

    {"rgb":   (3, 224, 224) uint8,     # DINOv2 ViT-S/14 input resolution
     "state": (d_q,)        float32}   # dm_control proprio (walker: 24)

Three decisions worth stating, because each has a cost:

**uint8, not float.** The replay buffer stores raw observations, and TD-MPC2
sizes it at `min(buffer_size, steps)`. At 500K capacity a 224x224x3 frame costs
147 KB as uint8 and 588 KB as float32 -- 75 GB vs 301 GB of host RAM. uint8 is
the only workable option, so ImageNet normalisation happens inside the adapter,
on GPU, exactly as TD-MPC2's own `PixelPreprocess` does for its pixel path.

**Single frame, no stack.** The E2E-0 contract is z = MLP(flatten(S_fg) + q_t):
one frame plus proprio, no temporal window (that is E2E-1's binding
transformer). A 3-stack would triple the buffer to 225 GB for no benefit at
this level.

**`state` is dm_control's own observation vector**, i.e. what TD-MPC2's
`DMControlWrapper` already concatenates (walker-walk: orientations 14 + height 1
+ velocity 9 = 24), not raw qpos/qvel. It is the proprio channel the baseline
would have had access to, so the comparison stays about the visual encoder.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from tensordict import TensorDict

HIPPOACT_IMAGE_SIZE = 224


class HippoActObs(gym.Wrapper):
    """DMControlWrapper -> {"rgb": uint8 (3,S,S), "state": float32 (d_q,)}."""

    def __init__(self, env, size: int = HIPPOACT_IMAGE_SIZE):
        super().__init__(env)
        self.env = env
        self._size = size
        state_space = env.observation_space
        self.observation_space = gym.spaces.Dict({
            "rgb": gym.spaces.Box(low=0, high=255, shape=(3, size, size), dtype=np.uint8),
            "state": state_space,
        })

    def _obs(self, state):
        """Returns a TensorDict, not a plain dict.

        TD-MPC2's single-task path never exercised dict observations: `act()`
        calls `obs.to(device).unsqueeze(0)` and `update()` indexes a leading
        time axis. A plain dict has none of that; a TensorDict has all of it,
        and the replay buffer is TensorDict-based already.
        """
        frame = self.env.render(width=self._size, height=self._size)   # (S,S,3) uint8
        return TensorDict({
            "rgb": torch.from_numpy(np.ascontiguousarray(frame.transpose(2, 0, 1))),
            "state": (state if torch.is_tensor(state)
                      else torch.from_numpy(state)).float(),
        }, batch_size=())

    def reset(self):
        return self._obs(self.env.reset())

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        return self._obs(state), reward, done, info
