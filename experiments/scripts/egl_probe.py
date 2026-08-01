#!/usr/bin/env python3
"""Find which physical GPU each MUJOCO_EGL_DEVICE_ID actually renders on.

EGL enumerates devices in its own order, which on this box does NOT match the
CUDA/nvidia-smi order — setting MUJOCO_EGL_DEVICE_ID=<cuda id> silently put the
render context on a different card.  This probe renders in a loop for a few
seconds so `nvidia-smi` can be sampled to see which GPU picked up the context.

Usage:  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=<n> python egl_probe.py [seconds]
"""
import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402
from dm_control import suite  # noqa: E402

seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 8.0
print(f"pid={os.getpid()} MUJOCO_EGL_DEVICE_ID={os.environ.get('MUJOCO_EGL_DEVICE_ID')} "
      f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)

env = suite.load("walker", "walk", task_kwargs={"random": 0})
env.reset()
spec = env.action_spec()
t0 = time.time()
n = 0
while time.time() - t0 < seconds:
    env.physics.render(height=64, width=64, camera_id=0)
    env.step(np.zeros(spec.shape, dtype=spec.dtype))
    n += 1
print(f"rendered {n} frames in {time.time()-t0:.1f}s", flush=True)
