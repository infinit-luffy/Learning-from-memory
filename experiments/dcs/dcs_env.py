"""Distracting Control Suite environment for TD-MPC2 (W1.2 / Q2).

Task naming::

    dcs-<difficulty>-<domain>-<task>       e.g. dcs-none-walker-walk
                                                dcs-easy-walker-walk
                                                dcs-hard-cheetah-run

Design constraint (PHASE2_PLAN: "TD-MPC2 本体不改一行算法"): everything
downstream of environment construction — action scaling, ``action_repeat=2``,
the 3x64x64 frame stack, the 500-step timeout — is TD-MPC2's own
``envs/dmcontrol.py`` code, imported and reused verbatim.  The *only*
difference between ``dcs-none-walker-walk`` and the official
``task=walker-walk obs=rgb`` run is that this module goes through
``distracting_control`` when ``difficulty != none``.

``difficulty=none`` deliberately bypasses ``distracting_control`` entirely and
calls ``dm_control.suite.load``, so the clean arm is bit-for-bit the official
pixel baseline.  That is what makes the W1.2 "对齐官方" check meaningful.

Distraction scope: ``background`` only (dynamic DAVIS video), matching W1.1.
Camera shake and colour distraction are out of scope for Q2 (背景鲁棒性).

Requires ``MUJOCO_GL=egl`` on a headless machine.
"""
from __future__ import annotations

import os

import numpy as np

DIFFICULTIES = ("none", "easy", "medium", "hard")

DEFAULT_DAVIS_PATH = "/usr1/home/s125mdg56_03/datasets/davis/DAVIS/JPEGImages/480p"


def davis_path() -> str:
    """Directory that *directly contains* the DAVIS video sequence folders.

    ``distracting_control`` does ``os.path.join(path, video_name)``, so this
    must be the ``JPEGImages/480p`` level, not the dataset root
    (PHASE2_PLAN §2 坑 5).
    """
    path = os.environ.get("DAVIS_PATH", DEFAULT_DAVIS_PATH)
    if not os.path.isdir(os.path.join(path, "bear")):
        raise FileNotFoundError(
            f"DAVIS sequences not found under {path} (expected sequence dirs "
            f"such as 'bear/' directly inside). Set $DAVIS_PATH to the "
            f"DAVIS/JPEGImages/480p directory."
        )
    return path


def parse_task(task: str):
    """``dcs-easy-walker-walk`` -> ``('easy', 'walker', 'walk')``."""
    parts = task.split("-")
    if len(parts) < 4 or parts[0] != "dcs":
        raise ValueError(f"Not a DCS task: {task}")
    difficulty = parts[1]
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"Unknown DCS difficulty {difficulty!r} in {task}")
    domain = parts[2]
    subtask = "_".join(parts[3:])
    domain = dict(cup="ball_in_cup", pointmass="point_mass").get(domain, domain)
    return difficulty, domain, subtask


def make_dm_env(difficulty, domain, subtask, seed, dynamic=True):
    """Build the raw dm_env, with or without distractions."""
    if difficulty == "none":
        from dm_control import suite

        return suite.load(
            domain, subtask,
            task_kwargs={"random": seed},
            visualize_reward=False,
        )

    from distracting_control import suite as dcs_suite

    # difficulty selects how many DAVIS videos the background is drawn from
    # (easy=4, medium=8, hard=all) via suite_utils.DIFFICULTY_NUM_VIDEOS.
    # `distraction_seed` seeds the wrapper's own RandomState (video choice,
    # start frame, play direction) so a run is reproducible from cfg.seed.
    np.random.seed(seed)
    return dcs_suite.load(
        domain, subtask,
        difficulty=difficulty,
        background_dataset_path=davis_path(),
        background_dataset_videos="train",
        distraction_types=("background",),
        dynamic=bool(dynamic),
        distraction_seed=seed,
        task_kwargs={"random": seed},
        visualize_reward=False,
        from_pixels=False,
        pixels_only=False,
    )


def make_env(cfg):
    """TD-MPC2 entry point. Raises ValueError for non-DCS tasks.

    `cfg.obs`:
      state    — dm_control's proprio vector (TD-MPC2 default)
      rgb      — 3x64x64 frame stack (the official pixel baseline)
      hippoact — {"rgb": 3x224x224 uint8, "state": proprio} for the HippoAct
                 encoder; see experiments/dcs/hippoact_obs.py
    """
    # Imported lazily/relatively: these live on TD-MPC2's sys.path, which is
    # only set up once tdmpc2/train.py is the running program.
    from dm_control.suite.wrappers import action_scale
    from envs.dmcontrol import DMControlWrapper, Pixels
    from envs.wrappers.timeout import Timeout

    difficulty, domain, subtask = parse_task(cfg.task)
    assert cfg.obs in {"state", "rgb", "hippoact"}, \
        f"DCS supports obs in {{state, rgb, hippoact}}, got {cfg.obs!r}"

    env = make_dm_env(difficulty, domain, subtask, cfg.seed)
    env = action_scale.Wrapper(env, minimum=-1.0, maximum=1.0)
    env = DMControlWrapper(env, domain)
    if cfg.obs == "rgb":
        env = Pixels(env, cfg)
    elif cfg.obs == "hippoact":
        from experiments.dcs.hippoact_obs import HippoActObs
        env = HippoActObs(env, size=int(cfg.get("hippoact_image_size", 224)))
    elif cfg.obs == "state" and cfg.get("hippoact_precompute", ""):
        # E2E-0 fast path: apply the frozen HippoAct encoder here, so TD-MPC2
        # sees a plain state vector [flatten(fast_slots) ⊕ q_t] and runs
        # completely unmodified. See experiments/dcs/hippoact_slots.py.
        from experiments.dcs.hippoact_slots import HippoActSlots
        env = HippoActSlots(env, str(cfg.hippoact_precompute),
                            size=int(cfg.get("hippoact_image_size", 224)),
                            slot_init_seed=int(cfg.get("hippoact_slot_init_seed", 0)))
    env = Timeout(env, max_episode_steps=500)
    return env
