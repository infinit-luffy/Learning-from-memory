"""DCS support for DrQ-v2 (W1.3), sharing the TD-MPC2 arm's distraction settings.

Design constraint, same as the TD-MPC2 side: DrQ-v2's algorithm and per-task
hyperparameters are untouched.  The distraction is selected by a new
``distraction: none|easy|medium|hard`` config field rather than by a new task
name, so `cfgs/task/walker_walk.yaml` (nstep, batch_size, stddev_schedule,
num_train_frames) applies unchanged to both the clean and the distracted arm.

The raw dm_env is built by ``dcs_env.make_dm_env`` — the *same* function the
TD-MPC2 arm uses — so both baselines see identical backgrounds (DAVIS `train`
split, background distraction only, ``dynamic=True``, seeded by run seed).
Anything below that (84x84 render, 3-frame stack, action repeat, action
scaling) stays DrQ-v2's own code.

Note the deliberate resolution difference: DrQ-v2 renders 84x84, TD-MPC2 64x64.
Each is its published configuration; disclosed in the paper rather than
equalised.
"""
from __future__ import annotations

from .dcs_env import make_dm_env

DIFFICULTIES = ("none", "easy", "medium", "hard")


def load(name: str, seed: int, distraction: str = "none"):
    """`name` is DrQ-v2's `<domain>_<task>`, e.g. `walker_walk`.

    Returns the raw dm_env with the distraction wrapper already applied
    (or a plain dm_control env when ``distraction='none'``).
    """
    if distraction not in DIFFICULTIES:
        raise ValueError(
            f"unknown distraction {distraction!r}, expected one of {DIFFICULTIES}")
    domain, task = name.split("_", 1)
    domain = dict(cup="ball_in_cup", pointmass="point_mass").get(domain, domain)
    return make_dm_env(distraction, domain, task, seed)
