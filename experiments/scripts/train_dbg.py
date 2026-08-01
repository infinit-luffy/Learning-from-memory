#!/usr/bin/env python3
"""Launcher for tdmpc2's train.py that stays introspectable during a multi-day run.

Two additions, neither of which touches TD-MPC2 code:

* ``SIGUSR1`` dumps all thread stacks to stderr (``kill -USR1 <pid>``).  The
  server has ``kernel.yama.ptrace_scope=1`` and no passwordless sudo, so py-spy
  cannot attach to an already-running job — this is the only way to see where a
  stalled run actually is.
* ``prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY)`` so py-spy *can* attach after all.

Usage: same argv as train.py, e.g.
    python experiments/scripts/train_dbg.py task=walker-walk obs=rgb ...
"""
from __future__ import annotations

import ctypes
import faulthandler
import os
import runpy
import signal
import sys
from pathlib import Path

PR_SET_PTRACER = 0x59616d61
PR_SET_PTRACER_ANY = -1

TRAIN_PY = (Path(__file__).resolve().parents[2]
            / "third_party" / "tdmpc2" / "tdmpc2" / "train.py")


def _allow_ptrace():
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(PR_SET_PTRACER, ctypes.c_ulong(PR_SET_PTRACER_ANY), 0, 0, 0)
    except Exception as e:                      # non-Linux / no libc: not fatal
        print(f"[train_dbg] PR_SET_PTRACER failed: {e}", file=sys.stderr)


def main():
    _allow_ptrace()
    faulthandler.enable()
    faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)
    print(f"[train_dbg] pid={os.getpid()}  SIGUSR1 -> stack dump", flush=True)

    # train.py resolves `from common...` against its own directory, and hydra
    # resolves config_path against the __main__ file's location.
    sys.path.insert(0, str(TRAIN_PY.parent))
    sys.argv[0] = str(TRAIN_PY)
    runpy.run_path(str(TRAIN_PY), run_name="__main__")


if __name__ == "__main__":
    main()
