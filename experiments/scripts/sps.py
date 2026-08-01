#!/usr/bin/env python3
"""Report throughput / progress / ETA for the W1.2 runs from their console logs.

Parses TD-MPC2's console lines (``train  E: .. I: <step> R: .. S: .. T: h:mm:ss``)
and reports steady-state steps/second plus a projected finish time.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parents[1] / "logs" / "console"
LINE = re.compile(
    r"(train|eval)\s+.*?E:\s*([\d,]+).*?I:\s*([\d,]+).*?R:\s*(-?[\d.]+).*?T:\s*(\d+):(\d+):(\d+)"
)
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def parse(path):
    rows = []
    for raw in path.read_text(errors="ignore").splitlines():
        m = LINE.search(ANSI.sub("", raw))
        if m:
            cat, ep, step, rew, h, mi, s = m.groups()
            rows.append(dict(cat=cat, step=int(step.replace(",", "")),
                             reward=float(rew),
                             t=int(h) * 3600 + int(mi) * 60 + int(s)))
    return rows


def main():
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 500_000
    paths = sorted(LOG_DIR.glob("*.log"))
    if not paths:
        print("no logs yet")
        return
    print(f"{'run':<40} {'step':>8} {'SPS':>7} {'elapsed':>9} {'ETA(h)':>7}  last evals")
    print("-" * 118)
    for p in paths:
        rows = parse(p)
        if not rows:
            print(f"{p.stem:<40} {'--':>8} (no metrics yet)")
            continue
        last = rows[-1]
        # steady-state SPS: measured over the tail, past compile + seed pretraining
        tail = [r for r in rows if r["step"] >= min(20_000, last["step"] // 2)]
        if len(tail) >= 2 and tail[-1]["t"] > tail[0]["t"]:
            sps = (tail[-1]["step"] - tail[0]["step"]) / (tail[-1]["t"] - tail[0]["t"])
        else:
            sps = last["step"] / max(last["t"], 1)
        eta = (target - last["step"]) / sps / 3600 if sps > 0 else float("nan")
        evals = [r for r in rows if r["cat"] == "eval"][-4:]
        ev = "  ".join(f"{r['step']//1000}K:{r['reward']:.0f}" for r in evals)
        print(f"{p.stem:<40} {last['step']:>8,} {sps:>7.1f} "
              f"{last['t']/3600:>8.2f}h {eta:>7.1f}  {ev}")


if __name__ == "__main__":
    main()
