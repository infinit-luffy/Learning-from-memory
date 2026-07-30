#!/usr/bin/env python3
"""Make the pip-installed distracting_control work with modern MuJoCo.

Run once after `pip install distracting-control`. Idempotent.

Background: distracting_control writes the DAVIS frame into the skybox texture
via ``physics.model.tex_rgb``. MuJoCo renamed that buffer to ``tex_data`` when
textures gained a variable channel count (``tex_nchannel``). Verified on the
walker skybox that the layout is unchanged for RGB:

    tex[0] type=2 (skybox) 800x4800 nchannel=3 adr=0   ->  tex_data[adr : adr+3*H*W]

so this is a pure rename, not a reindex. tex_adr / tex_width / tex_height are
untouched by the rename.

Why patch rather than pin MuJoCo: dm_control 1.0.14 (the version PHASE2_PLAN
pins) is itself incompatible with several MuJoCo releases -- it needs
MjModel.bvh_geomid, absent in 3.0.0 / 3.0.1 / 3.1.6 -- so there is no single
MuJoCo version satisfying both the pinned dm_control and distracting_control.
Upgrading dm_control and patching this one rename is the combination that works.
"""
from __future__ import annotations

import pathlib
import sys

import distracting_control


def main() -> int:
    p = pathlib.Path(distracting_control.__file__).parent / "background.py"
    src = p.read_text()
    if "tex_rgb" not in src:
        print(f"already patched (no tex_rgb in {p})")
        return 0
    n = src.count("tex_rgb")
    p.write_text(src.replace("tex_rgb", "tex_data"))
    print(f"patched {p}: {n} occurrence(s) tex_rgb -> tex_data")
    return 0


if __name__ == "__main__":
    sys.exit(main())
