#!/usr/bin/env python3
"""Proxy segmentation of the walker from clean DMC frames.

There is no per-object ground truth in DMC as shipped, but on the *clean*
background the walker is the only moving thing, so a frame difference followed
by a morphological close recovers it. Only valid on clean -- on `easy` the DAVIS
video dominates the difference (measured: 47.9% of pixels change per frame on
easy vs 12.7% on clean), so the same construction there would segment the
background, not the walker.

Lesson carried over from the synthetic phase: a mask inferred from consecutive
differences covers the union of the old and new pose. Using three frames and
intersecting, obj_t = motion(t-1,t) & motion(t,t+1), isolates the pose at t.
The closing then fills the limb interiors that frame differencing leaves hollow.
"""
from __future__ import annotations

import numpy as np
import scipy.ndimage as ndi


def walker_mask(prev, cur, nxt, thresh: float = 0.02, close_iter: int = 3,
                min_size: int = 200):
    """prev/cur/nxt: (H,W,3) float in [0,1]. Return bool (H,W) mask at `cur`."""
    d0 = np.abs(cur - prev).max(-1) > thresh
    d1 = np.abs(nxt - cur).max(-1) > thresh
    m = d0 & d1                                    # pose at `cur` only
    m = ndi.binary_closing(m, structure=np.ones((3, 3)), iterations=close_iter)
    m = ndi.binary_fill_holes(m)
    lbl, n = ndi.label(m)
    if n == 0:
        return m
    sizes = ndi.sum(m, lbl, index=range(1, n + 1))
    keep = {i + 1 for i, s in enumerate(sizes) if s >= min_size}
    return np.isin(lbl, list(keep)) if keep else np.zeros_like(m)
