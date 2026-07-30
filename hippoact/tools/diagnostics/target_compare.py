#!/usr/bin/env python3
"""Which candidate target actually predicts 'this slot covers moving content'?

Compares, per slot, against the ground-truth motion score:
  (a) content diff      = ||slots_t - slots_prev||^2         <- current target
  (b) centroid shift    = ||centroid(alpha_t) - centroid(alpha_prev)||  <- proposed
  (c) alpha L1 change   = ||alpha_t - alpha_prev||_1         <- cheap alternative

Run before writing the patch: if (b) does not beat (a) here, it will not beat
it in training either.
"""
from __future__ import annotations

import argparse
import glob

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from hippoact.utils.config import load_config
from hippoact.utils.slot_matching import match_slots_nn
from validate_semantics import build_encoder


def centroids(a, g):
    """a: (K,N) alpha. Return (K,2) centroid in patch coords."""
    an = a / (a.sum(-1, keepdim=True) + 1e-8)
    ys, xs = torch.meshgrid(
        torch.arange(g, device=a.device, dtype=torch.float),
        torch.arange(g, device=a.device, dtype=torch.float), indexing="ij")
    ys, xs = ys.reshape(-1), xs.reshape(-1)
    return torch.stack([(an * ys).sum(-1), (an * xs).sum(-1)], dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--n-clips", type=int, default=150)
    ap.add_argument("--gap", type=int, default=1, help="frame gap within clip")
    args = ap.parse_args()

    cfg = load_config(args.config)
    size = cfg.encoder.image_size
    enc = build_encoder(cfg).cuda()
    ck = torch.load(args.ckpt, map_location="cuda", weights_only=False)
    enc.load_state_dict(ck["encoder"], strict=False)
    enc.eval()
    SA = enc.slot_attn

    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    base = transforms.Compose([
        transforms.Resize(size), transforms.CenterCrop(size), transforms.ToTensor(),
    ])

    D, C, A, M = [], [], [], []
    for c in sorted(glob.glob(args.data_dir + "/clip_*"))[: args.n_clips]:
        fs = sorted(glob.glob(c + "/*.png"))
        if len(fs) <= args.gap:
            continue
        prev = base(Image.open(fs[0]).convert("RGB"))
        cur = base(Image.open(fs[args.gap]).convert("RGB"))
        motion = (cur - prev).abs().max(0).values[None, None]
        xp, xc = norm(prev)[None].cuda(), norm(cur)[None].cuda()
        with torch.no_grad():
            fp, fc = enc.dino(xp), enc.dino(xc)
            init = SA.sample_init(1, device=fc.device, dtype=fc.dtype)
            sp, sc_ = SA(fp, slots_init=init), SA(fc, slots_init=init)
            spm = match_slots_nn(sc_, sp)
            _r1, alpha_t = enc.slot_decoder(sc_)
            _r2, alpha_p = enc.slot_decoder(spm)
        at, apv = alpha_t[0], alpha_p[0]
        K, N = at.shape
        g = int(round(N ** 0.5))

        D += (sc_ - spm).pow(2).sum(-1)[0].tolist()
        C += (centroids(at, g) - centroids(apv, g)).norm(dim=-1).tolist()
        A += (at - apv).abs().sum(-1).tolist()

        m = F.adaptive_avg_pool2d(motion, (g, g)).reshape(1, N).cuda()
        m = m / (m.sum() + 1e-8)
        an = at / (at.sum(-1, keepdim=True) + 1e-8)
        M += ((an * m).sum(-1) * N).tolist()

    D, C, A, M = map(np.array, (D, C, A, M))

    def rep(name, v):
        pear = np.corrcoef(np.log(v + 1e-9), np.log(M + 1e-9))[0, 1]
        rv = v.argsort().argsort().astype(float)
        rm = M.argsort().argsort().astype(float)
        spear = np.corrcoef(rv, rm)[0, 1]
        # if we thresholded this signal at its top 10%, how motion-y are those slots?
        hi = v >= np.quantile(v, 0.9)
        auc_top = M[hi].mean() / max(M.mean(), 1e-9)
        print(f"{name:<28} pearson(log) {pear:+.3f}   spearman {spear:+.3f}   "
              f"top10%的 motion score / 全体均值 = {auc_top:.2f}x")

    print(f"gap={args.gap}  slots: {len(M)}   motion score mean {M.mean():.3f} "
          f"median {np.median(M):.3f}")
    print(f"centroid shift 幅度 (patch 单位): median {np.median(C):.3f} "
          f"p90 {np.quantile(C, 0.9):.3f}   [<1.0 表示亚 patch, 被网格量化]")
    print()
    print("=== 各候选 target 对『覆盖运动区域』的预测力 ===")
    rep("(a) content diff  [现行]", D)
    rep("(b) centroid shift [你提议]", C)
    rep("(c) alpha L1 change", A)


if __name__ == "__main__":
    main()
