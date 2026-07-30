#!/usr/bin/env python3
"""Probe SAVi-style slot carryover WITHOUT changing repo code.

slots_init already accepts an arbitrary tensor, so we can feed the previous
frame's OUTPUT slots as the current frame's init and measure whether slots
then track their object.

Three regimes compared, all on the same checkpoint and data:
  A. independent init  (pre-c5ee180 behaviour)
  B. shared init       (current default)
  C. carryover init    (prev frame's output slots as current init) -- SAVi-like

Metrics per regime:
  - centroid shift of each slot between the two frames (should track object
    motion, ~2.3 patch on the new data, if slots follow their object)
  - init-noise floor (same image encoded twice)
  - corr(content diff, motion score) and corr(centroid shift, motion score)
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
    an = a / (a.sum(-1, keepdim=True) + 1e-8)
    ys, xs = torch.meshgrid(
        torch.arange(g, device=a.device, dtype=torch.float),
        torch.arange(g, device=a.device, dtype=torch.float), indexing="ij")
    return torch.stack([(an * ys.reshape(-1)).sum(-1),
                        (an * xs.reshape(-1)).sum(-1)], dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--n-clips", type=int, default=150)
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

    res = {k: {"cs": [], "cd": [], "noise": [], "ms": []} for k in "ABC"}

    for c in sorted(glob.glob(args.data_dir + "/clip_*"))[: args.n_clips]:
        fs = sorted(glob.glob(c + "/*.png"))
        prev = base(Image.open(fs[0]).convert("RGB"))
        cur = base(Image.open(fs[1]).convert("RGB"))
        motion = (cur - prev).abs().max(0).values[None, None]
        xp, xc = norm(prev)[None].cuda(), norm(cur)[None].cuda()

        with torch.no_grad():
            fp, fc = enc.dino(xp), enc.dino(xc)
            init = SA.sample_init(1, device=fc.device, dtype=fc.dtype)

            runs = {}
            # A: independent init
            sp = SA(fp)
            sc = SA(fc)
            runs["A"] = (sp, sc, SA(fc))
            # B: shared init
            sp = SA(fp, slots_init=init)
            sc = SA(fc, slots_init=init)
            runs["B"] = (sp, sc, SA(fc, slots_init=init))
            # C: carryover -- prev output slots become current init
            sp = SA(fp, slots_init=init)
            sc = SA(fc, slots_init=sp)
            runs["C"] = (sp, sc, SA(fc, slots_init=sp))

            for key, (s_prev, s_cur, s_rep) in runs.items():
                if key == "C":
                    prev_aligned = s_prev            # identity by construction
                else:
                    prev_aligned = match_slots_nn(s_cur, s_prev)
                _, a_cur = enc.slot_decoder(s_cur)
                _, a_prev = enc.slot_decoder(prev_aligned)
                K, N = a_cur.shape[1], a_cur.shape[2]
                g = int(round(N ** 0.5))
                cs = (centroids(a_cur[0], g) - centroids(a_prev[0], g)).norm(dim=-1)
                cd = (s_cur - prev_aligned).pow(2).sum(-1)[0]
                noise = (s_cur - s_rep).pow(2).sum(-1).mean().item()
                m = F.adaptive_avg_pool2d(motion, (g, g)).reshape(1, N).cuda()
                m = m / (m.sum() + 1e-8)
                an = a_cur[0] / (a_cur[0].sum(-1, keepdim=True) + 1e-8)
                ms = (an * m).sum(-1) * N
                res[key]["cs"] += cs.tolist()
                res[key]["cd"] += cd.tolist()
                res[key]["ms"] += ms.tolist()
                res[key]["noise"].append(noise)

    names = {"A": "A 独立随机 init", "B": "B 共享 init (现默认)", "C": "C carryover (SAVi 式)"}
    print("物体真实位移 ≈ 2.28 patch (新数据实测)")
    print()
    hdr = (f"{'regime':<24}{'centroid shift':>15}{'噪声底':>12}"
           f"{'corr(cd,ms)':>14}{'corr(cs,ms)':>14}")
    print(hdr)
    print("-" * len(hdr))
    for k in "ABC":
        cs = np.array(res[k]["cs"]); cd = np.array(res[k]["cd"])
        ms = np.array(res[k]["ms"]); nz = np.mean(res[k]["noise"])
        r_cd = np.corrcoef(np.log(cd + 1e-9), np.log(ms + 1e-9))[0, 1]
        r_cs = np.corrcoef(np.log(cs + 1e-9), np.log(ms + 1e-9))[0, 1]
        print(f"{names[k]:<24}{np.median(cs):>15.3f}{nz:>12.4f}"
              f"{r_cd:>14.3f}{r_cs:>14.3f}")


if __name__ == "__main__":
    main()
