#!/usr/bin/env python3
"""Cross-frame binding check for one checkpoint, replicating the training-time
slot_init_mode exactly.

Primary criterion (user-defined):
    alpha centroid shift median  vs  true object displacement (~2.28 patch)
      ~2.28      -> binding works
      1.0-1.8    -> partial
      <1.0       -> vanilla Slot Attention cannot bind at this scale

Also reports whether the NN matching applied in the trainer is the identity
permutation under carryover (it should be, by construction — if it is not,
matching is actively scrambling an already-aligned pairing).
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
    ap.add_argument("--n-clips", type=int, default=120)
    ap.add_argument("--mode", default=None, help="override slot_init_mode")
    args = ap.parse_args()

    cfg = load_config(args.config)
    mode = args.mode or str(cfg.train.get("slot_init_mode", "shared"))
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

    CS, CD, MS, ident, obj_disp = [], [], [], [], []

    for c in sorted(glob.glob(args.data_dir + "/clip_*"))[: args.n_clips]:
        fs = sorted(glob.glob(c + "/*.png"))
        prev = base(Image.open(fs[0]).convert("RGB"))
        cur = base(Image.open(fs[1]).convert("RGB"))
        motion = (cur - prev).abs().max(0).values[None, None]
        xp, xc = norm(prev)[None].cuda(), norm(cur)[None].cuda()

        with torch.no_grad():
            fp, fc = enc.dino(xp), enc.dino(xc)
            B = 1
            if mode == "random":
                sp, sc = SA(fp), SA(fc)
            elif mode == "shared":
                init = SA.sample_init(B, device=fc.device, dtype=fc.dtype)
                sp, sc = SA(fp, slots_init=init), SA(fc, slots_init=init)
            elif mode == "carryover":
                init = SA.sample_init(B, device=fc.device, dtype=fc.dtype)
                sp = SA(fp, slots_init=init)
                sc = SA(fc, slots_init=sp.detach())
            else:
                raise ValueError(mode)

            # what the trainer actually feeds L_slow
            a_ = F.normalize(sc, dim=-1)
            b_ = F.normalize(sp, dim=-1)
            idx = torch.matmul(a_, b_.transpose(-1, -2)).argmax(-1)[0]
            ident.append(float((idx == torch.arange(idx.numel(),
                                                    device=idx.device)).float().mean()))
            spm = match_slots_nn(sc, sp)

            _, a_cur = enc.slot_decoder(sc)
            _, a_prv = enc.slot_decoder(spm)

        K, N = a_cur.shape[1], a_cur.shape[2]
        g = int(round(N ** 0.5))
        CS += (centroids(a_cur[0], g) - centroids(a_prv[0], g)).norm(dim=-1).tolist()
        CD += (sc - spm).pow(2).sum(-1)[0].tolist()
        m = F.adaptive_avg_pool2d(motion, (g, g)).reshape(1, N).cuda()
        mn = m / (m.sum() + 1e-8)
        an = a_cur[0] / (a_cur[0].sum(-1, keepdim=True) + 1e-8)
        MS += ((an * mn).sum(-1) * N).tolist()

        # per-connected-component displacement. A global bbox over all disks
        # conflates their separation with their displacement (that bug gave a
        # bogus ~12 patch instead of the true ~2.3).
        import scipy.ndimage as ndi
        mm = (motion[0, 0] > 0.02).cpu().numpy()
        if mm.sum() > 30:
            lbl, n = ndi.label(mm)
            for i in range(1, n + 1):
                blob = lbl == i
                if blob.sum() < 30:
                    continue
                ys, xs = np.where(blob)
                span = np.sqrt((ys.max() - ys.min()) ** 2 + (xs.max() - xs.min()) ** 2)
                obj_disp.append(span / 14.0)

    CS, CD, MS = map(np.array, (CS, CD, MS))
    r_cd = np.corrcoef(np.log(CD + 1e-9), np.log(MS + 1e-9))[0, 1]
    r_cs = np.corrcoef(np.log(CS + 1e-9), np.log(MS + 1e-9))[0, 1]
    print(f"ckpt step={ck['step']}  mode={mode}")
    print(f"  centroid shift : median {np.median(CS):.3f}  p90 {np.quantile(CS,0.9):.3f}"
          f"   <-- 判据 (物体位移 ~{np.mean(obj_disp):.2f} patch)")
    print(f"  corr(content diff, motion) = {r_cd:+.3f}"
          f"   corr(centroid shift, motion) = {r_cs:+.3f}")
    print(f"  NN 匹配为恒等置换的比例 = {np.mean(ident):.3f}"
          f"   [carryover 下理应接近 1.0]")


if __name__ == "__main__":
    main()
