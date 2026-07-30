#!/usr/bin/env python3
"""A/B eval: SNR + Cohen's d for a checkpoint, using shared slot init the way
training does. Prints both so shared-init contribution can be attributed."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--n-clips", type=int, default=150)
    ap.add_argument("--shared-init", action="store_true",
                    help="use one init for both frames (as training does)")
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

    t_all, n_all, fast, slow, corrs = [], [], [], [], []
    gum, arg = [], []

    for c in sorted(glob.glob(args.data_dir + "/clip_*"))[: args.n_clips]:
        fs = sorted(glob.glob(c + "/*.png"))
        prev = base(Image.open(fs[0]).convert("RGB"))
        cur = base(Image.open(fs[1]).convert("RGB"))
        motion = (cur - prev).abs().max(0).values[None, None]
        xp, xc = norm(prev)[None].cuda(), norm(cur)[None].cuda()

        with torch.no_grad():
            fp, fc = enc.dino(xp), enc.dino(xc)
            if args.shared_init:
                init = SA.sample_init(1, device=fc.device, dtype=fc.dtype)
                sp = SA(fp, slots_init=init)
                sc_ = SA(fc, slots_init=init)
                sc2 = SA(fc, slots_init=init)
            else:
                sp, sc_, sc2 = SA(fp), SA(fc), SA(fc)
            spm = match_slots_nn(sc_, sp)
            d = (sc_ - spm).pow(2).sum(-1)[0]
            t_all.append(d.mean().item())
            n_all.append((sc_ - match_slots_nn(sc_, sc2)).pow(2).sum(-1).mean().item())
            _recon, alpha = enc.slot_decoder(sc_)
            g_hard, lg = enc.router(sc_)

        K, N = alpha.shape[1], alpha.shape[2]
        gg = int(round(N ** 0.5))
        m = F.adaptive_avg_pool2d(motion, (gg, gg)).reshape(1, N).cuda()
        m = m / (m.sum() + 1e-8)
        a = alpha[0]
        a = a / (a.sum(-1, keepdim=True) + 1e-8)
        ms = (a * m).sum(-1) * N

        isl = lg[0].argmax(-1) == 0
        fast += ms[~isl].tolist()
        slow += ms[isl].tolist()
        gum.append(g_hard[..., 0].mean().item())
        arg.append(isl.float().mean().item())
        corrs.append(np.corrcoef(np.log(d.cpu().numpy() + 1e-9),
                                 np.log(ms.cpu().numpy() + 1e-9))[0, 1])

    t, n = np.mean(t_all), np.mean(n_all)
    print(f"shared_init = {args.shared_init}")
    print()
    print("=== SNR decomposition ===")
    print(f"adjacent-frame diff (matched) = {t:.4f}")
    print(f"same-image repeat   diff      = {n:.6f}   <- init-noise floor")
    print(f"true temporal signal          = {t - n:.4f}")
    snr = "inf (noise floor == 0)" if n < 1e-6 else f"{(t - n) / n:.2f}"
    print(f"SNR = {snr}")
    print(f"corr(log diff, log motion_score) = {np.nanmean(corrs):+.3f}")
    print()
    fast, slow = np.array(fast), np.array(slow)
    print("=== Cohen's d (motion score: fast vs slow) ===")
    print(f"FAST n={len(fast)} mean {fast.mean():.3f} median {np.median(fast):.3f}")
    print(f"SLOW n={len(slow)} mean {slow.mean():.3f} median {np.median(slow):.3f}")
    if len(fast) > 1 and len(slow) > 1:
        pooled = np.sqrt((fast.var(ddof=1) + slow.var(ddof=1)) / 2)
        d_ = (fast.mean() - slow.mean()) / (pooled + 1e-12)
        tt = (fast.mean() - slow.mean()) / np.sqrt(
            fast.var(ddof=1) / len(fast) + slow.var(ddof=1) / len(slow) + 1e-12)
        allv = np.concatenate([fast, slow])
        r = allv.argsort().argsort().astype(float)
        auc = (r[: len(fast)].mean() - r[len(fast):].mean()) / len(allv) + 0.5
        print(f"Cohen's d = {d_:+.3f}   Welch t = {tt:+.2f}   rank AUC = {auc:.3f}")
    print()
    print(f"slow_ratio gumbel = {np.mean(gum):.4f}   argmax = {np.mean(arg):.4f}"
          f"   gap = {abs(np.mean(gum) - np.mean(arg)):.4f}")


if __name__ == "__main__":
    main()
