#!/usr/bin/env python3
"""W1.1 判据: router 判 fast 的 slot 对 walker 的富集度.

用 MuJoCo 分割渲染的精确掩膜 (collect_frames.py --save-segmentation), 而非
帧差+闭运算的 proxy —— 后者会把地板倒影一并圈入, 且在 easy 上完全失效
(DAVIS 视频主导帧差: easy 47.9% vs clean 12.7% 的像素变化).

  enrichment(k) = (slot k 落在 walker 上的 alpha 质量) / (walker 占画面比例)
                  1.0 = chance, 判据要求 fast slots >= 3.0

同时报 oracle / uniform 对照 (合成阶段的教训: 没有对照就无法区分
「模型坏了」和「尺子坏了」), 以及 easy 上背景的路由分布.
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
from validate_semantics import build_encoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--n-clips", type=int, default=100)
    ap.add_argument("--frames-per-clip", type=int, default=5)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    cfg = load_config(args.config); size = cfg.encoder.image_size
    enc = build_encoder(cfg).cuda()
    ck = torch.load(args.ckpt, map_location="cuda", weights_only=False)
    enc.load_state_dict(ck["encoder"], strict=False); enc.eval()
    SA = enc.slot_attn
    nm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    base = transforms.Compose([transforms.Resize(size),
                              transforms.CenterCrop(size), transforms.ToTensor()])

    enr, is_fast, p_fast, orc, unif, share_l = [], [], [], [], [], []

    for c in sorted(glob.glob(args.data_dir + "/clip_*"))[: args.n_clips]:
        st = np.load(f"{c}/states.npz")
        if "walker_mask" not in st:
            raise SystemExit(f"{c}/states.npz 无 walker_mask —— 需用 "
                             "--save-segmentation 采集")
        fs = sorted(glob.glob(c + "/*.png"))
        idx = np.linspace(2, len(fs) - 3, args.frames_per_clip).astype(int)
        for t in idx:
            img = base(Image.open(fs[t]).convert("RGB"))
            with torch.no_grad():
                g = enc.dino(nm(img)[None].cuda())
                s = SA(g, slots_init=SA.sample_init(1, device=g.device, dtype=g.dtype))
                _, a = enc.slot_decoder(s)
                _, lg = enc.router(s)
            K, N = a.shape[1], a.shape[2]
            gg = int(round(N ** 0.5))
            A = a[0] / (a[0].sum(-1, keepdim=True) + 1e-8)
            m = torch.from_numpy(st["walker_mask"][t].astype(np.float32))[None, None]
            mp = F.adaptive_avg_pool2d(m, (gg, gg)).reshape(-1).cuda()
            share = mp.sum() / N          # walker 占画面比例
            share_l.append(float(share))
            # 富集度 = slot 落在 walker 上的 alpha 质量 / walker 占画面比例。
            # mp 必须用逐 patch 占用率本身, 不能归一化成和为 1 的分布 ——
            # 后者会让均匀 slot 得到 1/(N*share) 而非 1.0, 整体缩小 mp.sum() 倍。
            # uniform 对照就是用来抓这个的: 它必须等于 1.00。
            enr += ((A @ mp) / (share + 1e-8)).tolist()
            is_fast += (lg[0].argmax(-1) == 1).float().tolist()
            p_fast += lg[0].softmax(-1)[:, 1].tolist()
            oracle_alpha = mp / (mp.sum() + 1e-8)      # alpha 恰好等于掩膜
            orc.append(float((oracle_alpha @ mp) / (share + 1e-8)))
            u = torch.full_like(mp, 1.0 / N)
            unif.append(float((u @ mp) / (share + 1e-8)))

    enr = np.array(enr); is_fast = np.array(is_fast); p_fast = np.array(p_fast)
    f, s = enr[is_fast == 1], enr[is_fast == 0]
    print(f"{args.label or args.ckpt}  step={ck['step']}  frames={len(share_l)}")
    print(f"  walker 占画面 = {np.mean(share_l):.4f}")
    print(f"  对照: oracle slot 富集 = {np.mean(orc):.2f}   uniform slot = "
          f"{np.mean(unif):.2f}  (定义上应为 1.00)")
    print(f"  router 判 fast 占比 = {is_fast.mean():.3f}")
    print(f"  富集度  FAST: n={len(f):5d} mean {f.mean() if len(f) else np.nan:6.2f}"
          f"  median {np.median(f) if len(f) else np.nan:6.2f}")
    print(f"  富集度  SLOW: n={len(s):5d} mean {s.mean() if len(s) else np.nan:6.2f}"
          f"  median {np.median(s) if len(s) else np.nan:6.2f}")
    if len(f) > 1 and len(s) > 1:
        pooled = np.sqrt((f.var(ddof=1) + s.var(ddof=1)) / 2)
        d = (f.mean() - s.mean()) / (pooled + 1e-12)
        r = enr.argsort().argsort().astype(float)
        auc = (r[is_fast == 1].mean() - r[is_fast == 0].mean()) / len(enr) + 0.5
        print(f"  Cohen's d = {d:+.3f}   秩 AUC = {auc:.3f}")
    print()
    verdict = "通过" if (len(f) and f.mean() >= 3.0) else "不通过"
    print(f"  >>> W1.1 判据 (fast slots 富集 >= 3.0): {f.mean() if len(f) else np.nan:.2f}"
          f"  --> {verdict}")


if __name__ == "__main__":
    main()
