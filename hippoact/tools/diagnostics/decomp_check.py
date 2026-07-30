#!/usr/bin/env python3
"""Is Slot Attention actually partitioning the scene, or have all slots
collapsed onto the salient foreground?

Motion score median ~4.1 for nearly every slot suggests the latter. Measures:
  - per-slot alpha centroid + spread
  - pairwise overlap of each slot's top-k patches (do slots look at the same place?)
  - how many patches are 'owned' by each slot under argmax over slots
  - background coverage: is any slot dedicated to the static 95.7% of pixels?
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

    owner_ent, topk_ov, spreads, bg_cov, mo_cov, n_owning = [], [], [], [], [], []

    for c in sorted(glob.glob(args.data_dir + "/clip_*"))[: args.n_clips]:
        fs = sorted(glob.glob(c + "/*.png"))
        prev = base(Image.open(fs[0]).convert("RGB"))
        cur = base(Image.open(fs[1]).convert("RGB"))
        motion = (cur - prev).abs().max(0).values[None, None]
        x = norm(cur)[None].cuda()
        with torch.no_grad():
            f = enc.dino(x)
            init = SA.sample_init(1, device=f.device, dtype=f.dtype)
            s = SA(f, slots_init=init)
            _recon, alpha = enc.slot_decoder(s)
        a = alpha[0]                                    # (K, N)
        K, N = a.shape
        g = int(round(N ** 0.5))

        # --- which slot owns each patch (alpha is softmax over slots) ---
        owner = a.argmax(dim=0)                         # (N,)
        cnt = torch.bincount(owner, minlength=K).float()
        p = cnt / cnt.sum()
        ent = -(p[p > 0] * p[p > 0].log()).sum() / np.log(K)
        owner_ent.append(ent.item())
        n_owning.append(int((cnt > 0).sum()))

        # --- top-k patch overlap between slot pairs ---
        k = max(1, N // 16)
        top = a.topk(k, dim=-1).indices                 # (K, k)
        masks = torch.zeros(K, N, device=a.device)
        masks.scatter_(1, top, 1.0)
        inter = masks @ masks.t()
        iou = inter / (2 * k - inter + 1e-8)
        off = iou[~torch.eye(K, dtype=bool, device=a.device)]
        topk_ov.append(off.mean().item())

        # --- spatial spread of each slot's alpha ---
        an = a / (a.sum(-1, keepdim=True) + 1e-8)
        ys, xs = torch.meshgrid(torch.arange(g, device=a.device, dtype=torch.float),
                                torch.arange(g, device=a.device, dtype=torch.float),
                                indexing="ij")
        ys, xs = ys.reshape(-1), xs.reshape(-1)
        cy = (an * ys).sum(-1); cx = (an * xs).sum(-1)
        var = (an * ((ys - cy[:, None]) ** 2 + (xs - cx[:, None]) ** 2)).sum(-1)
        spreads.append(var.sqrt().mean().item())

        # --- coverage of static vs moving regions ---
        m = F.adaptive_avg_pool2d(motion, (g, g)).reshape(-1).cuda()
        mo = m > m.mean() + m.std()                     # moving patches
        bg = ~mo
        # best slot for background: max alpha mass on bg patches
        bg_cov.append(an[:, bg].sum(-1).max().item())
        mo_cov.append(an[:, mo].sum(-1).max().item() if mo.any() else float("nan"))

    print("=== 场景是否被分割 (K=16, N=256, 16x16 patch 网格) ===")
    print(f"patch 归属熵 (1.0=16 个 slot 均分, 0=全归一个)   : {np.mean(owner_ent):.3f}")
    print(f"实际拥有 >=1 个 patch 的 slot 个数 / 16          : {np.mean(n_owning):.1f}")
    print()
    print(f"slot 两两 top-{256//16} patch 的平均 IoU (0=互不重叠) : {np.mean(topk_ov):.3f}")
    print(f"单个 slot alpha 的空间标准差 (patch 单位)        : {np.mean(spreads):.2f}"
          f"   [均匀铺满 16x16 约为 6.5]")
    print()
    print("=== 前景/背景覆盖 ===")
    print(f"最专注背景的 slot, 其 alpha 质量落在静止区的比例 : {np.mean(bg_cov):.3f}")
    print(f"最专注运动的 slot, 其 alpha 质量落在运动区的比例 : {np.nanmean(mo_cov):.3f}")
    print()
    print("参考: 静止 patch 约占 95%, 所以一个真正的背景 slot 该有 ~0.95 的质量在静止区;")
    print("      一个真正的物体 slot 该有远高于 5% 的质量在运动区.")


if __name__ == "__main__":
    main()
