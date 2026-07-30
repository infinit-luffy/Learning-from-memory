#!/usr/bin/env python3
"""Are slots object-anchored or position-anchored?

If Slot Attention has degenerated into a fixed spatial tiling, then for a
FIXED slots_init the per-slot alpha centroid will be nearly the same on two
completely unrelated images, and will vary a lot when the init changes on the
SAME image. That is the opposite of object-centric behaviour.

  same_init_diff_image  << diff_init_same_image   =>  position-anchored (bad)
  same_init_diff_image  >> diff_init_same_image   =>  content-anchored (good)
"""
from __future__ import annotations

import argparse
import glob

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from hippoact.utils.config import load_config
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
    ap.add_argument("--n-pairs", type=int, default=100)
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

    clips = sorted(glob.glob(args.data_dir + "/clip_*"))
    same_init_diff_img, diff_init_same_img, alpha_cos_same_init = [], [], []

    for i in range(args.n_pairs):
        cA, cB = clips[i], clips[(i + 500) % len(clips)]
        fa = sorted(glob.glob(cA + "/*.png"))[0]
        fb = sorted(glob.glob(cB + "/*.png"))[0]
        xa = norm(base(Image.open(fa).convert("RGB")))[None].cuda()
        xb = norm(base(Image.open(fb).convert("RGB")))[None].cuda()
        with torch.no_grad():
            fA, fB = enc.dino(xa), enc.dino(xb)
            init1 = SA.sample_init(1, device=fA.device, dtype=fA.dtype)
            init2 = SA.sample_init(1, device=fA.device, dtype=fA.dtype)
            sA1 = SA(fA, slots_init=init1)
            sB1 = SA(fB, slots_init=init1)     # same init, different image
            sA2 = SA(fA, slots_init=init2)     # different init, same image
            _, aA1 = enc.slot_decoder(sA1)
            _, aB1 = enc.slot_decoder(sB1)
            _, aA2 = enc.slot_decoder(sA2)
        K, N = aA1.shape[1], aA1.shape[2]
        g = int(round(N ** 0.5))
        cen = lambda a: centroids(a[0], g)
        same_init_diff_img.append((cen(aA1) - cen(aB1)).norm(dim=-1).mean().item())
        diff_init_same_img.append((cen(aA1) - cen(aA2)).norm(dim=-1).mean().item())
        # cosine of the alpha maps themselves, same init different image
        u = aA1[0] / aA1[0].norm(dim=-1, keepdim=True)
        v = aB1[0] / aB1[0].norm(dim=-1, keepdim=True)
        alpha_cos_same_init.append((u * v).sum(-1).mean().item())

    a = np.mean(same_init_diff_img)
    b = np.mean(diff_init_same_img)
    print("=== slot alpha centroid 的位移来源 (patch 单位) ===")
    print(f"同 init, 不同图像  : {a:.3f}   <- 若很小, alpha 由 init 决定, 与内容无关")
    print(f"不同 init, 同一图像: {b:.3f}   <- 若很大, init 主导 alpha")
    print()
    print(f"比值 (同init不同图 / 不同init同图) = {a / max(b, 1e-9):.3f}")
    print(f"同 init 下两张不同图像的 alpha 图余弦相似度 = "
          f"{np.mean(alpha_cos_same_init):.3f}   [1.0 = 完全相同]")
    print()
    print("参考: 真正 object-centric 时, 换图像应大幅改变 alpha (比值 >> 1);")
    print("      退化成空间切块时, 换图像几乎不改变 alpha (比值 << 1).")


if __name__ == "__main__":
    main()
