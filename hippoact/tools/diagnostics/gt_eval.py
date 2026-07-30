#!/usr/bin/env python3
"""Exact-ground-truth evaluation of slot localization and cross-frame identity.

Requires a dataset generated with --save-annotations. Every earlier metric in
this directory inferred object masks from pixel differences; those masks are
edge slivers (median 670 px against a 616-2124 px disk), so numbers measured
against them are unreliable. Here masks are rendered from exact geometry.

Because annotations index disks consistently across frames, object identity is
known. Tracking therefore becomes a direct test rather than a proxy:

  localization      : per disk, is there a slot whose alpha concentrates on it?
                      concentration = (alpha mass on disk) / (disk area / N)
                      1.0 = chance, >= 2.0 counts as "covered"
  identity persist. : owner_t(j)   = argmax_k alpha_k mass on disk j at t
                      tracked if owner_{t+1}(j) == owner_t(j)   (chance = 1/K)
  exclusivity       : does the owner slot put most of its mass on that one disk
                      rather than spread over other disks / background?
"""
from __future__ import annotations
import argparse, glob, json
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from scipy.optimize import linear_sum_assignment
from torchvision import transforms
from hippoact.utils.config import load_config
from validate_semantics import build_encoder


def disk_masks(frame_ann, size):
    """Return (n_disks, size, size) bool masks from exact geometry."""
    y, x = np.ogrid[:size, :size]
    return np.stack([((x - d["cx"]) ** 2 + (y - d["cy"]) ** 2) <= d["r"] ** 2
                     for d in frame_ann])


def pool(masks, g, dev):
    """(n, H, W) bool -> (n, g*g) float, mean occupancy per patch."""
    t = torch.from_numpy(masks.astype(np.float32))[:, None]
    return F.adaptive_avg_pool2d(t, (g, g)).reshape(masks.shape[0], -1).to(dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--n-clips", type=int, default=400)
    ap.add_argument("--mode", default="shared",
                    choices=["fresh", "shared", "carryover"])
    ap.add_argument("--match", action="store_true",
                    help="Hungarian-match slots across frames before the "
                         "identity test (for fresh init, index is meaningless)")
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

    conc, covered, persist, excl, n_disk_l, used = [], [], [], [], [], 0

    for c in sorted(glob.glob(args.data_dir + "/clip_*"))[: args.n_clips]:
        ann_p = f"{c}/annotations.json"
        try:
            ann = json.loads(open(ann_p).read())
        except FileNotFoundError:
            raise SystemExit(f"{ann_p} 不存在 —— 数据集需用 --save-annotations 生成")
        fs = sorted(glob.glob(c + "/*.png"))
        if len(fs) < 2:
            continue
        used += 1
        imgs = [base(Image.open(p).convert("RGB")) for p in fs[:2]]
        with torch.no_grad():
            g1, g2 = (enc.dino(nm(imgs[0])[None].cuda()),
                      enc.dino(nm(imgs[1])[None].cuda()))
            if args.mode == "shared":
                init = SA.sample_init(1, device=g1.device, dtype=g1.dtype)
                s1, s2 = SA(g1, slots_init=init), SA(g2, slots_init=init)
            elif args.mode == "carryover":
                init = SA.sample_init(1, device=g1.device, dtype=g1.dtype)
                s1 = SA(g1, slots_init=init)
                s2 = SA(g2, slots_init=s1.detach())
            else:
                s1 = SA(g1, slots_init=SA.sample_init(1, device=g1.device, dtype=g1.dtype))
                s2 = SA(g2, slots_init=SA.sample_init(1, device=g1.device, dtype=g1.dtype))
            _, a1 = enc.slot_decoder(s1)
            _, a2 = enc.slot_decoder(s2)

        K, N = a1.shape[1], a1.shape[2]
        g = int(round(N ** 0.5))
        A1 = a1[0] / (a1[0].sum(-1, keepdim=True) + 1e-8)
        A2 = a2[0] / (a2[0].sum(-1, keepdim=True) + 1e-8)

        m1 = pool(disk_masks(ann["disks"][0], ann["size"]), g, A1.device)  # (nd, N)
        m2 = pool(disk_masks(ann["disks"][1], ann["size"]), g, A1.device)
        nd = m1.shape[0]
        n_disk_l.append(nd)

        # concentration: alpha mass on disk / chance share of that disk
        share1 = m1.sum(-1, keepdim=True) / N                      # (nd,1)
        mass1 = A1 @ m1.t()                                        # (K, nd)
        conc1 = (mass1 / (share1.t() + 1e-8))                      # (K, nd)
        share2 = m2.sum(-1, keepdim=True) / N
        conc2 = ((A2 @ m2.t()) / (share2.t() + 1e-8))

        best1 = conc1.max(0)
        conc += best1.values.tolist()
        covered += (best1.values >= 2.0).float().tolist()

        owner1 = best1.indices                                     # (nd,)
        owner2 = conc2.argmax(0)
        if args.match:
            k = 64
            b1 = torch.zeros_like(A1); b1.scatter_(1, A1.topk(k, -1).indices, 1.)
            b2 = torch.zeros_like(A2); b2.scatter_(1, A2.topk(k, -1).indices, 1.)
            inter = b1 @ b2.t(); iou = inter / (2 * k - inter + 1e-8)
            cos = F.normalize(s1[0], dim=-1) @ F.normalize(s2[0], dim=-1).t()
            _, perm = linear_sum_assignment(-(0.5 * iou + 0.5 * cos).cpu().numpy())
            perm = torch.as_tensor(perm, device=A1.device)
            persist += (perm[owner1] == owner2).float().tolist()
        else:
            persist += (owner1 == owner2).float().tolist()

        # exclusivity: owner slot's mass on its own disk vs on all disks+bg
        own_mass = mass1[owner1, torch.arange(nd, device=A1.device)]
        excl += (own_mass / (A1[owner1].sum(-1) + 1e-8)).tolist()

    K = 16
    print(f"{args.label or args.ckpt}  step={ck['step']}  mode={args.mode}"
          f"{'  +match' if args.match else ''}  clips={used}  disks={np.mean(n_disk_l):.1f}/clip")
    print(f"  concentration (最佳 slot 在该物体上的富集倍数, 1.0=chance)"
          f"  : mean {np.mean(conc):.2f}  median {np.median(conc):.2f}")
    print(f"  coverage      (富集 >= 2.0 的物体占比)"
          f"                     : {np.mean(covered):.3f}")
    print(f"  identity persistence (同一物体跨帧仍归同一 slot)"
          f"        : {np.mean(persist):.3f}   [chance = 1/{K} = {1/K:.3f}]")
    print(f"  exclusivity   (owner slot 的质量落在该物体上的比例)"
          f"    : {np.mean(excl):.3f}")


if __name__ == "__main__":
    main()
