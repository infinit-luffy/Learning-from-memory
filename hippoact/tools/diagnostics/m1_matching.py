#!/usr/bin/env python3
"""TODO Step M1 -- Tracking-by-Matching probe (eval-only).

Encoder does localization only (fresh init per frame, NOT shared -- shared init
freezes the partition). Cross-frame identity is recovered afterwards by
Hungarian assignment on a mix of alpha-IoU and slot-feature cosine.

    cost[i,j] = -(w_iou * IoU(alpha_prev[i], alpha_cur[j])
                  + w_feat * cos(slot_prev[i], slot_cur[j]))

Object masks use the intersection definition on both sides (4-frame clips):
    obj1 = d01 & d12      obj2 = d12 & d23

Controls (metric hygiene): 'identity' = naive index pairing, 'random' = random
permutation. With fresh init both should sit at chance; if Hungarian does not
beat them, matching is not doing any work.
"""
from __future__ import annotations
import argparse, glob
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from scipy.optimize import linear_sum_assignment
from torchvision import transforms
from hippoact.utils.config import load_config
from validate_semantics import build_encoder


def pool_mask(m, g, dev):
    x = F.adaptive_avg_pool2d(torch.from_numpy(m.astype(np.float32))[None, None],
                              (g, g)).reshape(-1).to(dev)
    return x / (x.sum() + 1e-8)


def topk_binary(a, k):
    m = torch.zeros_like(a)
    m.scatter_(1, a.topk(k, dim=-1).indices, 1.0)
    return m


def hungarian_match(a_prev, a_cur, s_prev, s_cur, w_iou, w_feat, k):
    bp, bc = topk_binary(a_prev, k), topk_binary(a_cur, k)
    inter = bp @ bc.t()
    iou = inter / (2 * k - inter + 1e-8)
    fp = F.normalize(s_prev, dim=-1)
    fc = F.normalize(s_cur, dim=-1)
    cos = fp @ fc.t()
    score = w_iou * iou + w_feat * cos
    r, c = linear_sum_assignment(-score.cpu().numpy())
    return torch.as_tensor(c, device=a_prev.device), score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--n-clips", type=int, default=400)
    ap.add_argument("--w-iou", type=float, default=0.7)
    ap.add_argument("--w-feat", type=float, default=0.3)
    ap.add_argument("--topk", type=int, default=16)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    cfg = load_config(args.config); size = cfg.encoder.image_size
    enc = build_encoder(cfg).cuda()
    ck = torch.load(args.ckpt, map_location="cuda", weights_only=False)
    enc.load_state_dict(ck["encoder"], strict=False); enc.eval()
    SA = enc.slot_attn
    nm = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    base = transforms.Compose([transforms.Resize(size), transforms.CenterCrop(size),
                              transforms.ToTensor()])

    trk = {"matched": [], "identity": [], "random": []}
    loc, stab, iou_matched = [], [], []
    used = 0

    for c in sorted(glob.glob(args.data_dir + "/clip_*"))[: args.n_clips]:
        fs = sorted(glob.glob(c + "/*.png"))
        if len(fs) < 4: continue
        f = [base(Image.open(p).convert("RGB")) for p in fs[:4]]
        d = [((f[i+1]-f[i]).abs().max(0).values > 0.02).numpy() for i in range(3)]
        obj1, obj2 = d[0] & d[1], d[1] & d[2]
        if obj1.sum() < 30 or obj2.sum() < 30: continue
        used += 1
        x1, x2 = nm(f[1])[None].cuda(), nm(f[2])[None].cuda()
        with torch.no_grad():
            g1, g2 = enc.dino(x1), enc.dino(x2)
            N = g1.shape[1]; gg = int(round(N ** 0.5))
            # independent fresh init per frame -- encoder does localization only
            s1 = SA(g1, slots_init=SA.sample_init(1, device=g1.device, dtype=g1.dtype))
            s2 = SA(g2, slots_init=SA.sample_init(1, device=g2.device, dtype=g2.dtype))
            _, a1 = enc.slot_decoder(s1)
            _, a2 = enc.slot_decoder(s2)
        A1 = a1[0] / (a1[0].sum(-1, keepdim=True) + 1e-8)
        A2 = a2[0] / (a2[0].sum(-1, keepdim=True) + 1e-8)
        K = A1.shape[0]
        o1 = pool_mask(obj1, gg, A1.device)
        o2 = pool_mask(obj2, gg, A1.device)

        on1 = (A1 * o1).sum(-1) * N
        loc.append((on1 > 2.0).float().mean().item())
        sel = (on1 > 2.0).nonzero().flatten()
        if len(sel) == 0: continue

        perm, score = hungarian_match(A1, A2, s1[0], s2[0],
                                      args.w_iou, args.w_feat, args.topk)
        pairings = {
            "matched":  perm,
            "identity": torch.arange(K, device=A1.device),
            "random":   torch.randperm(K, device=A1.device),
        }
        base_mass = (A1[sel] * o2).sum(-1) * N
        for name, p in pairings.items():
            tgt = A2[p[sel]]
            gained = (tgt * o2).sum(-1) * N - base_mass
            trk[name] += (gained > 0).float().tolist()
        iou_matched += score[sel, perm[sel]].tolist()

        # assignment stability: does an on-object slot chain stay on-object?
        stab.append(((A2[perm[sel]] * o2).sum(-1) * N > 2.0).float().mean().item())

    print(f"{args.label or args.ckpt}  step={ck['step']}  clips_used={used}"
          f"  w_iou={args.w_iou} w_feat={args.w_feat} topk={args.topk}")
    print()
    print(f"  localization (on-object 交集)        : {np.mean(loc):.3f}"
          f"    [参照 shared-init = 0.510]")
    print()
    print("  matched tracking (方向一致性, 0.5=随机):")
    for name in ("matched", "identity", "random"):
        v = trk[name]
        print(f"    {name:<9} = {np.mean(v):.3f}   (n={len(v)})")
    print()
    print(f"  assignment 稳定性 (配对后仍在物体上的比例): {np.mean(stab):.3f}")
    print(f"  被选中 slot 的匹配得分 (w_iou*IoU+w_feat*cos): "
          f"mean {np.mean(iou_matched):.3f}")


if __name__ == "__main__":
    main()
