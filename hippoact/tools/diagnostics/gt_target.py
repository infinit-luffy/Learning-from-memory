#!/usr/bin/env python3
"""Which per-slot signal predicts 'this slot sits on a moving object'?

Exact ground truth (needs --save-annotations). Candidates:
  (a) content diff        ||slot_t - slot_prev||^2      <- current L_slow target
  (b) centroid shift      ||centroid(alpha_t) - centroid(alpha_prev)||
  (c) alpha IoU drop      1 - IoU(top-k alpha_t, top-k alpha_prev)

A valid 'fast' signal must correlate POSITIVELY with objectness. The current
target is expected to be negative: an object slot that successfully tracks its
object has a stable representation, so content diff is small precisely when the
slot is doing its job.
"""
from __future__ import annotations
import argparse, glob, json
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from hippoact.utils.config import load_config
from validate_semantics import build_encoder
from gt_eval import disk_masks, pool


def centroid(a, g):
    an = a / (a.sum(-1, keepdim=True) + 1e-8)
    ys, xs = torch.meshgrid(torch.arange(g, device=a.device, dtype=torch.float),
                            torch.arange(g, device=a.device, dtype=torch.float),
                            indexing="ij")
    return torch.stack([(an * ys.reshape(-1)).sum(-1),
                        (an * xs.reshape(-1)).sum(-1)], -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", required=True); ap.add_argument("--n-clips", type=int, default=400)
    ap.add_argument("--mode", default="carryover", choices=["shared", "carryover"])
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
    CD, CS, IO, OB = [], [], [], []
    for c in sorted(glob.glob(args.data_dir + "/clip_*"))[: args.n_clips]:
        ann = json.loads(open(f"{c}/annotations.json").read())
        fs = sorted(glob.glob(c + "/*.png"))
        if len(fs) < 2: continue
        i0, i1 = [base(Image.open(p).convert("RGB")) for p in fs[:2]]
        with torch.no_grad():
            g0, g1 = enc.dino(nm(i0)[None].cuda()), enc.dino(nm(i1)[None].cuda())
            init = SA.sample_init(1, device=g1.device, dtype=g1.dtype)
            s0 = SA(g0, slots_init=init)
            s1 = SA(g1, slots_init=(s0.detach() if args.mode == "carryover" else init))
            _, a0 = enc.slot_decoder(s0); _, a1 = enc.slot_decoder(s1)
        K, N = a1.shape[1], a1.shape[2]; g = int(round(N ** 0.5))
        A0 = a0[0]/(a0[0].sum(-1,keepdim=True)+1e-8)
        A1 = a1[0]/(a1[0].sum(-1,keepdim=True)+1e-8)
        CD += (s1[0]-s0[0]).pow(2).sum(-1).tolist()
        CS += (centroid(A1,g)-centroid(A0,g)).norm(dim=-1).tolist()
        k = N//16
        b0 = torch.zeros_like(A0); b0.scatter_(1, A0.topk(k,-1).indices, 1.)
        b1 = torch.zeros_like(A1); b1.scatter_(1, A1.topk(k,-1).indices, 1.)
        inter = (b0*b1).sum(-1); IO += (1 - inter/(2*k-inter+1e-8)).tolist()
        dm = pool(disk_masks(ann["disks"][1], ann["size"]), g, A1.device)
        anyd = dm.sum(0).clamp(max=1.0); share = anyd.sum()/N
        OB += ((A1 @ anyd)/(share+1e-8)).tolist()
    CD, CS, IO, OB = map(np.array, (CD, CS, IO, OB))
    print(f"{args.label or args.ckpt}  step={ck['step']}  mode={args.mode}  slots={len(OB)}")
    print(f"  objectness: mean {OB.mean():.2f}  median {np.median(OB):.2f}  (1.0=chance)")
    print()
    print(f"  {'候选 target':<34}{'pearson(log)':>14}{'spearman':>11}{'top25% 的 objectness':>22}")
    print("  " + "-"*79)
    for name, v in (("(a) content diff  [现行 L_slow]", CD),
                    ("(b) centroid shift [你提议]", CS),
                    ("(c) alpha IoU drop", IO)):
        pe = np.corrcoef(np.log(v+1e-9), np.log(OB+1e-9))[0,1]
        rv, ro = v.argsort().argsort().astype(float), OB.argsort().argsort().astype(float)
        sp = np.corrcoef(rv, ro)[0,1]
        hi = v >= np.quantile(v, 0.75)
        print(f"  {name:<34}{pe:>+14.3f}{sp:>+11.3f}{OB[hi].mean():>22.2f}")
    print()
    print(f"  参照: 全体 objectness 均值 {OB.mean():.2f} —— 一个有效的 fast 信号,")
    print(f"        其 top25% 的 objectness 应显著高于这个值")


if __name__ == "__main__":
    main()
