#!/usr/bin/env python3
"""Search for a routing signal that identifies object slots without needing
cross-frame slot identity, and without a size confound.

Why the earlier candidates failed: every 'temporal change of the slot' signal
(content diff, centroid shift, alpha IoU drop) is dominated by slot drift.
Background covers ~95% of the frame with no unique slot assignment, so
background slots wander and any change measure ranks them first.

Candidates here:
  spatial_std      alpha spatial standard deviation (low = compact)
                   known: spearman +0.652 but corr(radius, rank) = -0.247
  cc_fraction      mass in the largest connected component of the top-k alpha
                   patches / total. Scale-invariant: a large object is still ONE
                   blob, a background slot is scattered.
  isoperimetric    area / perimeter^2 of the binarized alpha. Scale-invariant.
  pixel_motion     sum_n alpha[n] * |frame_t - frame_{t-1}|[n]
                   Needs no slot identity at all and is self-supervised, but
                   routes a MOVING video background as 'fast' -- i.e. it breaks
                   exactly in the Distracting-Suite setting of Q2.

Reported per candidate: spearman with objectness, top-25% objectness against
the overall mean, and corr with object radius (the size-confound check).
"""
from __future__ import annotations
import argparse, glob, json
import numpy as np, torch, torch.nn.functional as F
import scipy.ndimage as ndi
from PIL import Image
from torchvision import transforms
from hippoact.utils.config import load_config
from validate_semantics import build_encoder
from gt_eval import disk_masks, pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", required=True); ap.add_argument("--n-clips", type=int, default=250)
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

    sig = {k: [] for k in ("spatial_std", "cc_fraction", "isoperimetric", "pixel_motion")}
    OB, OWNR, OWRANK = [], [], {k: [] for k in sig}

    for c in sorted(glob.glob(args.data_dir + "/clip_*"))[: args.n_clips]:
        ann = json.loads(open(f"{c}/annotations.json").read())
        fs = sorted(glob.glob(c + "/*.png"))
        if len(fs) < 2: continue
        i0, i1 = [base(Image.open(p).convert("RGB")) for p in fs[:2]]
        mot = (i1 - i0).abs().max(0).values                      # (H,W)
        with torch.no_grad():
            g0, g1 = enc.dino(nm(i0)[None].cuda()), enc.dino(nm(i1)[None].cuda())
            init = SA.sample_init(1, device=g1.device, dtype=g1.dtype)
            s0 = SA(g0, slots_init=init)
            s1 = SA(g1, slots_init=(s0.detach() if args.mode == "carryover" else init))
            _, a1 = enc.slot_decoder(s1)
        K, N = a1.shape[1], a1.shape[2]; g = int(round(N ** 0.5))
        A = a1[0] / (a1[0].sum(-1, keepdim=True) + 1e-8)
        dev = A.device

        # --- spatial std ---
        ys, xs = torch.meshgrid(torch.arange(g, device=dev, dtype=torch.float),
                                torch.arange(g, device=dev, dtype=torch.float), indexing="ij")
        ys, xs = ys.reshape(-1), xs.reshape(-1)
        cy, cx = (A * ys).sum(-1), (A * xs).sum(-1)
        std = (A * ((ys - cy[:, None]) ** 2 + (xs - cx[:, None]) ** 2)).sum(-1).sqrt()
        sig["spatial_std"] += (-std).tolist()                     # 取负: 高=紧凑

        # --- connected-component fraction & isoperimetric (scale invariant) ---
        k = max(2, N // 16)
        top = A.topk(k, dim=-1).indices
        ccf, iso = [], []
        Anp = A.cpu().numpy()
        for kk in range(K):
            m = np.zeros(N, bool); m[top[kk].cpu().numpy()] = True
            m2 = m.reshape(g, g)
            lbl, n = ndi.label(m2)
            if n == 0:
                ccf.append(0.0); iso.append(0.0); continue
            sizes = ndi.sum(m2, lbl, index=range(1, n + 1))
            big = int(np.argmax(sizes)) + 1
            ccf.append(float(sizes.max() / m2.sum()))
            blob = lbl == big
            area = blob.sum()
            per = (np.abs(np.diff(blob.astype(int), axis=0)).sum()
                   + np.abs(np.diff(blob.astype(int), axis=1)).sum()
                   + blob[0].sum() + blob[-1].sum() + blob[:,0].sum() + blob[:,-1].sum())
            iso.append(float(4 * np.pi * area / max(per ** 2, 1)))
        sig["cc_fraction"] += ccf
        sig["isoperimetric"] += iso

        # --- pixel motion inside footprint ---
        mp = F.adaptive_avg_pool2d(mot[None, None], (g, g)).reshape(-1).to(dev)
        mp = mp / (mp.sum() + 1e-8)
        sig["pixel_motion"] += ((A @ mp) * N).tolist()

        # --- objectness + per-object owner radius ---
        dms = disk_masks(ann["disks"][1], ann["size"])
        anyd = pool(dms, g, dev).sum(0).clamp(max=1.0)
        share = anyd.sum() / N
        ob = (A @ anyd) / (share + 1e-8)
        OB += ob.tolist()
        # 每个物体的 owner slot 在各信号下的排名分位, 用于尺寸混淆检查
        ranks = {kk: (torch.as_tensor(v[-K:], device=dev).argsort().argsort().float() / (K - 1))
                 for kk, v in sig.items()}
        for j, d in enumerate(ann["disks"][1]):
            m1 = pool(dms[j:j+1], g, dev)[0]; sh = m1.sum() / N
            owner = int(((A @ m1) / (sh + 1e-8)).argmax())
            OWNR.append(d["r"])
            for kk in sig: OWRANK[kk].append(float(ranks[kk][owner]))

    OB = np.array(OB); OWNR = np.array(OWNR)
    print(f"{args.label or args.ckpt}  step={ck['step']}  mode={args.mode}")
    print(f"  slots={len(OB)}  objects={len(OWNR)}  半径 {OWNR.min():.0f}-{OWNR.max():.0f}px"
          f" (直径 {2*OWNR.min()/14:.1f}-{2*OWNR.max()/14:.1f} patch)")
    print(f"  全体 objectness 均值 = {OB.mean():.2f}")
    print()
    print(f"  {'候选信号':<20}{'spearman':>10}{'top25% obj':>13}{'corr(半径,排名)':>18}  判读")
    print("  " + "-" * 78)
    for name in ("spatial_std", "cc_fraction", "isoperimetric", "pixel_motion"):
        v = np.array(sig[name])
        rv, ro = v.argsort().argsort().astype(float), OB.argsort().argsort().astype(float)
        sp = np.corrcoef(rv, ro)[0, 1]
        hi = v >= np.quantile(v, 0.75)
        conf = np.corrcoef(OWNR, np.array(OWRANK[name]))[0, 1]
        verdict = ("有效" if (sp > 0.4 and OB[hi].mean() > 1.5 * OB.mean()) else
                   "弱" if sp > 0.2 else "无效")
        if abs(conf) > 0.2 and verdict == "有效": verdict += "(有尺寸混淆)"
        print(f"  {name:<20}{sp:>+10.3f}{OB[hi].mean():>13.2f}{conf:>+18.3f}  {verdict}")


if __name__ == "__main__":
    main()
