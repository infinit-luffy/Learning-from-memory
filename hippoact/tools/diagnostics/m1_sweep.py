#!/usr/bin/env python3
"""M1 一轮 sweep: 单遍数据内评估所有 (w_iou, w_feat, topk) 组合。
TODO 中间档授权范围: 只调匹配的超参, 不改编码器/loss/判据。"""
from __future__ import annotations
import argparse, glob, itertools
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from scipy.optimize import linear_sum_assignment
from torchvision import transforms
from hippoact.utils.config import load_config
from validate_semantics import build_encoder

def pool_mask(m, g, dev):
    x = F.adaptive_avg_pool2d(torch.from_numpy(m.astype(np.float32))[None,None],
                              (g,g)).reshape(-1).to(dev)
    return x/(x.sum()+1e-8)

def topk_binary(a, k):
    m = torch.zeros_like(a); m.scatter_(1, a.topk(k,dim=-1).indices, 1.0); return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", required=True); ap.add_argument("--n-clips", type=int, default=400)
    args = ap.parse_args()
    cfg = load_config(args.config); size = cfg.encoder.image_size
    enc = build_encoder(cfg).cuda()
    ck = torch.load(args.ckpt, map_location="cuda", weights_only=False)
    enc.load_state_dict(ck["encoder"], strict=False); enc.eval()
    SA = enc.slot_attn
    nm = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    base = transforms.Compose([transforms.Resize(size), transforms.CenterCrop(size),
                              transforms.ToTensor()])
    TOPK = [8, 16, 32, 64]
    WS = [(1.0,0.0),(0.7,0.3),(0.5,0.5),(0.3,0.7),(0.0,1.0)]
    trk = {(w,k): [] for w in WS for k in TOPK}
    stab = {(w,k): [] for w in WS for k in TOPK}
    ctrl = {"identity": [], "random": []}
    loc, used = [], 0
    for c in sorted(glob.glob(args.data_dir + "/clip_*"))[:args.n_clips]:
        fs = sorted(glob.glob(c + "/*.png"))
        if len(fs) < 4: continue
        f = [base(Image.open(p).convert("RGB")) for p in fs[:4]]
        d = [((f[i+1]-f[i]).abs().max(0).values > 0.02).numpy() for i in range(3)]
        obj1, obj2 = d[0]&d[1], d[1]&d[2]
        if obj1.sum() < 30 or obj2.sum() < 30: continue
        used += 1
        with torch.no_grad():
            g1, g2 = enc.dino(nm(f[1])[None].cuda()), enc.dino(nm(f[2])[None].cuda())
            N = g1.shape[1]; gg = int(round(N**0.5))
            s1 = SA(g1, slots_init=SA.sample_init(1,device=g1.device,dtype=g1.dtype))
            s2 = SA(g2, slots_init=SA.sample_init(1,device=g2.device,dtype=g2.dtype))
            _, a1 = enc.slot_decoder(s1); _, a2 = enc.slot_decoder(s2)
        A1 = a1[0]/(a1[0].sum(-1,keepdim=True)+1e-8)
        A2 = a2[0]/(a2[0].sum(-1,keepdim=True)+1e-8)
        K = A1.shape[0]
        o1, o2 = pool_mask(obj1,gg,A1.device), pool_mask(obj2,gg,A1.device)
        on1 = (A1*o1).sum(-1)*N
        loc.append((on1>2.0).float().mean().item())
        sel = (on1>2.0).nonzero().flatten()
        if len(sel)==0: continue
        base_mass = (A1[sel]*o2).sum(-1)*N
        fp, fc = F.normalize(s1[0],dim=-1), F.normalize(s2[0],dim=-1)
        cos = fp @ fc.t()
        bins = {k: (topk_binary(A1,k), topk_binary(A2,k)) for k in TOPK}
        for k in TOPK:
            bp, bc = bins[k]
            inter = bp @ bc.t(); iou = inter/(2*k-inter+1e-8)
            for (wi, wf) in WS:
                score = wi*iou + wf*cos
                _, cc = linear_sum_assignment(-score.cpu().numpy())
                perm = torch.as_tensor(cc, device=A1.device)
                gained = (A2[perm[sel]]*o2).sum(-1)*N - base_mass
                trk[((wi,wf),k)] += (gained>0).float().tolist()
                stab[((wi,wf),k)].append((((A2[perm[sel]]*o2).sum(-1)*N)>2.0).float().mean().item())
        for name, p in (("identity", torch.arange(K,device=A1.device)),
                        ("random", torch.randperm(K,device=A1.device))):
            gained = (A2[p[sel]]*o2).sum(-1)*N - base_mass
            ctrl[name] += (gained>0).float().tolist()
    print(f"clips_used={used}   localization={np.mean(loc):.3f}")
    print(f"对照: identity={np.mean(ctrl['identity']):.3f}  random={np.mean(ctrl['random']):.3f}"
          f"   <- 经验地板, 非 0.5")
    print()
    hdr = "  (w_iou,w_feat)  " + "".join(f"{('top'+str(k)):>10}" for k in TOPK)
    print(hdr); print("  " + "-"*(len(hdr)-2))
    best = (None, -1)
    for w in WS:
        row = f"  {str(w):<16}"
        for k in TOPK:
            v = np.mean(trk[(w,k)]); row += f"{v:>10.3f}"
            if v > best[1]: best = ((w,k), v)
        print(row)
    print()
    print(f"  最佳: w={best[0][0]} topk={best[0][1]}  matched tracking={best[1]:.3f}"
          f"  assignment 稳定性={np.mean(stab[best[0]]):.3f}")

if __name__ == "__main__":
    main()
