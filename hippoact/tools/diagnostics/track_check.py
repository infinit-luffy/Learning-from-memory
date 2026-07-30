#!/usr/bin/env python3
"""Unambiguous tracking test.

motion(prev,cur) covers the object at BOTH t-1 and t, so a slot parked between
the two positions scores high without tracking anything. Fix: the object's
position at time t is (approximately) the INTERSECTION of motion(t-1,t) and
motion(t,t+1) -- both differences contain the object at t.

With obj_t and obj_{t+1} available we can ask the real question:
  does a slot's alpha centroid move from obj_t toward obj_{t+1}?
"""
from __future__ import annotations
import argparse, glob
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from hippoact.utils.config import load_config
from validate_semantics import build_encoder

def cen(a, g):
    an = a / (a.sum(-1, keepdim=True) + 1e-8)
    ys, xs = torch.meshgrid(torch.arange(g, device=a.device, dtype=torch.float),
                            torch.arange(g, device=a.device, dtype=torch.float),
                            indexing="ij")
    return torch.stack([(an*ys.reshape(-1)).sum(-1), (an*xs.reshape(-1)).sum(-1)], -1)

def pool(mask, g):
    return F.adaptive_avg_pool2d(mask[None,None].float(), (g,g)).reshape(-1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", required=True); ap.add_argument("--n-clips", type=int, default=150)
    ap.add_argument("--mode", default="carryover")
    args = ap.parse_args()
    cfg = load_config(args.config); size = cfg.encoder.image_size
    enc = build_encoder(cfg).cuda()
    ck = torch.load(args.ckpt, map_location="cuda", weights_only=False)
    enc.load_state_dict(ck["encoder"], strict=False); enc.eval()
    SA = enc.slot_attn
    norm = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    base = transforms.Compose([transforms.Resize(size), transforms.CenterCrop(size),
                               transforms.ToTensor()])
    on_obj, follow, shift, n_ok = [], [], [], 0
    for c in sorted(glob.glob(args.data_dir + "/clip_*"))[:args.n_clips]:
        fs = sorted(glob.glob(c + "/*.png"))
        if len(fs) < 3: continue
        f0,f1,f2 = [base(Image.open(f).convert("RGB")) for f in fs[:3]]
        d01 = (f1-f0).abs().max(0).values > 0.02
        d12 = (f2-f1).abs().max(0).values > 0.02
        obj1 = (d01 & d12)                      # object position at t=1
        obj2 = (d12 & ~d01)                     # roughly position at t=2 only
        if obj1.sum() < 30 or obj2.sum() < 30: continue
        n_ok += 1
        x1, x2 = norm(f1)[None].cuda(), norm(f2)[None].cuda()
        with torch.no_grad():
            ft1, ft2 = enc.dino(x1), enc.dino(x2)
            init = SA.sample_init(1, device=ft1.device, dtype=ft1.dtype)
            if args.mode == "carryover":
                s1 = SA(ft1, slots_init=init); s2 = SA(ft2, slots_init=s1.detach())
            else:
                s1 = SA(ft1, slots_init=init); s2 = SA(ft2, slots_init=init)
            _, a1 = enc.slot_decoder(s1); _, a2 = enc.slot_decoder(s2)
        K, N = a1.shape[1], a1.shape[2]; g = int(round(N**0.5))
        o1, o2 = pool(obj1, g).cuda(), pool(obj2, g).cuda()
        o1n, o2n = o1/(o1.sum()+1e-8), o2/(o2.sum()+1e-8)
        an1 = a1[0]/(a1[0].sum(-1,keepdim=True)+1e-8)
        an2 = a2[0]/(a2[0].sum(-1,keepdim=True)+1e-8)
        s_on = (an1*o1n).sum(-1)*N              # slot on object at t=1?
        # for slots that were on the object, did their alpha follow to obj2?
        sel = s_on > 2.0
        if sel.any():
            gained = ((an2[sel]*o2n).sum(-1) - (an1[sel]*o2n).sum(-1))*N
            follow += gained.tolist()
        on_obj += s_on.tolist()
        shift += (cen(a2[0],g)-cen(a1[0],g)).norm(dim=-1).tolist()
    on_obj, follow, shift = map(np.array,(on_obj,follow,shift))
    print(f"clips used = {n_ok}   slots = {len(on_obj)}   mode = {args.mode}")
    print()
    print("=== 有 slot 真的锁在物体上吗 (t=1) ===")
    print(f"  每帧最高 on-object 分数的分布: 见下 (1.0=chance)")
    print(f"  on-object 分数 > 2.0 的 slot 比例 = {(on_obj>2.0).mean():.3f}")
    print(f"  on-object 分数中位数 {np.median(on_obj):.3f}  p90 {np.quantile(on_obj,0.9):.3f}")
    print()
    print("=== 锁在物体上的 slot, 有跟着物体走吗 (t=1 -> t=2) ===")
    if len(follow):
        print(f"  n={len(follow)}   alpha 在新位置上的增益 = mean {follow.mean():+.3f}"
              f"  median {np.median(follow):+.3f}")
        print(f"  增益 > 0 的比例 = {(follow>0).mean():.3f}   [0.5 = 随机, >0.7 = 在跟踪]")
    else:
        print("  没有 slot 的 on-object 分数超过 2.0 -- 压根没有 slot 锁上物体")

if __name__ == "__main__":
    main()
