#!/usr/bin/env python3
"""Does successful tracking CAUSE content diff to go down?

Split slots by centroid shift. If tracking slots (large shift, following their
object) have lower content diff than still slots, then the negative
corr(content diff, motion) that appears as binding forms is explained by the
concept hole: stable tracking => stable representation. That rules out content
diff as a target on principle, not just empirically.
"""
from __future__ import annotations
import argparse, glob
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from hippoact.utils.config import load_config
from validate_semantics import build_encoder

def centroids(a, g):
    an = a / (a.sum(-1, keepdim=True) + 1e-8)
    ys, xs = torch.meshgrid(
        torch.arange(g, device=a.device, dtype=torch.float),
        torch.arange(g, device=a.device, dtype=torch.float), indexing="ij")
    return torch.stack([(an*ys.reshape(-1)).sum(-1), (an*xs.reshape(-1)).sum(-1)], -1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--n-clips", type=int, default=150)
    args = ap.parse_args()
    cfg = load_config(args.config); size = cfg.encoder.image_size
    enc = build_encoder(cfg).cuda()
    ck = torch.load(args.ckpt, map_location="cuda", weights_only=False)
    enc.load_state_dict(ck["encoder"], strict=False); enc.eval()
    SA = enc.slot_attn
    norm = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    base = transforms.Compose([transforms.Resize(size), transforms.CenterCrop(size),
                               transforms.ToTensor()])
    CS, CD, MS = [], [], []
    for c in sorted(glob.glob(args.data_dir + "/clip_*"))[:args.n_clips]:
        fs = sorted(glob.glob(c + "/*.png"))
        prev, cur = base(Image.open(fs[0]).convert("RGB")), base(Image.open(fs[1]).convert("RGB"))
        motion = (cur-prev).abs().max(0).values[None,None]
        xp, xc = norm(prev)[None].cuda(), norm(cur)[None].cuda()
        with torch.no_grad():
            fp, fc = enc.dino(xp), enc.dino(xc)
            init = SA.sample_init(1, device=fc.device, dtype=fc.dtype)
            sp = SA(fp, slots_init=init)
            sc = SA(fc, slots_init=sp.detach())
            _, a_cur = enc.slot_decoder(sc)
            _, a_prv = enc.slot_decoder(sp)
        K, N = a_cur.shape[1], a_cur.shape[2]; g = int(round(N**0.5))
        CS += (centroids(a_cur[0],g)-centroids(a_prv[0],g)).norm(dim=-1).tolist()
        CD += (sc-sp).pow(2).sum(-1)[0].tolist()
        m = F.adaptive_avg_pool2d(motion,(g,g)).reshape(1,N).cuda(); mn = m/(m.sum()+1e-8)
        an = a_cur[0]/(a_cur[0].sum(-1,keepdim=True)+1e-8)
        MS += ((an*mn).sum(-1)*N).tolist()
    CS, CD, MS = map(np.array,(CS,CD,MS))
    print(f"ckpt step={ck['step']}   slots={len(CS)}")
    print()
    track, still = CS > 1.5, CS < 0.3
    print("按 centroid shift 分组:")
    print(f"  跟踪型 (shift>1.5 patch)  n={track.sum():5d}  content diff 中位数 {np.median(CD[track]):8.3f}  motion score 中位数 {np.median(MS[track]):.3f}")
    print(f"  静止型 (shift<0.3 patch)  n={still.sum():5d}  content diff 中位数 {np.median(CD[still]):8.3f}  motion score 中位数 {np.median(MS[still]):.3f}")
    print()
    if track.sum() > 5 and still.sum() > 5:
        r = np.median(CD[track])/max(np.median(CD[still]),1e-9)
        print(f"  跟踪型/静止型 content diff 比值 = {r:.3f}")
        if r < 1.0:
            print("  => 跟踪型 content diff 更低: 概念漏洞成立")
            print("     跟踪成功 => 表征稳定 => content diff 不能代表『覆盖运动』")
        else:
            print("  => 跟踪型 content diff 更高: 此 checkpoint 上概念漏洞不成立")
    print()
    print("两两相关 (log 空间):")
    pairs = [("centroid shift",CS),("content diff",CD),("motion score",MS)]
    for i in range(len(pairs)):
        for j in range(i+1,len(pairs)):
            na,a = pairs[i]; nb,b = pairs[j]
            print(f"  corr({na:<15}, {nb:<15}) = {np.corrcoef(np.log(a+1e-9),np.log(b+1e-9))[0,1]:+.3f}")

if __name__ == "__main__":
    main()
