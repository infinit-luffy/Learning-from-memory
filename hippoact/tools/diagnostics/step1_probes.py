#!/usr/bin/env python3
"""TODO.md Step 1 -- three eval-only probes on a carryover-trained checkpoint.

P1  fresh init, single frame        -> is the encoder itself damaged, or is it
                                      purely the init distribution?
P2  moment-matched carryover init   -> decisive: does mapping prev slots back
                                      onto the init manifold restore
                                      localization while keeping identity?
P3  eval-time iters sweep 3/5/8     -> budget vs attractor (reference only)

Metric hygiene per TODO rule 4: object mask is the INTERSECTION
motion(t-1,t) & motion(t,t+1). Both the strict-intersection and the older
union numbers are reported, because the thresholds in TODO.md were calibrated
on union-mask measurements and the two scales differ.
"""
from __future__ import annotations
import argparse, glob
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from hippoact.utils.config import load_config
from validate_semantics import build_encoder


def pool(mask_bool, g, dev):
    return F.adaptive_avg_pool2d(
        torch.from_numpy(mask_bool.astype(np.float32))[None, None], (g, g)
    ).reshape(-1).to(dev)


def norm_dist(x):
    return x / (x.sum() + 1e-8)


def moment_match(prev, SA):
    """Map prev output slots back onto the init manifold (TODO Step1 P2)."""
    z = (prev - prev.mean(-1, keepdim=True)) / (prev.std(-1, keepdim=True) + 1e-6)
    return SA.slots_mu + SA.slots_logsigma.exp() * z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--n-clips", type=int, default=150)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    cfg = load_config(args.config); size = cfg.encoder.image_size
    enc = build_encoder(cfg).cuda()
    ck = torch.load(args.ckpt, map_location="cuda", weights_only=False)
    enc.load_state_dict(ck["encoder"], strict=False); enc.eval()
    SA = enc.slot_attn
    iters0 = SA.iters

    nm = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    base = transforms.Compose([transforms.Resize(size), transforms.CenterCrop(size),
                              transforms.ToTensor()])

    # regimes: name -> (needs_prev, init_fn)
    acc = {k: {"on_i": [], "on_u": [], "trk": []} for k in
           ("P1_fresh", "P2_moment", "raw_carryover")}
    sweep = {3: [], 5: [], 8: []}

    clips = sorted(glob.glob(args.data_dir + "/clip_*"))[: args.n_clips]
    used = 0
    for c in clips:
        fs = sorted(glob.glob(c + "/*.png"))
        if len(fs) < 3: continue
        f0, f1, f2 = [base(Image.open(f).convert("RGB")) for f in fs[:3]]
        d01 = ((f1-f0).abs().max(0).values > 0.02).numpy()
        d12 = ((f2-f1).abs().max(0).values > 0.02).numpy()
        obj1 = d01 & d12                      # strict: object at t=1
        obj2 = d12 & ~d01                     # roughly object at t=2
        if obj1.sum() < 30 or obj2.sum() < 30: continue
        used += 1
        x0, x1, x2 = (nm(f0)[None].cuda(), nm(f1)[None].cuda(), nm(f2)[None].cuda())
        with torch.no_grad():
            g0, g1, g2 = enc.dino(x0), enc.dino(x1), enc.dino(x2)
            N = g1.shape[1]; g = int(round(N ** 0.5))
            oi = norm_dist(pool(obj1, g, g1.device))
            ou = norm_dist(pool(d01, g, g1.device))     # union (old+new)
            o2 = norm_dist(pool(obj2, g, g1.device))

            init = SA.sample_init(1, device=g1.device, dtype=g1.dtype)
            s_prev = SA(g0, slots_init=init)

            regimes = {
                "P1_fresh":      SA(g1, slots_init=SA.sample_init(1, device=g1.device,
                                                                  dtype=g1.dtype)),
                "P2_moment":     SA(g1, slots_init=moment_match(s_prev.detach(), SA)),
                "raw_carryover": SA(g1, slots_init=s_prev.detach()),
            }
            for name, s1 in regimes.items():
                _, a1 = enc.slot_decoder(s1)
                an = a1[0] / (a1[0].sum(-1, keepdim=True) + 1e-8)
                on_i = (an * oi).sum(-1) * N
                on_u = (an * ou).sum(-1) * N
                acc[name]["on_i"].append((on_i > 2.0).float().mean().item())
                acc[name]["on_u"].append((on_u > 2.0).float().mean().item())
                # tracking: advance one more frame under the same regime
                if name == "P1_fresh":
                    s2 = SA(g2, slots_init=SA.sample_init(1, device=g1.device,
                                                          dtype=g1.dtype))
                elif name == "P2_moment":
                    s2 = SA(g2, slots_init=moment_match(s1.detach(), SA))
                else:
                    s2 = SA(g2, slots_init=s1.detach())
                _, a2 = enc.slot_decoder(s2)
                an2 = a2[0] / (a2[0].sum(-1, keepdim=True) + 1e-8)
                sel = on_i > 2.0
                if sel.any():
                    gained = ((an2[sel] * o2).sum(-1) - (an[sel] * o2).sum(-1)) * N
                    acc[name]["trk"] += (gained > 0).float().tolist()

            # P3: eval-time iters sweep, fresh init
            for it in (3, 5, 8):
                SA.iters = it
                s = SA(g1, slots_init=SA.sample_init(1, device=g1.device, dtype=g1.dtype))
                _, a = enc.slot_decoder(s)
                anx = a[0] / (a[0].sum(-1, keepdim=True) + 1e-8)
                sweep[it].append((((anx * oi).sum(-1) * N) > 2.0).float().mean().item())
            SA.iters = iters0

    print(f"{args.label or args.ckpt}   step={ck['step']}  clips_used={used}  "
          f"train_iters={iters0}")
    print()
    hdr = f"{'regime':<16}{'on-object(交集)':>16}{'on-object(并集)':>16}{'tracking':>12}"
    print(hdr); print("-" * len(hdr))
    for k in ("raw_carryover", "P1_fresh", "P2_moment"):
        d = acc[k]
        trk = np.mean(d["trk"]) if d["trk"] else float("nan")
        print(f"{k:<16}{np.mean(d['on_i']):>16.3f}{np.mean(d['on_u']):>16.3f}{trk:>12.3f}")
    print()
    print("P3 测试时 iters sweep (fresh init, on-object 交集):")
    for it in (3, 5, 8):
        print(f"  iters={it}: {np.mean(sweep[it]):.3f}")


if __name__ == "__main__":
    main()
