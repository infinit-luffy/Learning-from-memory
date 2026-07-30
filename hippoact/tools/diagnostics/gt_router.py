#!/usr/bin/env python3
"""The actual P1 test: does the router's slow/fast split match ground truth?

Requires --save-annotations. Uses exact geometry, so 'which slot sits on a
moving object' is known rather than inferred.

  objectness(k) = fraction of slot k's alpha mass landing on any disk,
                  normalized by the disks' chance share of the frame.
                  1.0 = chance, > 1 = object slot, < 1 = background slot.
  router says fast  <=> argmax(logits) == 1     (class 0 = slow by convention)

Reported:
  Cohen's d between objectness of fast-routed and slow-routed slots
  AUC of the router's fast-probability as a predictor of "is object slot"
  oracle / uniform controls, so a null result cannot be blamed on the metric
"""
from __future__ import annotations
import argparse, glob, json
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from hippoact.utils.config import load_config
from validate_semantics import build_encoder
from gt_eval import disk_masks, pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--n-clips", type=int, default=400)
    ap.add_argument("--mode", default="shared",
                    choices=["fresh", "shared", "carryover"])
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

    obj, is_fast, p_fast, orc, unif = [], [], [], [], []
    for c in sorted(glob.glob(args.data_dir + "/clip_*"))[: args.n_clips]:
        ann = json.loads(open(f"{c}/annotations.json").read())
        fs = sorted(glob.glob(c + "/*.png"))
        if len(fs) < 2:
            continue
        imgs = [base(Image.open(p).convert("RGB")) for p in fs[:2]]
        with torch.no_grad():
            g1 = enc.dino(nm(imgs[0])[None].cuda())
            if args.mode == "shared":
                init = SA.sample_init(1, device=g1.device, dtype=g1.dtype)
                s1 = SA(g1, slots_init=init)
            elif args.mode == "carryover":
                g0 = enc.dino(nm(imgs[1])[None].cuda())
                init = SA.sample_init(1, device=g1.device, dtype=g1.dtype)
                s1 = SA(g1, slots_init=SA(g0, slots_init=init).detach())
            else:
                s1 = SA(g1, slots_init=SA.sample_init(1, device=g1.device,
                                                      dtype=g1.dtype))
            _, a1 = enc.slot_decoder(s1)
            _, lg = enc.router(s1)
        K, N = a1.shape[1], a1.shape[2]
        g = int(round(N ** 0.5))
        A = a1[0] / (a1[0].sum(-1, keepdim=True) + 1e-8)
        dm = pool(disk_masks(ann["disks"][0], ann["size"]), g, A.device)
        anyd = dm.sum(0).clamp(max=1.0)                       # (N,) 任一圆盘
        share = anyd.sum() / N                                # 圆盘占画面比例
        o = (A @ anyd) / (share + 1e-8)                        # (K,) objectness
        obj += o.tolist()
        pr = lg[0].softmax(-1)[:, 1]
        p_fast += pr.tolist()
        is_fast += (lg[0].argmax(-1) == 1).float().tolist()
        # 对照：alpha 就是圆盘掩膜 / 均匀
        on = anyd / (anyd.sum() + 1e-8)
        orc.append(float((on @ anyd) / (share + 1e-8)))
        u = torch.full_like(anyd, 1.0 / N)
        unif.append(float((u @ anyd) / (share + 1e-8)))

    obj = np.array(obj); is_fast = np.array(is_fast); p_fast = np.array(p_fast)
    f, s = obj[is_fast == 1], obj[is_fast == 0]
    print(f"{args.label or args.ckpt}  step={ck['step']}  mode={args.mode}")
    print(f"  对照: oracle slot objectness = {np.mean(orc):.2f}   "
          f"uniform slot = {np.mean(unif):.2f}  (定义上应为 1.00)")
    print(f"  router 判 fast 的 slot 占比 = {is_fast.mean():.3f}")
    print(f"  objectness  FAST slots: n={len(f):5d} mean {f.mean() if len(f) else float('nan'):6.2f}"
          f"  median {np.median(f) if len(f) else float('nan'):6.2f}")
    print(f"  objectness  SLOW slots: n={len(s):5d} mean {s.mean() if len(s) else float('nan'):6.2f}"
          f"  median {np.median(s) if len(s) else float('nan'):6.2f}")
    if len(f) > 1 and len(s) > 1:
        pooled = np.sqrt((f.var(ddof=1) + s.var(ddof=1)) / 2)
        d = (f.mean() - s.mean()) / (pooled + 1e-12)
        r = obj.argsort().argsort().astype(float)
        auc = (r[is_fast == 1].mean() - r[is_fast == 0].mean()) / len(obj) + 0.5
        print(f"  Cohen's d = {d:+.3f}    秩 AUC = {auc:.3f}   [0.5 = 无分离]")
        hi = obj >= np.quantile(obj, 0.75)
        print(f"  router 的 P(fast) 对『真的是物体 slot』:  "
              f"物体 slot 上均值 {p_fast[hi].mean():.3f}  "
              f"非物体 slot 上均值 {p_fast[~hi].mean():.3f}")


if __name__ == "__main__":
    main()
