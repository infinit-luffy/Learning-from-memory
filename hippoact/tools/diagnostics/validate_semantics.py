#!/usr/bin/env python3
"""Decisive test for the cp5b fix: does the router's slow/fast split actually
track real motion, or is it just satisfying the quantile tautology?

slow_ratio ~= prior_slow is guaranteed by construction (the CE target is a
quantile threshold), so it cannot validate semantics. Here we use the
synthetic ground truth instead:

  1. motion mask  = |frame_t - frame_prev| per pixel, pooled to the 16x16
                    patch grid that DINOv2 actually sees.
  2. motion score = alpha-weighted mean of that mask, per slot.
  3. compare motion score of router-fast slots vs router-slow slots.

If the pair loader + slot matching fix is semantically real, fast slots must
score significantly higher than slow slots. If it is hollow, the two
distributions overlap.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from hippoact.utils.config import load_config


def build_encoder(cfg):
    from hippoact.encoders.hippo_encoder import HippoActEncoder
    e = cfg.encoder
    return HippoActEncoder(
        num_slots=e.num_slots, slot_dim=e.slot_dim, proprio_dim=e.proprio_dim,
        c_dim=e.c_dim, t_window=e.t_window, image_size=e.image_size,
        patch_size=e.patch_size, dino_model=e.dino_model,
        binding_layers=e.binding_layers, binding_heads=e.binding_heads,
        binding_dropout=e.binding_dropout, slot_iters=e.slot_iters,
        slot_hidden=e.slot_hidden, router_hidden=e.router_hidden,
        gumbel_tau_init=e.gumbel_tau_init,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--n-clips", type=int, default=100)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = load_config(args.config)
    size = cfg.encoder.image_size
    dev = args.device

    enc = build_encoder(cfg).to(dev)
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    enc.load_state_dict(ck["encoder"], strict=False)
    enc.eval()
    print(f"checkpoint step={ck['step']}  gumbel_tau={ck['gumbel_tau']:.4f}")

    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    base = transforms.Compose([
        transforms.Resize(size), transforms.CenterCrop(size), transforms.ToTensor(),
    ])

    clips = sorted(glob.glob(str(Path(args.data_dir) / "clip_*")))[: args.n_clips]
    fast_scores, slow_scores, all_scores = [], [], []
    n_fast = n_slow = 0

    for c in clips:
        fs = sorted(glob.glob(c + "/*.png"))
        if len(fs) < 2:
            continue
        prev = base(Image.open(fs[0]).convert("RGB"))
        cur = base(Image.open(fs[1]).convert("RGB"))

        # ---- ground-truth motion, pooled to the patch grid ----
        motion = (cur - prev).abs().max(dim=0).values.unsqueeze(0).unsqueeze(0)
        x = norm(cur).unsqueeze(0).to(dev)
        with torch.no_grad():
            feats = enc.dino(x)
            slots = enc.slot_attn(feats)
            _recon, alpha = enc.slot_decoder(slots)      # (1,K,N)
            _g, logits = enc.router(slots)               # (1,K,2)

        K, N = alpha.shape[1], alpha.shape[2]
        g = int(round(N ** 0.5))
        m = F.adaptive_avg_pool2d(motion, (g, g)).reshape(1, N).to(dev)
        m = m / (m.sum() + 1e-8)                          # normalize to a distribution

        a = alpha[0]                                      # (K,N)
        a = a / (a.sum(dim=-1, keepdim=True) + 1e-8)
        score = (a * m).sum(dim=-1) * N                   # (K,) 1.0 == uniform coverage

        is_slow = (logits[0].argmax(dim=-1) == 0)         # class 0 == slow
        fast_scores += score[~is_slow].tolist()
        slow_scores += score[is_slow].tolist()
        all_scores += score.tolist()
        n_slow += int(is_slow.sum()); n_fast += int((~is_slow).sum())

    fast = np.array(fast_scores); slow = np.array(slow_scores)
    print()
    print(f"slots examined: slow={n_slow}  fast={n_fast}  "
          f"(slow_ratio={n_slow / max(1, n_slow + n_fast):.3f})")
    print()
    print("motion score  (1.0 = slot attends motion exactly as much as chance;")
    print("               >1 = concentrated on moving pixels, <1 = on static bg)")
    print(f"  router-FAST slots : mean {fast.mean():.3f}  median {np.median(fast):.3f}"
          f"  std {fast.std():.3f}")
    print(f"  router-SLOW slots : mean {slow.mean():.3f}  median {np.median(slow):.3f}"
          f"  std {slow.std():.3f}")
    print()

    if len(fast) > 1 and len(slow) > 1:
        pooled = np.sqrt((fast.var(ddof=1) + slow.var(ddof=1)) / 2)
        d = (fast.mean() - slow.mean()) / (pooled + 1e-12)
        # Welch t
        t = (fast.mean() - slow.mean()) / np.sqrt(
            fast.var(ddof=1) / len(fast) + slow.var(ddof=1) / len(slow) + 1e-12
        )
        print(f"  Cohen's d = {d:+.3f}   Welch t = {t:+.2f}")
        print()
        if d > 0.5:
            print("  => VERDICT: fast slots really do track motion. Fix is semantically real.")
        elif d > 0.2:
            print("  => VERDICT: weak but present separation. Signal exists, undertrained.")
        else:
            print("  => VERDICT: NO separation. slow/fast split is still semantically hollow,")
            print("     even though slow_ratio looks healthy (it is a tautology).")


if __name__ == "__main__":
    main()
