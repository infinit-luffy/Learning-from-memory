#!/usr/bin/env python3
"""Post-hoc slot-alpha visualization from a Stage-1 checkpoint.

Lives outside the repo on purpose: the trainer never calls save_slot_grid(),
so this reproduces what README 4.3 claims is dumped to outputs/stage1/.

Works around the viz.py default-arg bug: overlay_slot_alpha defaults to a
14x14=196 grid, but DINOv2 at image_size=224/patch=14 emits 16x16=256 tokens.
We pass grid_h/grid_w=16 and upsample=14 (16*14=224) explicitly.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hippoact.utils.config import load_config              # noqa: E402
from hippoact.utils.viz import overlay_slot_alpha          # noqa: E402


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
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-frames", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = load_config(args.config)
    size = cfg.encoder.image_size

    enc = build_encoder(cfg).to(args.device)
    payload = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    missing, unexpected = enc.load_state_dict(payload["encoder"], strict=False)
    print(f"loaded step={payload['step']} tau={payload['gumbel_tau']:.4f}")
    print(f"missing={len(missing)} unexpected={len(unexpected)}")
    enc.eval()

    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    base_tf = transforms.Compose([
        transforms.Resize(size), transforms.CenterCrop(size), transforms.ToTensor(),
    ])

    paths = sorted(Path(args.data_dir).rglob("*.png"))
    step = max(1, len(paths) // args.n_frames)
    picks = paths[::step][:args.n_frames]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for i, p in enumerate(picks):
        raw = base_tf(Image.open(p).convert("RGB"))          # (3,H,W) in [0,1]
        x = norm(raw).unsqueeze(0).to(args.device)
        with torch.no_grad():
            feats = enc.dino(x)
            slots = enc.slot_attn(feats)
            _recon, alpha = enc.slot_decoder(slots)          # alpha (1,K,N)
        a = alpha[0].cpu()
        K, N = a.shape
        g = int(round(N ** 0.5))
        assert g * g == N, f"non-square token grid: N={N}"
        ov = overlay_slot_alpha(raw, a, grid_h=g, grid_w=g, upsample=size // g)

        # how peaked is each slot's mask -> proxy for "clean segmentation"
        an = a / (a.sum(dim=-1, keepdim=True) + 1e-8)
        ent = -(an * (an + 1e-8).log()).sum(-1) / np.log(N)
        active = int((a.max(dim=-1).values > 0.5 * a.max()).sum())
        print(f"{p.name}: K={K} N={N} grid={g}x{g} "
              f"mean_norm_entropy={ent.mean():.3f} (1.0=uniform, 0=one patch) "
              f"active_slots~{active}")

        cols, rows = 8, (K + 7) // 8
        fig, axes = plt.subplots(rows, cols + 1, figsize=((cols + 1) * 2, rows * 2))
        axes[0, 0].imshow(raw.permute(1, 2, 0).numpy())
        axes[0, 0].set_title("input", fontsize=9)
        for r in range(rows):
            axes[r, 0].axis("off")
        for k in range(K):
            r, c = divmod(k, cols)
            ax = axes[r, c + 1]
            ax.imshow(ov[k]); ax.set_title(f"slot {k}", fontsize=8); ax.axis("off")
        for k in range(K, rows * cols):
            r, c = divmod(k, cols)
            axes[r, c + 1].axis("off")
        fig.tight_layout()
        f = out_dir / f"slots_frame{i}_{p.stem}.png"
        fig.savefig(f, dpi=90, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> {f}")


if __name__ == "__main__":
    main()
