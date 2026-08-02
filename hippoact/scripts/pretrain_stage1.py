#!/usr/bin/env python3
"""Stage-1 pretraining entry point.

Usage:
    python scripts/pretrain_stage1.py --config configs/default.yaml --data-dir data/frames

Data layout (default ImageFolder-style):
    data/frames/
        session_0/*.png
        session_1/*.png
        ...

Any RGB image is fine — teleop demos, random arm motion, static scenes.
Aim for ~50-100K diverse frames for Stage 1.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from hippoact.encoders.hippo_encoder import HippoActEncoder
from hippoact.training.stage1 import Stage1Config, Stage1Trainer
from hippoact.utils.config import load_config


IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _default_transform(image_size: int):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ])


class ImageDirDataset(Dataset):
    """Flat image directory. Yields {'img': ...}. Legacy for smoke testing.

    Emits a semantically weak L_slow signal because consecutive iterations see
    unrelated shuffled frames. Prefer ``ImageClipPairDataset`` for real runs.
    """

    def __init__(self, root: str | Path, image_size: int = 224):
        self.paths = sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in IMG_EXT)
        if not self.paths:
            raise ValueError(f"No images found under {root}")
        self.tf = _default_transform(image_size)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict:
        return {"img": self.tf(Image.open(self.paths[idx]).convert("RGB"))}


class ImageClipPairDataset(Dataset):
    """Yield (previous, current) frame pairs from clip subdirectories.

    Root layout:
        root/
            clip_000/frame_0000.png, frame_0001.png, ...
            clip_001/...

    Each subdirectory is one clip; frames are sorted lexicographically. A pair
    (frame_i, frame_{i+gap}) is emitted for each valid i in each clip.
    """

    def __init__(self, root: str | Path, image_size: int = 224, gap: int = 1):
        self.gap = gap
        self.pairs: list[tuple[Path, Path]] = []
        for clip_dir in sorted(Path(root).iterdir()):
            if not clip_dir.is_dir():
                continue
            frames = sorted(
                p for p in clip_dir.iterdir() if p.suffix.lower() in IMG_EXT
            )
            for i in range(len(frames) - gap):
                self.pairs.append((frames[i], frames[i + gap]))
        if not self.pairs:
            raise ValueError(
                f"No frame pairs found under {root}. "
                f"Expected subdirectories, each with ≥ {gap + 1} frames."
            )
        self.tf = _default_transform(image_size)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        p_prev, p_cur = self.pairs[idx]
        return {
            "img_prev": self.tf(Image.open(p_prev).convert("RGB")),
            "img_t":    self.tf(Image.open(p_cur).convert("RGB")),
        }


def _looks_like_clip_layout(root: Path) -> bool:
    """Return True iff root has at least one subdirectory that itself has 2+
    image files. Otherwise fall back to flat mode.
    """
    for p in root.iterdir():
        if p.is_dir():
            imgs = [q for q in p.iterdir() if q.suffix.lower() in IMG_EXT]
            if len(imgs) >= 2:
                return True
    return False


def build_encoder(cfg) -> HippoActEncoder:
    e = cfg.encoder
    return HippoActEncoder(
        num_slots=e.num_slots,
        slot_dim=e.slot_dim,
        proprio_dim=e.proprio_dim,
        c_dim=e.c_dim,
        t_window=e.t_window,
        image_size=e.image_size,
        patch_size=e.patch_size,
        dino_model=e.dino_model,
        binding_layers=e.binding_layers,
        binding_heads=e.binding_heads,
        binding_dropout=e.binding_dropout,
        slot_iters=e.slot_iters,
        slot_hidden=e.slot_hidden,
        router_hidden=e.router_hidden,
        gumbel_tau_init=e.gumbel_tau_init,
        slot_query_mode=str(e.get("slot_query_mode", "sampled")),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--data-dir", type=str, required=True)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--run-name", type=str, default=None)
    ap.add_argument("--force-flat", action="store_true",
                    help="Force legacy flat loader even if clip layout detected")
    ap.add_argument("--pair-gap", type=int, default=1,
                    help="For clip loader: frame gap between (prev, cur)")
    ap.add_argument("--seed", type=int, default=None,
                    help="Override train.seed; use for Stage-1 variance sweeps")
    ap.add_argument("--viz-every", type=int, default=None,
                    help="Override log.viz_every from config")
    args = ap.parse_args()

    cfg = load_config(args.config)

    # Stage-1 was previously unseeded: `train.seed` sat in every config but was
    # never read, so each run was an independent draw and no run could be
    # reproduced. Seed before building the encoder — slot query init, the
    # decoder, and the router are all sampled at construction time.
    # `--seed` overrides the config so a variance sweep needs no config edits.
    seed = args.seed if args.seed is not None else int(cfg.train.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"[Stage1] seed = {seed}")

    encoder = build_encoder(cfg)

    root = Path(args.data_dir)
    if not args.force_flat and _looks_like_clip_layout(root):
        dataset = ImageClipPairDataset(root, image_size=cfg.encoder.image_size,
                                       gap=args.pair_gap)
        print(f"[Stage1] clip layout detected → paired loader "
              f"({len(dataset)} (prev,cur) pairs, gap={args.pair_gap})")
    else:
        dataset = ImageDirDataset(root, image_size=cfg.encoder.image_size)
        print(f"[Stage1] flat layout → LEGACY loader ({len(dataset)} frames). "
              "L_slow signal will be semantically weak; see README §4.1.")

    loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    viz_every = args.viz_every if args.viz_every is not None \
                else int(cfg.log.get("viz_every", 5000))

    stage1_cfg = Stage1Config(
        steps=cfg.train.stage1_steps,
        lr=cfg.train.stage1_lr,
        weight_decay=cfg.train.stage1_wd,
        warmup_steps=cfg.train.warmup_steps,
        grad_clip=cfg.train.grad_clip,
        lambda_slow=cfg.loss.lambda_slow,
        lambda_route=cfg.loss.lambda_route,
        lambda_div=cfg.loss.lambda_div,
        route_prior_slow=cfg.encoder.route_prior_slow,
        tau_anneal_factor=cfg.encoder.gumbel_tau_decay,
        tau_min=cfg.encoder.gumbel_tau_min,
        log_every=cfg.log.log_every,
        ckpt_every=cfg.log.ckpt_every,
        viz_every=viz_every,
        slow_signal=str(cfg.loss.get("slow_signal", "content_diff")),
        slow_variant=str(cfg.loss.get("slow_variant", "soft_bce")),
        slow_temperature=float(cfg.loss.get("slow_temperature", 1.0)),
        slot_init_mode=str(cfg.train.get("slot_init_mode", "shared")),
        out_dir=cfg.log.out_dir + "/stage1",
        device=cfg.train.device,
        use_wandb=args.wandb,
        wandb_project=cfg.log.wandb_project,
        wandb_entity=cfg.log.wandb_entity,
        run_name=args.run_name,
    )

    trainer = Stage1Trainer(encoder, stage1_cfg)
    print(f"[Stage1] steps: {stage1_cfg.steps} | viz_every: {stage1_cfg.viz_every} "
          f"| DINOv2 mock={encoder.dino.is_mock}")
    trainer.fit(loader)


if __name__ == "__main__":
    main()
