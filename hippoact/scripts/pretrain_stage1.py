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
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from hippoact.encoders.hippo_encoder import HippoActEncoder
from hippoact.training.stage1 import Stage1Config, Stage1Trainer
from hippoact.utils.config import load_config


class ImageDirDataset(Dataset):
    """Simple recursive image-directory dataset."""

    IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

    def __init__(self, root: str | Path, image_size: int = 224):
        self.paths = sorted(
            [p for p in Path(root).rglob("*") if p.suffix.lower() in self.IMG_EXT]
        )
        if not self.paths:
            raise ValueError(f"No images found under {root}")
        # DINOv2 expects ImageNet-normalized RGB.
        self.tf = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict:
        img = Image.open(self.paths[idx]).convert("RGB")
        return {"img": self.tf(img)}


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
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--data-dir", type=str, required=True)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--run-name", type=str, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    encoder = build_encoder(cfg)

    dataset = ImageDirDataset(args.data_dir, image_size=cfg.encoder.image_size)
    loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

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
        out_dir=cfg.log.out_dir + "/stage1",
        device=cfg.train.device,
        use_wandb=args.wandb,
        wandb_project=cfg.log.wandb_project,
        wandb_entity=cfg.log.wandb_entity,
        run_name=args.run_name,
    )

    trainer = Stage1Trainer(encoder, stage1_cfg)
    print(f"[Stage1] dataset: {len(dataset)} frames | steps: {stage1_cfg.steps}")
    print(f"[Stage1] DINOv2 mock={encoder.dino.is_mock}")
    trainer.fit(loader)


if __name__ == "__main__":
    main()
