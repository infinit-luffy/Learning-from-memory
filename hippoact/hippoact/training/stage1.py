"""Stage-1: unsupervised representation pretraining.

Trains SlotAttention + SlotDecoder + SlotRouter on top of the frozen DINOv2
backbone. No proprioception, no reward, no environment interaction. Data is a
plain image loader — synthetic, teleop, or free-exploration frames all work.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import DataLoader

from hippoact.encoders.hippo_encoder import HippoActEncoder
from hippoact.losses import (
    route_prior_kl,
    slot_diversity_loss,
    slot_reconstruction_loss,
    slow_temporal_loss,
)


@dataclass
class Stage1Config:
    steps: int = 100_000
    lr: float = 3.0e-4
    weight_decay: float = 1.0e-4
    warmup_steps: int = 2000
    grad_clip: float = 20.0
    lambda_slow: float = 0.5
    lambda_route: float = 0.05
    lambda_div: float = 0.05
    route_prior_slow: float = 0.7
    tau_anneal_factor: float = 0.9995
    tau_min: float = 0.3
    log_every: int = 50
    ckpt_every: int = 10_000
    viz_every: int = 5000        # dump slot alpha overlays every N steps
    out_dir: str = "outputs/stage1"
    device: str = "cuda"
    use_wandb: bool = False
    wandb_project: str = "hippoact"
    wandb_entity: str | None = None
    run_name: str | None = None
    extras: dict = field(default_factory=dict)


class Stage1Trainer:
    def __init__(self, encoder: HippoActEncoder, cfg: Stage1Config):
        self.enc = encoder
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.enc.to(self.device)

        params = [
            {"params": list(encoder.slot_attn.parameters()), "lr": cfg.lr},
            {"params": list(encoder.slot_decoder.parameters()), "lr": cfg.lr},
            {"params": list(encoder.router.parameters()), "lr": cfg.lr},
        ]
        self.optim = torch.optim.AdamW(params, weight_decay=cfg.weight_decay)

        self._wandb = None
        if cfg.use_wandb:
            try:
                import wandb
                wandb.init(
                    project=cfg.wandb_project,
                    entity=cfg.wandb_entity,
                    name=cfg.run_name,
                    config=cfg.__dict__,
                )
                self._wandb = wandb
            except Exception as e:  # noqa: BLE001
                print(f"[Stage1] wandb disabled ({e}).")

        self.out_dir = Path(cfg.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------

    def _lr_at(self, step: int) -> float:
        if step < self.cfg.warmup_steps:
            return self.cfg.lr * (step + 1) / max(1, self.cfg.warmup_steps)
        return self.cfg.lr

    def _step(self, imgs: torch.Tensor, prev_slots: torch.Tensor | None):
        """One optimizer step. Return (total loss, dict of components, slots, alpha)."""
        with torch.no_grad():
            target = self.enc.dino(imgs)                        # (B, N, D)
        slots = self.enc.slot_attn(target)                      # (B, K, D_s)
        recon, alpha = self.enc.slot_decoder(slots)             # alpha: (B, K, N)
        g, logits = self.enc.router(slots)                      # g: (B, K, 2)
        slow_mask = g[..., 0]                                    # (B, K)

        losses = {
            "L_slot":   slot_reconstruction_loss(recon, target),
            "L_route":  route_prior_kl(logits, prior_slow=self.cfg.route_prior_slow),
            "L_div":    slot_diversity_loss(slots),
        }
        if prev_slots is not None:
            # New: pass router logits, not the (gamable) slow_mask.
            losses["L_slow"] = slow_temporal_loss(
                slots, prev_slots, logits, prior_slow=self.cfg.route_prior_slow
            )
        else:
            losses["L_slow"] = slots.new_zeros(())
        # Router health metric — the observed slow ratio in the batch.
        losses["slow_ratio"] = slow_mask.mean().detach()

        loss = (
            losses["L_slot"]
            + self.cfg.lambda_slow  * losses["L_slow"]
            + self.cfg.lambda_route * losses["L_route"]
            + self.cfg.lambda_div   * losses["L_div"]
        )
        return loss, losses, slots, alpha

    def fit(self, loader: Iterable[dict]) -> None:
        """Loader yields batches with key ``img`` — a (B,3,H,W) float tensor in [0,1]."""
        self.enc.train()
        step = 0
        prev_slots: torch.Tensor | None = None
        t0 = time.time()

        # Endlessly cycle the loader.
        loader_iter = iter(loader)

        while step < self.cfg.steps:
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                batch = next(loader_iter)

            imgs = batch["img"].to(self.device, non_blocking=True)

            # LR warmup.
            lr_now = self._lr_at(step)
            for pg in self.optim.param_groups:
                pg["lr"] = lr_now

            loss, components, slots, alpha = self._step(imgs, prev_slots)

            self.optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                (p for _, p in self.enc.trainable_parameters()),
                max_norm=self.cfg.grad_clip,
            )
            self.optim.step()

            # Anneal Gumbel temperature every step.
            tau_now = self.enc.anneal_router(
                factor=self.cfg.tau_anneal_factor, tau_min=self.cfg.tau_min
            )
            prev_slots = slots.detach()

            if step % self.cfg.log_every == 0:
                self._log(step, components, tau_now, lr_now, t0)
            if step and step % self.cfg.ckpt_every == 0:
                self._save_ckpt(step)
            if step and step % self.cfg.viz_every == 0:
                self._save_slot_viz(step, imgs, alpha)

            step += 1

        self._save_ckpt(step, final=True)
        if self._wandb is not None:
            self._wandb.finish()

    # ------------------------------------------------------------------

    def _log(self, step: int, components: dict, tau: float, lr: float, t0: float) -> None:
        elapsed = time.time() - t0
        summary = {
            "step": step,
            "elapsed_s": elapsed,
            "lr": lr,
            "gumbel_tau": tau,
            **{k: float(v.item()) for k, v in components.items()},
        }
        msg = " | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                        for k, v in summary.items())
        print(msg)
        if self._wandb is not None:
            self._wandb.log(summary, step=step)

    # ------------------------------------------------------------------

    # ImageNet stats used by the Stage-1 DataLoader.
    _NORM_MEAN = (0.485, 0.456, 0.406)
    _NORM_STD  = (0.229, 0.224, 0.225)

    def _save_slot_viz(self, step: int, imgs: torch.Tensor, alpha: torch.Tensor) -> None:
        """Dump a slot-alpha overlay grid for the first frame in the batch.

        Import viz lazily so a headless run without matplotlib still trains.
        """
        try:
            from hippoact.utils.viz import save_slot_grid
        except Exception as e:  # noqa: BLE001
            print(f"[Stage1] viz disabled ({e}).")
            return
        mean = torch.tensor(self._NORM_MEAN, device=imgs.device).view(3, 1, 1)
        std  = torch.tensor(self._NORM_STD,  device=imgs.device).view(3, 1, 1)
        img0  = (imgs[0].detach() * std + mean).clamp(0, 1).cpu()      # (3, H, W)
        a0    = alpha[0].detach().cpu()                                # (K, N)
        out_p = self.out_dir / f"slots_step{step:07d}.png"
        save_slot_grid(img0, a0, out_p)

    def _save_ckpt(self, step: int, final: bool = False) -> None:
        name = f"ckpt_final.pt" if final else f"ckpt_step{step:07d}.pt"
        p = self.out_dir / name
        payload = {
            "step": step,
            "encoder": self.enc.state_dict(),
            "optim": self.optim.state_dict(),
            "gumbel_tau": float(self.enc.router.tau.item()),
        }
        torch.save(payload, p)
        print(f"[Stage1] saved checkpoint → {p}")
