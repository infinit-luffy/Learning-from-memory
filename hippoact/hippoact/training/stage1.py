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
    slow_temporal_loss_soft,
    slow_connectivity_loss,
)
from hippoact.utils.slot_matching import match_slots_nn


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
    viz_every: int = 5000
    # Slow-temporal-loss variant. "soft_bce" (default) = quantile min-max
    # sigmoid target BCE; "quantile_ce" = old quantile-thresholded CE (kept
    # for A/B).
    slow_variant: str = "soft_bce"
    slow_temperature: float = 1.0
    # Routing signal for L_slow.
    #   "content_diff"      : temporal variance of the slot vector (legacy).
    #   "alpha_connectivity": spatial connectivity of the slot alpha map.
    # content_diff is dominated by slot drift rather than world motion:
    # background slots wander (no unique assignment over 95% of the frame)
    # while an object slot that tracks its object stays stable, so the
    # router learns the inverted split -- Cohen's d = -1.25 against exact
    # ground-truth masks. alpha_connectivity is a per-frame property and
    # needs no cross-frame slot identity at all (CP6: d -1.25 -> +1.89).
    slow_signal: str = "content_diff"
    # How to initialize slot queries across the paired forward passes.
    #  "random"    — independent fresh sample per frame (baseline)
    #  "shared"    — one fresh sample used for both frames (kills init noise
    #                but also freezes spatial partition — CP5d finding)
    #  "carryover" — fresh init for prev, prev's *output* as init for cur.
    #                SAVi-style, restores cross-frame binding.
    slot_init_mode: str = "shared"
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

        # === CP5e/f consistency warning ===
        # Under carryover, tracking slots have LOW content-diff (features are
        # stable — same object at a new spatial position) while drifting or
        # re-binding slots have HIGH content-diff. Consequently, using content
        # diff as the L_slow target under carryover *inverts* the semantics
        # the router should learn (measured corr(diff, motion) = -0.379 on
        # CP5e). The proper next-step fix is an alpha-displacement target.
        # Until then, set lambda_slow=0 under carryover to avoid teaching the
        # router the wrong mapping.
        if cfg.slot_init_mode in ("carryover", "carryover_norm") and cfg.lambda_slow > 0:
            print(
                "[Stage1] WARNING: slot_init_mode=carryover with "
                f"lambda_slow={cfg.lambda_slow} > 0. Content-diff target is "
                "anti-signal under carryover (see CP5e diagnostic). Strongly "
                "recommend lambda_slow=0 for slot_iters ablation; wait for "
                "alpha-displacement target patch before re-enabling."
            )

    # ------------------------------------------------------------------

    def _lr_at(self, step: int) -> float:
        if step < self.cfg.warmup_steps:
            return self.cfg.lr * (step + 1) / max(1, self.cfg.warmup_steps)
        return self.cfg.lr

    def _step(self, batch: dict, prev_slots_xiter: torch.Tensor | None):
        """One optimizer step. Return (total loss, dict of components, slots, alpha).

        Accepts two batch formats:
          * paired  {"img_t": ..., "img_prev": ...}  — proper temporal supervision;
                    slots at t and t-gap are matched via NN before the slow loss.
          * legacy  {"img": ...}                     — falls back to using the
                    slots from the previous iteration as ``slots_prev``. This is
                    semantically weak (unrelated images across iterations) and
                    is only kept for backward compatibility. A warning is issued
                    once per training run.
        """
        paired = "img_t" in batch and "img_prev" in batch
        if paired:
            imgs_t   = batch["img_t"].to(self.device, non_blocking=True)
            imgs_prev = batch["img_prev"].to(self.device, non_blocking=True)
            with torch.no_grad():
                target_t    = self.enc.dino(imgs_t)
                target_prev = self.enc.dino(imgs_prev)
            B = target_t.shape[0]

            mode = self.cfg.slot_init_mode
            if mode == "random":
                # CP5c baseline (before shared-init fix): each frame gets an
                # independent stochastic slot init. ~87 % of diff will be
                # init noise; kept only for A/B.
                slots_prev = self.enc.slot_attn(target_prev)
                slots      = self.enc.slot_attn(target_t)
            elif mode == "shared":
                # CP5c fix: pair-wide shared init eliminates noise floor but
                # freezes the spatial partition — slots do NOT track objects
                # across frames (CP5d finding: alpha centroid moves 0.4 patch
                # while objects move 2.3 patch).
                init = self.enc.slot_attn.sample_init(
                    B, device=target_t.device, dtype=target_t.dtype
                )
                slots_prev = self.enc.slot_attn(target_prev, slots_init=init)
                slots      = self.enc.slot_attn(target_t,    slots_init=init)
            elif mode == "carryover":
                # SAVi-style: previous frame's OUTPUT slots become the init
                # for the current frame's forward pass. Iterations then
                # update slots to fit the new image starting from a state
                # already fit to the previous image → cross-frame binding.
                # Detach so gradients don't chain through two forward passes.
                init = self.enc.slot_attn.sample_init(
                    B, device=target_t.device, dtype=target_t.dtype
                )
                slots_prev = self.enc.slot_attn(target_prev, slots_init=init)
                slots      = self.enc.slot_attn(
                    target_t, slots_init=slots_prev.detach()
                )
            elif mode == "carryover_norm":
                # Moment-matched carryover (CP5g). Raw carryover feeds
                # converged slots (norm ~1.3x the init distribution, mutually
                # distant) as init, which preserves slot identity but removes
                # the near-symmetric-init statistics that drive attention
                # competition — the emergent engine behind localization.
                # Fix: re-standardize each carried slot per-dim and map it
                # back onto the learned init manifold N(mu, sigma), keeping
                # its direction (identity) while restoring the statistics
                # that make slots compete for patches again.
                init = self.enc.slot_attn.sample_init(
                    B, device=target_t.device, dtype=target_t.dtype
                )
                slots_prev = self.enc.slot_attn(target_prev, slots_init=init)
                prev_d = slots_prev.detach()
                z = (prev_d - prev_d.mean(-1, keepdim=True)) / (
                    prev_d.std(-1, keepdim=True) + 1e-6
                )
                mu = self.enc.slot_attn.slots_mu          # (1, 1, D)
                sigma = self.enc.slot_attn.slots_logsigma.exp()
                carried_init = (mu + sigma * z).detach()
                slots = self.enc.slot_attn(target_t, slots_init=carried_init)
            else:
                raise ValueError(f"unknown slot_init_mode: {mode}")

            # Under carryover(-norm), slot k in cur is constructed FROM slot
            # k in prev (prev's output — possibly re-standardized — is cur's
            # init). Index correspondence is exact-by-construction, so cosine
            # NN matching would only scramble the alignments. Skip.
            if mode in ("carryover", "carryover_norm"):
                prev_for_slow = slots_prev.detach()
            else:
                prev_for_slow = match_slots_nn(slots.detach(), slots_prev.detach())
            target = target_t
        else:
            if not getattr(self, "_warned_flat", False):
                print("[Stage1] WARNING: legacy flat-image loader detected. "
                      "L_slow will supervise on a semantically weak signal "
                      "(cross-iteration slots of unrelated shuffled frames). "
                      "For a real Stage-1 run, use the clip loader "
                      "(see README §4.1).")
                self._warned_flat = True
            imgs_t = batch["img"].to(self.device, non_blocking=True)
            with torch.no_grad():
                target = self.enc.dino(imgs_t)
            slots = self.enc.slot_attn(target)
            prev_for_slow = prev_slots_xiter

        recon, alpha = self.enc.slot_decoder(slots)             # alpha: (B, K, N)
        g, logits = self.enc.router(slots)                      # g: (B, K, 2)
        slow_mask = g[..., 0]

        losses = {
            "L_slot":   slot_reconstruction_loss(recon, target),
            "L_route":  route_prior_kl(logits, prior_slow=self.cfg.route_prior_slow),
            "L_div":    slot_diversity_loss(slots),
        }
        if self.cfg.slow_signal == "alpha_connectivity":
            # Per-frame signal: no prev frame, no matching, no shared init needed.
            losses["L_slow"] = slow_connectivity_loss(
                alpha, logits, temperature=self.cfg.slow_temperature
            )
        elif prev_for_slow is not None:
            if self.cfg.slow_variant == "soft_bce":
                losses["L_slow"] = slow_temporal_loss_soft(
                    slots, prev_for_slow, logits,
                    temperature=self.cfg.slow_temperature,
                )
            elif self.cfg.slow_variant == "quantile_ce":
                losses["L_slow"] = slow_temporal_loss(
                    slots, prev_for_slow, logits,
                    prior_slow=self.cfg.route_prior_slow,
                )
            else:
                raise ValueError(f"unknown slow_variant: {self.cfg.slow_variant}")
        else:
            losses["L_slow"] = slots.new_zeros(())
        # Two slow-ratio metrics — gap between them exposes Gumbel/train-vs-
        # argmax/deploy drift; ideally < 0.05 before Stage 2.
        losses["slow_ratio_gumbel"] = slow_mask.mean().detach()
        with torch.no_grad():
            argmax_slow = (logits.argmax(dim=-1) == 0).float().mean()
        losses["slow_ratio_argmax"] = argmax_slow

        loss = (
            losses["L_slot"]
            + self.cfg.lambda_slow  * losses["L_slow"]
            + self.cfg.lambda_route * losses["L_route"]
            + self.cfg.lambda_div   * losses["L_div"]
        )
        # Return imgs_t so viz callback can dump the current-frame overlay.
        return loss, losses, slots, alpha, imgs_t

    def fit(self, loader: Iterable[dict]) -> None:
        """Train from a dataloader that yields either paired or flat batches.

        Paired (preferred): dict with keys ``img_t`` and ``img_prev``.
        Flat  (legacy):     dict with key  ``img``.  Warns once.
        """
        self.enc.train()
        step = 0
        prev_slots: torch.Tensor | None = None
        t0 = time.time()

        loader_iter = iter(loader)
        while step < self.cfg.steps:
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                batch = next(loader_iter)

            lr_now = self._lr_at(step)
            for pg in self.optim.param_groups:
                pg["lr"] = lr_now

            loss, components, slots, alpha, imgs_used = self._step(batch, prev_slots)

            self.optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                (p for _, p in self.enc.trainable_parameters()),
                max_norm=self.cfg.grad_clip,
            )
            self.optim.step()

            tau_now = self.enc.anneal_router(
                factor=self.cfg.tau_anneal_factor, tau_min=self.cfg.tau_min
            )
            # Only used in legacy flat mode; harmless in paired mode.
            prev_slots = slots.detach()

            if step % self.cfg.log_every == 0:
                self._log(step, components, tau_now, lr_now, t0)
            if step and step % self.cfg.ckpt_every == 0:
                self._save_ckpt(step)
            # Dump slot viz at step 0 too (baseline) and every viz_every after.
            if step % self.cfg.viz_every == 0:
                self._save_slot_viz(step, imgs_used, alpha)

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
        # Full provenance snapshot (R4.3 lesson: W1.1's exact config was not
        # archived, making a later quality gap between two Stage-1 runs
        # undiagnosable). Encoder architecture params are read back from the
        # module itself so the snapshot cannot drift from reality.
        enc = self.enc
        arch = {
            "num_slots": enc.num_slots,
            "slot_dim": enc.slot_dim,
            "t_window": enc.t_window,
            "c_dim": enc.c_dim,
            "slot_iters": enc.slot_attn.iters,
            "learned_queries": getattr(enc.slot_attn, "learned_queries", False),
            "dino_model": enc.dino.model_name,
            "dino_is_mock": enc.dino.is_mock,
            "image_size": enc.dino.image_size,
        }
        payload = {
            "step": step,
            "encoder": enc.state_dict(),
            "optim": self.optim.state_dict(),
            "gumbel_tau": float(enc.router.tau.item()),
            "trainer_config": dict(self.cfg.__dict__),
            "encoder_arch": arch,
            "torch_version": torch.__version__,
        }
        torch.save(payload, p)
        print(f"[Stage1] saved checkpoint → {p} (with config snapshot)")
