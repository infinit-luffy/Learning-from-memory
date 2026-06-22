from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SDAM3DScenePredictor(nn.Module):
    """Action-conditioned SDAM model for offline 3D scene prediction."""

    def __init__(
        self,
        channels: int,
        image_size: int,
        action_dim: int,
        static_dim: int,
        dynamic_dim: int,
        assoc_dim: int,
        hidden_channels: int = 32,
    ) -> None:
        super().__init__()
        if image_size % 8 != 0:
            raise ValueError("image_size must be divisible by 8")
        self.channels = channels
        self.image_size = image_size
        self.action_dim = action_dim
        self.static_dim = static_dim
        self.dynamic_dim = dynamic_dim
        self.assoc_dim = assoc_dim
        self.hidden_channels = hidden_channels
        self.spatial_size = image_size // 8
        self.encoder_channels = hidden_channels * 4
        self.encoder_width = self.encoder_channels * self.spatial_size * self.spatial_size

        self.frame_encoder = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels * 2, self.encoder_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.static_head = nn.Sequential(
            nn.Linear(self.encoder_width, static_dim),
            nn.ReLU(),
            nn.Linear(static_dim, static_dim),
        )
        self.dynamic_input = nn.Sequential(
            nn.Linear(self.encoder_width + action_dim, dynamic_dim),
            nn.ReLU(),
        )
        self.dynamic_gru = nn.GRU(input_size=dynamic_dim, hidden_size=dynamic_dim, batch_first=True)
        self.association = nn.Sequential(
            nn.Linear(static_dim + dynamic_dim + action_dim, assoc_dim),
            nn.ReLU(),
            nn.Linear(assoc_dim, assoc_dim),
        )
        predictor_width = assoc_dim + dynamic_dim + action_dim
        self.next_latent_head = nn.Sequential(
            nn.Linear(predictor_width, dynamic_dim),
            nn.ReLU(),
            nn.Linear(dynamic_dim, dynamic_dim),
        )
        self.decoder_input = nn.Linear(predictor_width, self.encoder_width)
        self.recon_input = nn.Linear(dynamic_dim, self.encoder_width)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.encoder_channels, hidden_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_channels * 2, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_channels, channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )
        self.change_head = nn.Sequential(
            nn.ConvTranspose2d(self.encoder_channels, hidden_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_channels * 2, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_channels, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        next_obs: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if obs.ndim != 5:
            raise ValueError("obs must have shape [B,K,C,H,W]")
        if actions.ndim != 3:
            raise ValueError("actions must have shape [B,K,action_dim]")
        batch_size, context_length, channels, height, width = obs.shape
        if channels != self.channels or height != self.image_size or width != self.image_size:
            raise ValueError(f"obs frames must have shape [{self.channels},{self.image_size},{self.image_size}]")
        if actions.shape != (batch_size, context_length, self.action_dim):
            raise ValueError(f"actions must have shape [B,K,{self.action_dim}]")

        frame_features = self._encode_frames(obs)
        static = self.static_head(frame_features.mean(dim=1))
        dynamic_inputs = self.dynamic_input(torch.cat([frame_features, actions], dim=-1))
        dynamic_seq, _ = self.dynamic_gru(dynamic_inputs)
        last_dynamic = dynamic_seq[:, -1]
        last_action = actions[:, -1]
        z_assoc = self.association(torch.cat([static, last_dynamic, last_action], dim=-1))
        predictor_input = torch.cat([z_assoc, last_dynamic, last_action], dim=-1)
        pred_next_latent = self.next_latent_head(predictor_input)
        pred_next_frame = self.decoder(self._unflatten(self.decoder_input(predictor_input)))
        pred_change_mask = self.change_head(self._unflatten(self.decoder_input(predictor_input)))
        recon_frame = self.decoder(self._unflatten(self.recon_input(last_dynamic)))

        if next_obs is None:
            target_next_latent = last_dynamic.detach()
        else:
            if next_obs.shape != (batch_size, self.channels, self.image_size, self.image_size):
                raise ValueError(f"next_obs must have shape [B,{self.channels},{self.image_size},{self.image_size}]")
            target_next_latent = self.dynamic_input(
                torch.cat(
                    [
                        self.frame_encoder(next_obs),
                        torch.zeros(batch_size, self.action_dim, device=next_obs.device, dtype=next_obs.dtype),
                    ],
                    dim=-1,
                )
            )

        return {
            "static": static,
            "dynamic_seq": dynamic_seq,
            "z_assoc": z_assoc,
            "pred_next_latent": pred_next_latent,
            "target_next_latent": target_next_latent,
            "pred_next_frame": pred_next_frame,
            "pred_change_mask": pred_change_mask,
            "recon_frame": recon_frame,
        }

    def _encode_frames(self, obs: torch.Tensor) -> torch.Tensor:
        batch_size, context_length = obs.shape[:2]
        flat_features = self.frame_encoder(obs.flatten(0, 1))
        return flat_features.reshape(batch_size, context_length, self.encoder_width)

    def _unflatten(self, features: torch.Tensor) -> torch.Tensor:
        return features.reshape(features.shape[0], self.encoder_channels, self.spatial_size, self.spatial_size)


def sdam_3d_prediction_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    frame_weight: float = 1.0,
    latent_weight: float = 1.0,
    change_weight: float = 0.25,
    recon_weight: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    frame_loss = F.mse_loss(outputs["pred_next_frame"], batch["next_obs"])
    latent_loss = F.mse_loss(outputs["pred_next_latent"], outputs["target_next_latent"].detach())
    change_loss = F.binary_cross_entropy(outputs["pred_change_mask"], batch["change_mask"])
    recon_loss = F.mse_loss(outputs["recon_frame"], batch["obs"][:, -1])
    total = (
        frame_weight * frame_loss
        + latent_weight * latent_loss
        + change_weight * change_loss
        + recon_weight * recon_loss
    )
    return total, {
        "total_loss": float(total.detach().cpu().item()),
        "frame_loss": float(frame_loss.detach().cpu().item()),
        "latent_loss": float(latent_loss.detach().cpu().item()),
        "change_loss": float(change_loss.detach().cpu().item()),
        "recon_loss": float(recon_loss.detach().cpu().item()),
    }
