from __future__ import annotations

import csv
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from sdam.config import AtariSDAMConfig
from sdam.experiments.atari import _SB3_EXTRA_MESSAGE, _resolve_device, _safe_torch_load


def _load_dqn():
    try:
        from stable_baselines3 import DQN
    except ImportError as exc:
        raise ImportError(_SB3_EXTRA_MESSAGE) from exc

    return DQN


def _load_vec_env_tools():
    try:
        from stable_baselines3.common.atari_wrappers import AtariWrapper
        from stable_baselines3.common.evaluation import evaluate_policy
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError as exc:
        raise ImportError(_SB3_EXTRA_MESSAGE) from exc

    return AtariWrapper, DummyVecEnv, evaluate_policy


def _load_gymnasium():
    try:
        import gymnasium as gym
        from gymnasium import spaces
    except ImportError as exc:
        raise ImportError(_SB3_EXTRA_MESSAGE) from exc

    return gym, spaces


class MainVanillaVAE(nn.Module):
    """VAE architecture compatible with origin/main's VanillaVAE checkpoints."""

    def __init__(self, in_channels: int = 1, latent_dim: int = 32) -> None:
        super().__init__()
        hidden_dims = [32, 64, 128, 256]
        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, hidden_dims[0], kernel_size=8, stride=4, padding=1),
                nn.BatchNorm2d(hidden_dims[0]),
                nn.LeakyReLU(),
            )
        )
        modules.append(
            nn.Sequential(
                nn.Conv2d(hidden_dims[0], hidden_dims[1], kernel_size=6, stride=3, padding=1),
                nn.BatchNorm2d(hidden_dims[1]),
                nn.LeakyReLU(),
            )
        )
        modules.append(
            nn.Sequential(
                nn.Conv2d(hidden_dims[1], hidden_dims[2], kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(hidden_dims[2]),
                nn.LeakyReLU(),
            )
        )
        modules.append(
            nn.Sequential(
                nn.Conv2d(hidden_dims[2], hidden_dims[3], kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(hidden_dims[3]),
                nn.LeakyReLU(),
            )
        )
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(1024, latent_dim)
        self.fc_var = nn.Linear(1024, latent_dim)
        self.feature = nn.Linear(1024, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, 1024)

        decoder_dims = [256, 128, 64, 32]
        decoder = []
        decoder.append(
            nn.Sequential(
                nn.ConvTranspose2d(decoder_dims[0], decoder_dims[1], kernel_size=7, stride=3, padding=1),
                nn.BatchNorm2d(decoder_dims[1]),
                nn.LeakyReLU(),
            )
        )
        decoder.append(
            nn.Sequential(
                nn.ConvTranspose2d(decoder_dims[1], decoder_dims[2], kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(decoder_dims[2]),
                nn.LeakyReLU(),
            )
        )
        decoder.append(
            nn.Sequential(
                nn.ConvTranspose2d(decoder_dims[2], decoder_dims[3], kernel_size=3, stride=3, padding=1),
                nn.BatchNorm2d(decoder_dims[3]),
                nn.LeakyReLU(),
            )
        )
        self.decoder = nn.Sequential(*decoder)
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, in_channels, kernel_size=3),
            nn.Sigmoid(),
        )

    def encode(self, input_tensor: torch.Tensor):
        result = self.encoder(input_tensor)
        result = torch.flatten(result, start_dim=1)
        return [self.fc_mu(result), self.fc_var(result)]

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        result = self.decoder_input(z)
        result = result.view(-1, 256, 2, 2)
        result = self.decoder(result)
        return self.final_layer(result)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input_tensor: torch.Tensor):
        input_tensor = input_tensor.unsqueeze(1)
        mu, log_var = self.encode(input_tensor)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input_tensor, mu, log_var]

    def generate(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.forward(input_tensor)[0]


class MainAttentionLayer(nn.Module):
    """Unused compatibility layer present in origin/main's AE_R checkpoints."""

    def __init__(self, hidden_size: int = 128, window_size: int = 3) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.attn = nn.Linear(hidden_size * 2, 1)

    def forward(self, encoder_output: torch.Tensor, encoder_hidden: torch.Tensor) -> torch.Tensor:
        hidden_temp = encoder_hidden[0].unsqueeze(0).expand(self.window_size, -1, -1)
        att_input = torch.cat((encoder_output, hidden_temp), dim=2)
        att_weights = torch.softmax(self.attn(att_input), dim=0)
        return torch.bmm(att_weights.permute(1, 2, 0), encoder_output.transpose(0, 1))


class MainAER(nn.Module):
    """Connection recognizer compatible with origin/main's AE_R state dict."""

    def __init__(self, hidden_size: int = 128, input_size: int = 32) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.gru = nn.GRU(input_size, hidden_size, self.num_layers, batch_first=True)
        self.conv_background = nn.Conv2d(1, 32, kernel_size=5, stride=2)
        self.conv_background_2 = nn.Conv2d(32, 1, kernel_size=3, stride=1)
        self.fc_b = nn.Linear(26 * 19, 128)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Linear(input_size, 1)
        self.decoder_input = nn.Linear(32, 1024)
        self.linear = nn.Linear(hidden_size, input_size)
        self.atten = MainAttentionLayer(128, 3)
        self.li = nn.Linear(hidden_size, 32)

        decoder = []
        decoder.append(
            nn.Sequential(
                nn.ConvTranspose2d(256, 128, kernel_size=7, stride=3, padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(),
            )
        )
        decoder.append(
            nn.Sequential(
                nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(),
            )
        )
        decoder.append(
            nn.Sequential(
                nn.ConvTranspose2d(64, 32, kernel_size=3, stride=3, padding=1),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(),
            )
        )
        self.decoder = nn.Sequential(*decoder)
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 1, kernel_size=3),
            nn.Sigmoid(),
        )

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        result = self.decoder_input(z)
        result = result.view(-1, 256, 2, 2)
        result = self.decoder(result)
        return self.final_layer(result)

    def encode(self, state: torch.Tensor, background: torch.Tensor) -> torch.Tensor:
        background = F.relu(self.conv_background(background))
        background = F.relu(self.conv_background_2(background))
        background = background.view(-1, 26 * 19)
        hidden = self.relu(self.fc_b(background)).unsqueeze(0)
        out, _ = self.gru(state, hidden)
        return F.relu(self.li(self.relu(out[:, -1, :])))

    def forward(self, state: torch.Tensor, background: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(state, background))


class MainEnvModelV2(nn.Module):
    """ENV_MODEL_V2 architecture compatible with origin/main checkpoints."""

    def __init__(self, input_size: int = 32, hidden_size: int = 128) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.bce = nn.BCELoss()
        self.connection_recog = MainAER(hidden_size=hidden_size, input_size=input_size)
        self.feature_recog = MainVanillaVAE(in_channels=1, latent_dim=input_size)

    def get_feature_encode(self, state: torch.Tensor) -> torch.Tensor:
        mu, log_var = self.feature_recog.encode(state)
        return self.feature_recog.reparameterize(mu, log_var)

    def get_z_encode(self, state: torch.Tensor, background: torch.Tensor) -> torch.Tensor:
        return self.connection_recog.encode(state, background)


class MainVectorObservationWrapper:
    """Origin/main-style wrapper: Atari frame -> 160-D frozen memory vector."""

    def __init__(self, env, vae: MainVanillaVAE, env_model: MainEnvModelV2, device: torch.device) -> None:
        gym, spaces = _load_gymnasium()

        class _Wrapper(gym.ObservationWrapper):
            def __init__(self, wrapped_env):
                super().__init__(wrapped_env)
                self.vae = vae.eval()
                self.env_model = env_model.eval()
                self.device = device
                self.state_background = None
                self.state_background_latent = None
                self.feature_deque = deque([np.zeros(32, dtype=np.float32) for _ in range(4)], maxlen=4)
                self.observation_space = spaces.Box(
                    low=-10,
                    high=10,
                    shape=(160,),
                    dtype=np.float32,
                )

            def reset(self, **kwargs):
                self.state_background = None
                self.state_background_latent = None
                self.feature_deque = deque([np.zeros(32, dtype=np.float32) for _ in range(4)], maxlen=4)
                return super().reset(**kwargs)

            def observation(self, frame):
                frame_tensor = torch.as_tensor(frame, dtype=torch.float32)
                frame_tensor = frame_tensor.squeeze()
                if frame_tensor.max().item() > 1.0:
                    frame_tensor = frame_tensor / 255.0
                with torch.no_grad():
                    if self.state_background is None:
                        vae_input = frame_tensor.unsqueeze(0).to(self.device)
                        self.state_background = self.vae.generate(vae_input).cpu()
                        mu, log_var = self.vae.encode(vae_input.unsqueeze(0))
                        self.state_background_latent = self.vae.reparameterize(mu, log_var).cpu()
                    residual = frame_tensor.cpu() - self.state_background.squeeze(0).squeeze(0)
                    binary = (residual > 0.1).float().unsqueeze(0).unsqueeze(0).to(self.device)
                    feature = self.env_model.get_feature_encode(binary)[0].detach().cpu().numpy()
                    self.feature_deque.append(feature.astype(np.float32))
                dynamic = np.stack(self.feature_deque, axis=0).astype(np.float32).reshape(-1)
                background = self.state_background_latent.detach().cpu().numpy().squeeze().astype(np.float32)
                return np.concatenate([dynamic, background], axis=0).astype(np.float32)

        self.wrapper = _Wrapper(env)

    def unwrap(self):
        return self.wrapper


def load_main_vector_models(
    vae_path: str | Path,
    env_model_path: str | Path,
    device: str | torch.device = "auto",
) -> tuple[MainVanillaVAE, MainEnvModelV2, torch.device]:
    resolved_device = _resolve_device(device)
    vae = MainVanillaVAE(in_channels=1, latent_dim=32).to(resolved_device)
    env_model = MainEnvModelV2(input_size=32, hidden_size=128).to(resolved_device)
    vae.load_state_dict(_safe_torch_load(vae_path, map_location=str(resolved_device)))
    env_model.load_state_dict(_safe_torch_load(env_model_path, map_location=str(resolved_device)))
    vae.eval()
    env_model.eval()
    return vae, env_model, resolved_device


def build_main_vector_env(
    config: AtariSDAMConfig,
    vae: MainVanillaVAE,
    env_model: MainEnvModelV2,
    device: torch.device,
):
    gym, _ = _load_gymnasium()
    AtariWrapper, DummyVecEnv, _ = _load_vec_env_tools()

    def make_env(rank: int):
        def _init():
            env = gym.make(config.env.env_id)
            env = AtariWrapper(
                env,
                terminal_on_life_loss=config.env.terminal_on_life_loss,
                clip_reward=True,
            )
            env = MainVectorObservationWrapper(env, vae, env_model, device).unwrap()
            env.reset(seed=config.env.seed + rank)
            return env

        return _init

    return DummyVecEnv([make_env(rank) for rank in range(config.env.n_envs)])


def train_main_vector_dqn(
    config: AtariSDAMConfig,
    vae_path: str | Path,
    env_model_path: str | Path,
    total_timesteps: int | None = None,
    save_path: str | Path | None = None,
    eval_episodes: int = 10,
    output_dir: str | Path | None = None,
    device: str = "auto",
    verbose: int = 1,
    batch_size: int = 256,
    buffer_size: int = 500000,
    learning_rate: float = 1e-4,
) -> dict[str, str | float | int]:
    if eval_episodes <= 0:
        raise ValueError("eval_episodes must be positive")
    steps = total_timesteps if total_timesteps is not None else config.training.total_timesteps
    if steps <= 0:
        raise ValueError("total_timesteps must be positive")

    vae, env_model, resolved_device = load_main_vector_models(vae_path, env_model_path, device)
    env = None
    try:
        env = build_main_vector_env(config, vae, env_model, resolved_device)
        DQN = _load_dqn()
        model = DQN(
            "MlpPolicy",
            env,
            verbose=verbose,
            batch_size=batch_size,
            buffer_size=buffer_size,
            learning_rate=learning_rate,
            device=str(resolved_device),
        )
        model.learn(total_timesteps=steps)
        output_path = Path(output_dir) if output_dir is not None else Path("runs/alien/main_vector_dqn")
        output_path.mkdir(parents=True, exist_ok=True)
        model_path = Path(save_path) if save_path is not None else output_path / "main_vector_dqn.zip"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(model_path)
        _, _, evaluate_policy = _load_vec_env_tools()
        rewards, lengths = evaluate_policy(
            model,
            env,
            n_eval_episodes=eval_episodes,
            deterministic=True,
            return_episode_rewards=True,
        )
        mean_reward = float(sum(rewards) / len(rewards))
        mean_length = float(sum(lengths) / len(lengths))
        std_reward = float((sum((reward - mean_reward) ** 2 for reward in rewards) / len(rewards)) ** 0.5)
        row = {
            "method": "main_vector_dqn",
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "mean_ep_length": mean_length,
            "episodes": len(rewards),
            "model_path": str(model_path),
        }
        with (output_path / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        return row
    finally:
        close = getattr(env, "close", None)
        if close is not None:
            close()
