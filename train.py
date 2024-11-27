from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3 import A2C, PPO, DQN
import os
import glob
import time
from datetime import datetime
import cv2
import torch
import numpy as np
from trains import *
from models.big_env_model import ENV_MODEL_V2
import gym
# import roboschool
from stable_baselines3.common.torch_layers import CombinedExtractor
from models.vanilla_vae import VanillaVAE
from utils import *
from collections import deque
from models.autoencoder import  *
# There already exists an environment generator
# that will make and wrap atari environments correctly.
# Here we are also multi-worker training (n_envs=4 => 4 environments)
from models.env_model import AE_R
def random_collect_gym(env, num_eps=150):
    frame_history = []
    for i in range(num_eps):
        env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            # print(action)
            action = np.array([action])
            # observation, reward, terminated, truncated, info = env.step(action)
            observation, reward, done, info = env.step(action)
            # print("Min:", observation.min(), "Max:", observation.max(), 'wi')
            # done = terminated or truncateds
            # frame = preprocess(observation)
            # print(observation[0,:,:,0].shape)
            frame_history.append(observation[0, :, :, 0])

    return frame_history
def random_collect_gym_2(env, vae, num_eps=10):
    org_history = []
    feature_history = []
    background_history = []
    binary_image_history = []
    state_d = deque(maxlen=4)

    for i in range(num_eps):
        env.reset()
        done = False
        for j in range(3):
            state_d.append(np.zeros((1, 84, 84)))
        while not done:
            action = env.action_space.sample()
            # print(action)
            action = np.array([action])
            # observation, reward, terminated, truncated, info = env.step(action)
            observation, reward, done, info = env.step(action)
            # org_history.append(observation)
            frame = torch.tensor(observation, dtype=torch.float32).squeeze(-1)
            org_history.append(frame)
            state_background = vae.generate(frame.to(device)).cpu()
            mu, log_var = vae.encode(frame.unsqueeze(0).to(device))
            state_background_latent = vae.reparameterize(mu, log_var).cpu()

            # print(state_background_latent.shape)
            state_background_ = resize_bcg(state_background)
            background_history.append(state_background_)
            frame = frame - state_background

            threshold = 0.1
            binary_image = (frame > threshold).cpu().numpy().astype(np.uint8)
            # print(binary_image.shape)
            # print(binary_image[0].shape)
            state_d.append(binary_image[0])

            binary_image_history.append(np.stack(state_d, axis=0))

            # print("Min:", observation.min(), "Max:", observation.max(), 'wi')
            # done = terminated or truncateds
            # frame = preprocess(observation)
            # print(observation[0,:,:,0].shape)
            # frame_history.append(observation[0, :, :, 0])

    return org_history, binary_image_history, background_history


def random_collect_gym_3(env, vae, ae, num_eps=10):
    org_history = []
    feature_history = []
    background_history = []
    binary_image_history = []
    state_d = deque(maxlen=4)
    for i in range(num_eps):
        env.reset()
        done = False

        # Initialize the deque with zeros (3 elements)
      # First two filled with zeros
        for j in range(3):
            state_d.append(np.zeros(32,))

        while not done:
            action = env.action_space.sample()
            action = np.array([action])
            observation, reward, done, info = env.step(action)

            # Process observation frame
            frame = torch.tensor(observation, dtype=torch.float32).squeeze(-1)
            org_history.append(frame.numpy())

            # Get background representation using VAE
            state_background = vae.generate(frame.to(device)).cpu()
            mu, log_var = vae.encode(frame.unsqueeze(0).to(device))
            state_background_latent = vae.reparameterize(mu, log_var).cpu()

            # Resize background and append to history
            state_background_ = resize_bcg(state_background)
            background_history.append(state_background_)

            # Subtract background from frame to get the foreground
            frame = frame - state_background

            # Create binary image with thresholding
            threshold = 0.1
            binary_image = (frame > threshold).cpu().numpy().astype(np.uint8)

            # observation = binary_image[0]
            # print(binary_image.shape)
            binary_image = torch.from_numpy(binary_image).float()
            feature_e = ae.encode(binary_image.to(device)).cpu().detach().numpy()
            state_d.append(feature_e[0])
            # print(state_d, feature_e.shape)
            # state_d.append(feature_e)

            # print(np.stack(state_d, axis=0).shape)
            # Append the current binary image to the deque
            binary_image_history.append(np.stack(state_d, axis=0))  # Append and pop from left if necessary
            # print(feature_e.shape)
            # At the end of each episode, pad the deque with zeros
            # if done:
            #     binary_image_queue.append(np.zeros(32,))  # Append a zero frame to fill last spot
            #     binary_image_queue.append(np.zeros(32,))  # Append another zero frame

            # Convert the deque to numpy array and add to history
            # print(binary_image_queue)
            # binary_image_history.append(np.array(binary_image_queue))  # Convert to numpy array

    return org_history, binary_image_history, background_history


#训练背景提取网络
device = 'cuda'
def preprocess(image):
    # image = image[34:194, :, :] # 160, 160, 3
    # image = np.mean(image, axis=2, keepdims=False) # 160, 160
    # image = image[::2, ::2] # 80, 80
    # print(image.shape)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    target_size = (84, 84)

    # 使用cv2进行图像调整大小
    resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    # image = image.transp      ose(2,0,1)
    resized_image = resized_image.astype(float) / 256  # remove background
    return resized_image

from tensorboardX import SummaryWriter
# TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
# log_dir_vae = "LOG/VAE_image/" + TIMESTAMP
# writer_vae = SummaryWriter(log_dir_vae)
# TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
# log_dir_ae = "LOG/ae_image/" + TIMESTAMP
# writer_ae = SummaryWriter(log_dir_ae)
# TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
# log_dir_aer = "LOG/rae_image/" + TIMESTAMP
# writer_aer = SummaryWriter(log_dir_aer)

# env = make_atari_env("Alien-v4", bg_model=None, connection_model=None, n_envs=1, seed=0)
vae = VanillaVAE(in_channels=1, latent_dim=32).to('cuda')
# frame_history = random_collect_gym(env, 1000)
# vae_global_step = 0
# vae_global_step = train_vae(vae, frame_history, 128, writer=writer_vae, global_step=vae_global_step, train_step=100)
# torch.save(vae.state_dict(), 'vae_Alien.pth')
# torch.cuda.empty_cache()
vae.load_state_dict(torch.load('vae_Alien.pth'))
#
# env_org_images, env_binary_imgs, env_background_low_dim  = random_collect_gym_2(env, vae, 1000)
# print(len(a), len(b), len(c))

# ae = AutoEncoder(in_channels=1, latent_dim=32).to(device)
# train_ae(ae, dataset=b, batch_size=64, writer=writer_ae, train_step=200)
# torch.save(ae.state_dict(), 'ae_Alien.pth')
# torch.cuda.empty_cache()
# ae.load_state_dict(torch.load('ae_Alien.pth'))

# ae_R = AE_R(3, 10, 128, 32).to(device)
env_model = ENV_MODEL_V2(input_size=32, hidden_size=128).to(device)
# org, bin, bg = random_collect_gym_3(env,vae, ae, 200)
# train_ae_R(ae_R, bg, bin, org, 64, writer=writer_aer)
# torch.save(ae_R.state_dict(), 'rae_Alien.pth')
# env_global_step = train_env_model_V2(env_model, env_binary_imgs, env_org_images, env_background_low_dim,
#                             batch_size=128, writer=writer_ae, global_step=0, train_step=100)
# torch.save(env_model.state_dict(), 'env_Alien.pth')
env_model.load_state_dict(torch.load('env_Alien.pth'))

vec_env = make_atari_env("Alien-v4", bg_model=vae, ae_model=env_model, n_envs=4, seed=0)
# Frame-stacking with 4 frames
vec_env = VecFrameStack(vec_env, n_stack=1)
from gym import spaces
observation_space = spaces.Dict({
    "image": spaces.Box(low=0, high=1, shape=(1, 84, 84), dtype=np.uint8),  # RGB image
    "vector": spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32)  # 1D vector of size 10
})


# Define policy_kwargs for PPO with the custom CombinedExtractor
policy_kwargs = dict(
    features_extractor_class=CombinedExtractor,
    features_extractor_kwargs=dict(observation_space=observation_space, cnn_output_dim=256, normalized_image=False)
)

# policy_kwargs = dict(
#     features_extractor_class=CombinedExtractor,
#     features_extractor_kwargs=dict(observation_space=observation_space, cnn_output_dim=256, normalized_image=False)
# )
from stable_baselines3 import DQN

# 指定 TensorBoard 日志目录

model = DQN("MlpPolicy", vec_env, verbose=1, batch_size = 256,buffer_size=500000, learning_rate=1e-4, tensorboard_log="./dqn_tensorboard_logs/")
model.learn(total_timesteps=5000000)

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs, deterministic=False)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")
