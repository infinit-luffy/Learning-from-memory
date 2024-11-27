from typing import Dict, SupportsFloat
import torchvision.transforms as transforms
from PIL import Image
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from collections import deque

from stable_baselines3.common.type_aliases import AtariResetReturn, AtariStepReturn
import torch
try:
    import cv2

    cv2.ocl.setUseOpenCL(False)
except ImportError:
    cv2 = None  # type: ignore[assignment]
def resize_bcg(background):
    pil_image = transforms.ToPILImage()(background.squeeze(0))  # 去除批处理和通道维度

    # 裁剪图像，然后调整大小
    resize_transform = transforms.Compose(
        [transforms.Resize((60, 45), interpolation=Image.BILINEAR)])

    resized_pil_image = resize_transform(pil_image)
    bc_ndara = np.expand_dims(np.array(resized_pil_image), axis=0)
    bc_ndara = np.expand_dims(bc_ndara, axis=0)

    return bc_ndara
device = 'cuda'
class StickyActionEnv(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    """
    Sticky action.

    Paper: https://arxiv.org/abs/1709.06009
    Official implementation: https://github.com/mgbellemare/Arcade-Learning-Environment

    :param env: Environment to wrap
    :param action_repeat_probability: Probability of repeating the last action
    """

    def __init__(self, env: gym.Env, action_repeat_probability: float) -> None:
        super().__init__(env)
        self.action_repeat_probability = action_repeat_probability
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"  # type: ignore[attr-defined]

    def reset(self, **kwargs) -> AtariResetReturn:
        self._sticky_action = 0  # NOOP
        return self.env.reset(**kwargs)

    def step(self, action: int) -> AtariStepReturn:
        if self.np_random.random() >= self.action_repeat_probability:
            self._sticky_action = action
        return self.env.step(self._sticky_action)


class NoopResetEnv(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    """
    Sample initial states by taking random number of no-ops on reset.
    No-op is assumed to be action 0.

    :param env: Environment to wrap
    :param noop_max: Maximum value of no-ops to run
    """

    def __init__(self, env: gym.Env, noop_max: int = 30) -> None:
        super().__init__(env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"  # type: ignore[attr-defined]

    def reset(self, **kwargs) -> AtariResetReturn:
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)
        assert noops > 0
        obs = np.zeros(0)
        info: Dict = {}
        for _ in range(noops):
            obs, _, terminated, truncated, info = self.env.step(self.noop_action)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info


class FireResetEnv(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    """
    Take action on reset for environments that are fixed until firing.

    :param env: Environment to wrap
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"  # type: ignore[attr-defined]
        assert len(env.unwrapped.get_action_meanings()) >= 3  # type: ignore[attr-defined]

    def reset(self, **kwargs) -> AtariResetReturn:
        self.env.reset(**kwargs)
        obs, _, terminated, truncated, _ = self.env.step(1)
        # obs = preprocess(obs)
        if terminated or truncated:
            self.env.reset(**kwargs)
        obs, _, terminated, truncated, _ = self.env.step(2)
        # obs = preprocess(obs)
        # print(obs.shape, '-----')
        if terminated or truncated:
            self.env.reset(**kwargs)
        return obs, {}


class EpisodicLifeEnv(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    """
    Make end-of-life == end-of-episode, but only reset on true game over.
    Done by DeepMind for the DQN and co. since it helps value estimation.

    :param env: Environment to wrap
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action: int) -> AtariStepReturn:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated or truncated
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()  # type: ignore[attr-defined]
        if 0 < lives < self.lives:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            terminated = True
        self.lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs) -> AtariResetReturn:
        """
        Calls the Gym environment reset, only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.

        :param kwargs: Extra keywords passed to env.reset() call
        :return: the first observation of the environment
        """
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, terminated, truncated, info = self.env.step(0)

            # The no-op step can lead to a game over, so we need to check it again
            # to see if we should reset the environment and avoid the
            # monitor.py `RuntimeError: Tried to step environment that needs reset`
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        self.lives = self.env.unwrapped.ale.lives()  # type: ignore[attr-defined]
        return obs, info


class MaxAndSkipEnv(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    """
    Return only every ``skip``-th frame (frameskipping)
    and return the max between the two last frames.

    :param env: Environment to wrap
    :param skip: Number of ``skip``-th frame
        The same action will be taken ``skip`` times.
    """

    def __init__(self, env: gym.Env, skip: int = 4) -> None:
        super().__init__(env)
        # most recent raw observations (for max pooling across time steps)
        assert env.observation_space.dtype is not None, "No dtype specified for the observation space"
        assert env.observation_space.shape is not None, "No shape defined for the observation space"
        # self._obs_buffer = np.zeros((2, *env.observation_space.shape), dtype=env.observation_space.dtype)

        self._obs_buffer = np.zeros((2, *env.observation_space.shape), dtype=env.observation_space.dtype)
        self._skip = skip

    def step(self, action: int) -> AtariStepReturn:
        """
        Step the environment with the given action
        Repeat action, sum reward, and max over last observations.

        :param action: the action
        :return: observation, reward, terminated, truncated, information
        """
        total_reward = 0.0
        terminated = truncated = False
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            # print(obs.shape)
            # print(obs)
            # if obs.shape == (210, 160, 3):
            #     obs = preprocess(obs)
            done = terminated or truncated
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += float(reward)
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, terminated, truncated, info


class ClipRewardEnv(gym.RewardWrapper):
    """
    Clip the reward to {+1, 0, -1} by its sign.

    :param env: Environment to wrap
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)

    def reward(self, reward: SupportsFloat) -> float:
        """
        Bin reward to {+1, 0, -1} by its sign.

        :param reward:
        :return:
        """
        return np.sign(float(reward))
def rle_encode(image):
    """
    对二值图像进行行程长度编码（RLE）。

    参数:
    image (list): 二值图像的二维列表，0 表示黑色像素，1 表示白色像素。

    返回:
    list: 编码后的 RLE 列表，格式为 [(值, 长度), ...]。
    """
    rle = []
    current_value = image[0][0]
    count = 0

    for row in image:
        for pixel in row:
            if pixel == current_value:
                count += 1
            else:
                rle.append((current_value, count))
                current_value = pixel
                count = 1
    rle.append((current_value, count))  # 追加最后一个值

    return rle

def encode_rare_value_by_blocks(image, rare_value=1, block_size=8):
    image = image[0]
    height, width = image.shape
    center_points = []

    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            # 计算当前块的边界
            block = image[i:i + block_size, j:j + block_size]
            # 检查块内稀有值的数量
            rare_count = np.sum(block == rare_value)

            if rare_count > 0:
                # 计算块的中心点
                center_x = i + block_size // 2
                center_y = j + block_size // 2
                center_points.append((center_x, center_y, rare_count))  # 记录中心点和稀有值数量

    # 按照稀有值数量降序排序，取前 10 个
    center_points.sort(key=lambda x: x[2], reverse=True)
    top_points = center_points[:10]  # 取前 10 个点

    # 生成一维编码向量
    encoding_vector = []
    for x, y, _ in top_points:  # 修正：解包时添加第三个值为 _
        encoding_vector.append(x)
        encoding_vector.append(y)

    # 如果需要固定长度的编码向量，填充或截断
    while len(encoding_vector) < 20:
        encoding_vector.append(-1)  # 使用 -1 填充表示无效位置

    return encoding_vector[:20]  # 返回前 20 个元素

class WarpFrame(gym.ObservationWrapper[np.ndarray, int, np.ndarray]):
    """
    Convert to grayscale and warp frames to 84x84 (default)
    as done in the Nature paper and later work.

    :param env: Environment to wrap
    :param width: New frame width
    :param height: New frame height
    """

    def __init__(self, env: gym.Env, width: int = 84, height: int = 84) -> None:
        super().__init__(env)
        self.width = width
        self.height = height
        assert isinstance(env.observation_space, spaces.Box), f"Expected Box space, got {env.observation_space}"

        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.height, self.width, 1),
            # dtype=env.observation_space.dtype,  # type: ignore[arg-type]
            dtype=np.float32,  # type: ignore[arg-type]
        )

    def observation(self, frame: np.ndarray) -> np.ndarray:
        """
        returns the current observation from a frame

        :param frame: environment frame
        :return: the observation
        """
        assert cv2 is not None, "OpenCV is not installed, you can do `pip install opencv-python`"
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        # print(frame)
        frame = frame.astype(float) / 255
        # print("Min:", frame.min(), "Max:", frame.max())
        # print("Min:", frame[:, :, None].min(), "Max:", frame[:, :, None].max(), 'te in')
        return frame[:, :, None]

class WarpFrame_bg(gym.ObservationWrapper[np.ndarray, int, np.ndarray]):
    """
    Convert to grayscale and warp frames to 84x84 (default)
    as done in the Nature paper and later work.

    :param env: Environment to wrap
    :param width: New frame width
    :param height: New frame height
    """

    def __init__(self, env: gym.Env, width: int = 84, height: int = 84, bg_model=None) -> None:
        super().__init__(env)
        self.width = width
        self.height = height
        self.vae = bg_model
        assert isinstance(env.observation_space, spaces.Box), f"Expected Box space, got {env.observation_space}"

        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.height, self.width, 1),
            dtype=np.float32,  # type: ignore[arg-type]
        )

    def observation(self, frame: np.ndarray) -> np.ndarray:
        """
        returns the current observation from a frame

        :param frame: environment frame
        :return: the observation
        """
        assert cv2 is not None, "OpenCV is not installed, you can do `pip install opencv-python`"
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        # frame = frame.astype(float) / 256
        # print(frame.shape)
        frame = frame.astype(float) / 255
        frame = torch.tensor(frame, dtype=torch.float32).unsqueeze(0)

        state_background = self.vae.generate(frame.unsqueeze(0).to(device)).cpu()
        # mu, log_var = self.vae.encode(frame.unsqueeze(0).to(device))
        # state_background_latent = self.vae.reparameterize(mu, log_var).cpu()

        # state_background_ = resize_bcg(state_background)
        frame = frame - state_background

        threshold = 0.1
        binary_image = (frame > threshold).cpu().numpy().astype(np.uint8)

        observation = binary_image[0]
        # print("Min:", observation.min(), "Max:", observation.max())
        # print(observation.shape)
        if observation.shape == (84,84):
            observation = observation.unsqueeze(2)
        elif observation.shape == (1, 84, 84):
            observation = np.transpose(observation, (1, 2, 0))
        else:
            print(observation.shape, 'reer')
        return np.float32(observation)

class WarpFrame_bg_bina(gym.ObservationWrapper[np.ndarray, int, np.ndarray]):
    """
    Convert to grayscale and warp frames to 84x84 (default)
    as done in the Nature paper and later work.

    :param env: Environment to wrap
    :param width: New frame width
    :param height: New frame height
    """

    def __init__(self, env: gym.Env, width: int = 84, height: int = 84, bg_model=None) -> None:
        super().__init__(env)
        self.width = width
        self.height = height
        self.vae = bg_model
        self.bin_img_dq = deque(maxlen=4)
        for i in range(4):
            self.bin_img_dq.append(np.zeros((84, 84)))
        assert isinstance(env.observation_space, spaces.Box), f"Expected Box space, got {env.observation_space}"

        # self.observation_space = spaces.Box(
        #     low=0,
        #     high=1,
        #     shape=(self.height, self.width, 1),
        #     dtype=np.float32,  # type: ignore[arg-type]
        # )
        self.observation_space = spaces.Dict({
            "image": spaces.Box(
            low=0,
            high=1,
            shape=(self.height, self.width, 4),
            dtype=np.float32,  # type: ignore[arg-type]
        ),  # RGB image
            "vector": spaces.Box(low=-10, high=10, shape=(32,), dtype=np.float32)  # 1D vector of size 10
        })

    def observation(self, frame: np.ndarray) -> np.ndarray:
        """
        returns the current observation from a frame

        :param frame: environment frame
        :return: the observation
        """
        assert cv2 is not None, "OpenCV is not installed, you can do `pip install opencv-python`"
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        # frame = frame.astype(float) / 256
        # print(frame.shape)
        frame = frame.astype(float) / 255
        frame = torch.tensor(frame, dtype=torch.float32)
        # print(frame.shape)
        if frame.shape == (84,84):
            frame = frame.unsqueeze(0)
        # print(frame.shape)
        state_background = self.vae.generate(frame.to(device)).cpu()
        mu, log_var = self.vae.encode(frame.unsqueeze(0).to(device))
        state_background_latent = self.vae.reparameterize(mu, log_var).cpu()

        # print(state_background_latent.shape)
        # state_background_ = resize_bcg(state_background)
        frame = frame - state_background
        
        threshold = 0.1
        binary_image = (frame > threshold).cpu().numpy().astype(np.uint8)

        observation =binary_image[0]
        # self.bin_img_dq.append(observation)

        # print(dd)
        # print("Min:", observation.min(), "Max:", observation.max())
        # print(observation.shape)
        if state_background_latent.shape != (64,):
            # print('fff', state_background_latent.shape)
            state_background_latent = state_background_latent.squeeze(0)
        if observation.shape == (84,84):
            pass
        elif observation.shape == (1, 84, 84):
            # observation = np.transpose(observation, (1, 2, 0))
            observation = observation.squeeze(0)
        else:
            print(observation.shape, 'reer')

        self.bin_img_dq.append(observation)
        observation = np.stack(self.bin_img_dq, axis=-1)
        # print(observation.shape)
        observations = {
            "image": np.float32(observation),
            "vector": state_background_latent.detach().numpy(),
        }

        return observations




class WarpFrame_bina_ecd(gym.ObservationWrapper[np.ndarray, int, np.ndarray]):
    """
    Convert to grayscale and warp frames to 84x84 (default)
    as done in the Nature paper and later work.

    :param env: Environment to wrap
    :param width: New frame width
    :param height: New frame height
    """

    def __init__(self, env: gym.Env, width: int = 84, height: int = 84, bg_model=None, ae_model=None) -> None:
        super().__init__(env)
        self.width = width
        self.height = height
        self.vae = bg_model
        self.ae = ae_model
        self.backgroun_list = deque(maxlen=1)
        self.d_state_list = deque(maxlen=4)
        self.d_c_state_list = deque(maxlen=1)
        assert isinstance(env.observation_space, spaces.Box), f"Expected Box space, got {env.observation_space}"

        self.observation_space = spaces.Box(
            low=-10,
            high=10,
            shape=(64,),
            dtype=np.float32,  # type: ignore[arg-type]
        )
        # self.observation_space = spaces.Dict({
        #     "image": spaces.Box(
        #         low=0,
        #         high=1,
        #         shape=(self.height, self.width, 1),
        #         dtype=np.float32,  # type: ignore[arg-type]
        #     ),  # RGB image
        #     "vector": spaces.Box(low=-10, high=10, shape=(64,), dtype=np.float32)  # 1D vector of size 10
        # })

    def observation(self, frame: np.ndarray) -> np.ndarray:
        """
        returns the current observation from a frame

        :param frame: environment frame
        :return: the observation
        """
        assert cv2 is not None, "OpenCV is not installed, you can do `pip install opencv-python`"
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        # frame = frame.astype(float) / 256
        # print(frame.shape)
        frame = frame.astype(float) / 255
        # frame = torch.tensor(frame, dtype=torch.float32).unsqueeze(0)
        frame = torch.tensor(frame, dtype=torch.float32)
        state_background = self.vae.generate(frame.unsqueeze(0).to(device)).cpu()
        # print(frame.shape)
        mu, log_var = self.vae.encode(frame.unsqueeze(0).unsqueeze(0).to(device))
        state_background_latent = self.vae.reparameterize(mu, log_var).cpu()
        # print(state_background_latent.shape)
        # state_background_ = resize_bcg(state_background)
        frame = frame - state_background

        threshold = 0.1
        binary_image = (frame > threshold).cpu().numpy().astype(np.uint8)

        # observation = binary_image[0]
        # print(observation.shape)
        observation = torch.from_numpy(binary_image).float().cuda()
        observation = self.ae.encode(observation)
        # print(ec1)

        # dd = encode_rare_value_by_blocks(observation)
        ec1 = state_background_latent.detach().cpu().numpy().squeeze()
        ec2 = observation.detach().cpu().numpy().squeeze()
        # print(ec1.shape, ec2.shape)
        combined = np.hstack((ec1, ec2))
        observation = combined
        # print(combined.shape)
        # print(np.array(dd)*1.0/84)
        # print("Min:", observation.min(), "Max:", observation.max())
        # print(observation.shape)
        # if state_background_latent.shape != (64,):
        #     # print('fff', state_background_latent.shape)
        #     state_background_latent = state_background_latent.squeeze(0)
        # if observation.shape == (84, 84):
        #     observation = observation.unsqueeze(2)
        # elif observation.shape == (1, 84, 84):
        #     observation = np.transpose(observation, (1, 2, 0))
        # else:
        #     print(observation.shape, 'reer')
        # observations = {
        #     "image": np.float32(observation),
        #     "vector": state_background_latent.detach().numpy(),
        # }

        return observation


class WarpFrame_bina_ecd_con(gym.ObservationWrapper[np.ndarray, int, np.ndarray]):
    """
    Convert to grayscale and warp frames to 84x84 (default)
    as done in the Nature paper and later work.

    :param env: Environment to wrap
    :param width: New frame width
    :param height: New frame height
    """

    def __init__(self, env: gym.Env, width: int = 84, height: int = 84, bg_model=None, ae_model=None, c_model=None) -> None:
        super().__init__(env)
        self.width = width
        self.height = height
        self.vae = bg_model
        self.env_model = ae_model
        self.c = c_model
        # self.background_ = None
        self.state_background = None
        self.state_background_ = None
        self.state_background_latent = None
        self.f_deque = deque(maxlen=4)
        for i in range(4):
            self.f_deque.append(np.zeros(32,))
        assert isinstance(env.observation_space, spaces.Box), f"Expected Box space, got {env.observation_space}"

        self.observation_space = spaces.Box(
            low=-10,
            high=10,
            shape=(160,),
            dtype=np.float32,  # type: ignore[arg-type]
        )
        # self.observation_space = spaces.Dict({
        #     "image": spaces.Box(
        #         low=0,
        #         high=1,
        #         shape=(self.height, self.width, 1),
        #         dtype=np.float32,  # type: ignore[arg-type]
        #     ),  # RGB image
        #     "vector": spaces.Box(low=-10, high=10, shape=(64,), dtype=np.float32)  # 1D vector of size 10
        # })

    def observation(self, frame: np.ndarray) -> np.ndarray:
        """
        returns the current observation from a frame

        :param frame: environment frame
        :return: the observation
        """
        assert cv2 is not None, "OpenCV is not installed, you can do `pip install opencv-python`"
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        # frame = frame.astype(float) / 256
        # print(frame.shape)
        frame = frame.astype(float) / 255
        # frame = torch.tensor(frame, dtype=torch.float32).unsqueeze(0)
        frame = torch.tensor(frame, dtype=torch.float32)
        if self.state_background != None:

            pass
        else:
            # print(self.state_background_.shape)
            state_background = self.vae.generate(frame.unsqueeze(0).to(device)).cpu()
            self.state_background = state_background
            # print(frame.shape)
            mu, log_var = self.vae.encode(frame.unsqueeze(0).unsqueeze(0).to(device))
            state_background_latent = self.vae.reparameterize(mu, log_var).cpu()
            self.state_background_latent = state_background_latent
            # print(state_background_latent.shape)
            state_background_ = resize_bcg(state_background)
            self.state_background_ = state_background_
            print(self.state_background_.shape)
        frame = frame - self.state_background
        threshold = 0.1
        binary_image = (frame > threshold).cpu().numpy().astype(np.uint8)

        # observation = binary_image[0]
        # print(observation.shape)
        observation = torch.from_numpy(binary_image).float().cuda()

        observation = self.env_model.get_feature_encode(observation)

        self.f_deque.append(observation[0].detach().cpu().numpy())
        # print(observation[0].shape, observation[0])
        observation_ = np.stack(self.f_deque, axis=0)

        observation_ = torch.from_numpy(observation_).unsqueeze(0).float().to(device)
        # print(observation.shape)
        # print(ec1)
        state_background_ = torch.from_numpy(self.state_background_).float().to(device)
        ec3 = self.env_model.get_z_encode(observation_, state_background_).detach().cpu().numpy()
        # dd = encode_rare_value_by_blocks(observation)
        ec1 = self.state_background_latent.detach().cpu().numpy().squeeze()
        ec2 = observation_.detach().cpu().numpy().squeeze().flatten()
        # ec2 = observation_.detach().cpu().numpy().squeeze()[0]
        ec3 = ec3.squeeze()

        # print(ec1.shape, ec2.shape,ec3.shape)
        combined = np.hstack((ec2, ec1))
        observation = combined
        # print(combined.shape)
        # print(np.array(dd)*1.0/84)
        # print("Min:", observation.min(), "Max:", observation.max())
        # print(observation.shape)
        # if state_background_latent.shape != (64,):
        #     # print('fff', state_background_latent.shape)
        #     state_background_latent = state_background_latent.squeeze(0)
        # if observation.shape == (84, 84):
        #     observation = observation.unsqueeze(2)
        # elif observation.shape == (1, 84, 84):
        #     observation = np.transpose(observation, (1, 2, 0))
        # else:
        #     print(observation.shape, 'reer')
        # observations = {
        #     "image": np.float32(observation),
        #     "vector": state_background_latent.detach().numpy(),
        # }

        return observation

class AtariWrapper(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    """
    Atari 2600 preprocessings

    Specifically:

    * Noop reset: obtain initial state by taking random number of no-ops on reset.
    * Frame skipping: 4 by default
    * Max-pooling: most recent two observations
    * Termination signal when a life is lost.
    * Resize to a square image: 84x84 by default
    * Grayscale observation
    * Clip reward to {-1, 0, 1}
    * Sticky actions: disabled by default

    See https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/
    for a visual explanation.

    .. warning::
        Use this wrapper only with Atari v4 without frame skip: ``env_id = "*NoFrameskip-v4"``.

    :param env: Environment to wrap
    :param noop_max: Max number of no-ops
    :param frame_skip: Frequency at which the agent experiences the game.
        This correspond to repeating the action ``frame_skip`` times.
    :param screen_size: Resize Atari frame
    :param terminal_on_life_loss: If True, then step() returns done=True whenever a life is lost.
    :param clip_reward: If True (default), the reward is clip to {-1, 0, 1} depending on its sign.
    :param action_repeat_probability: Probability of repeating the last action
    """

    def __init__(
        self,
        env: gym.Env,
        noop_max: int = 30,
        frame_skip: int = 4,
        screen_size: int = 84,
        terminal_on_life_loss: bool = True,
        clip_reward: bool = True,
        action_repeat_probability: float = 0.0,
    ) -> None:
        if action_repeat_probability > 0.0:
            env = StickyActionEnv(env, action_repeat_probability)
        if noop_max > 0:
            env = NoopResetEnv(env, noop_max=noop_max)
        # frame_skip=1 is the same as no frame-skip (action repeat)
        if frame_skip > 1:
            env = MaxAndSkipEnv(env, skip=frame_skip)
        if terminal_on_life_loss:
            env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():  # type: ignore[attr-defined]
            env = FireResetEnv(env)
        env = WarpFrame(env, width=screen_size, height=screen_size)
        if clip_reward:
            env = ClipRewardEnv(env)

        super().__init__(env)



class AtariWrapper_bg(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    """
    Atari 2600 preprocessings

    Specifically:

    * Noop reset: obtain initial state by taking random number of no-ops on reset.
    * Frame skipping: 4 by default
    * Max-pooling: most recent two observations
    * Termination signal when a life is lost.
    * Resize to a square image: 84x84 by default
    * Grayscale observation
    * Clip reward to {-1, 0, 1}
    * Sticky actions: disabled by default

    See https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/
    for a visual explanation.

    .. warning::
        Use this wrapper only with Atari v4 without frame skip: ``env_id = "*NoFrameskip-v4"``.

    :param env: Environment to wrap
    :param noop_max: Max number of no-ops
    :param frame_skip: Frequency at which the agent experiences the game.
        This correspond to repeating the action ``frame_skip`` times.
    :param screen_size: Resize Atari frame
    :param terminal_on_life_loss: If True, then step() returns done=True whenever a life is lost.
    :param clip_reward: If True (default), the reward is clip to {-1, 0, 1} depending on its sign.
    :param action_repeat_probability: Probability of repeating the last action
    """

    def __init__(
        self,
        env: gym.Env,
        bg_model = None,
        ae_model = None,
        c_model = None,
        noop_max: int = 30,
        frame_skip: int = 4,
        screen_size: int = 84,
        terminal_on_life_loss: bool = True,
        clip_reward: bool = True,
        action_repeat_probability: float = 0.0,
    ) -> None:
        if action_repeat_probability > 0.0:
            env = StickyActionEnv(env, action_repeat_probability)
        if noop_max > 0:
            env = NoopResetEnv(env, noop_max=noop_max)
        # frame_skip=1 is the same as no frame-skip (action repeat)
        if frame_skip > 1:
            env = MaxAndSkipEnv(env, skip=frame_skip)
        if terminal_on_life_loss:
            env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():  # type: ignore[attr-defined]
            env = FireResetEnv(env)
        env = WarpFrame_bina_ecd_con(env, width=screen_size, height=screen_size, bg_model=bg_model, ae_model=ae_model, c_model=c_model)
        if clip_reward:
            env = ClipRewardEnv(env)

        super().__init__(env)
