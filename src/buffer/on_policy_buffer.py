import gym
import torch
import numpy as np
from utility import get_device


class OnPolicyBuffer(object):
    def __init__(self,
                 env: gym.Env,
                 features_extractor,
                 actor,
                 critic,
                 device="auto",
                 gae_lambda: float = 1,
                 gamma: float = 0.99):
        self.env = env
        self.features_extractor = features_extractor
        self.actor = actor
        self.critic = critic
        self.device = get_device(device)
        self.gae_lambda = gae_lambda
        self.gamma = gamma

        self._last_obs = env.reset()
        self._last_dones = None
        self.exp_buffer = list()
        pass

    def collect(self, batch):
        assert self._last_obs is not None, "No previous observation was provided"
        n_steps = 0

        while n_steps < batch:
            with torch.no_grad():
                # Convert to pytorch tensor
                obs_tensor = torch.as_tensor(self._last_obs).to(self.device).to(torch.float32)
                feature = self.features_extractor(obs_tensor)
                action = self.actor(feature)
                value = self.critic(feature)
            action = action.cpu().numpy()

            if isinstance(self.env.action_space, gym.spaces.Box):
                action = np.clip(action, self.env.action_space.low, self.env.action_space.high)

            new_obs, reward, done, info = self.env.step(action)

            n_steps += 1

            if isinstance(self.env.action_space, gym.spaces.Discrete):
                action = action.reshape(-1, 1)
            self.add(self._last_obs, action, reward, new_obs, done, value)
            self._last_obs = new_obs
            self._last_done = done

        with torch.no_grad():
            # Compute value for the last time-step
            obs_tensor = torch.as_tensor(new_obs).to(self.device).to(torch.float32)
            value = self.critic(self.features_extractor(obs_tensor))

    def get(self, batch_size: int, critic):
        assert batch_size <= len(self.exp_buffer), ""
        last_obss = []
        actions = []
        rewards = []
        new_obss = []
        dones = []
        batch_data = self.exp_buffer[-batch_size:]
        for data in batch_data:
            last_obss.append(data['last_obs'])
            actions.append(data[actions])
            rewards.append(data['reward'])
            new_obss.append(data['new_obs'])
            dones.append(data['done'])

    def add(self, last_obs, action, reward, new_obs, done):
        self.exp_buffer.append({'last_obs': last_obs,
                             'action': action,
                             'reward': reward,
                             'new_obs': new_obs,
                             'done': done})

    def calculate(self):
        pass


if __name__ == '__main__':
    pass