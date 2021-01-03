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
        # reward and steps of one rollout
        self.sum_reward = 0
        self.sum_steps = 0
        self._last_dones = None
        self.exp_buffer = list()

    @staticmethod
    def tanh2bound(action: float, low, high):
        """
        transfer action from [-1, 1] to [low, high]
        :param action: action get from actor network
        :param low: env action-space low bound
        :param high: env action-space high bound
        :return: new action
        """
        return (action / 2) * (high - low) + (high + low) / 2


    def collect(self, batch, now_steps, logger):
        assert self._last_obs is not None, "No previous observation was provided"
        self.last_obss = []
        self.actions = []
        self.values = []
        self.rewards = []
        self.new_obss = []
        self.not_dones = []
        self.ref_values = []
        n_steps = 0

        while n_steps < batch:
            n_steps += 1
            with torch.no_grad():
                # Convert to pytorch tensor
                obs_tensor = torch.as_tensor(self._last_obs).to(self.device).to(torch.float32)
                feature = self.features_extractor(obs_tensor)
                action = self.actor(feature)
                value = self.critic(feature)

            action = action.cpu().numpy()
            if isinstance(self.env.action_space, gym.spaces.Box):
                action = self.tanh2bound(action, self.env.action_space.low, self.env.action_space.high)
            else:
                raise TypeError
            new_obs, reward, done, info = self.env.step(action)
            self.sum_reward += reward
            self.sum_steps += 1

            self.last_obss.append(self._last_obs)
            self.values.append(value)
            self.actions.append(action)
            self.rewards.append(reward)
            self.not_dones.append(not done)
            self.new_obss.append(new_obs)

            self._last_obs = new_obs

            if done:
                self._last_obs = self.env.reset()
                logger.log_var('run/steps', self.sum_steps, now_steps + n_steps)
                logger.log_var('run/reward', self.sum_reward, now_steps + n_steps)
                self.sum_reward = 0
                self.sum_steps = 0

        self.values.pop(0)
        with torch.no_grad():
            obs_tensor = torch.as_tensor(self._last_obs).to(self.device).to(torch.float32)
            feature = self.features_extractor(obs_tensor)
            value = self.critic(feature)
        self.values.append(value)

    def get(self, batch):
        rewards_np = np.array(self.rewards[-batch:], dtype=np.float32)
        not_dones_np = np.array(self.not_dones[-batch:])
        values_np = np.array(self.values[-batch:])
        ref_values = rewards_np
        ref_values[not_dones_np] = ref_values[not_dones_np] + values_np[not_dones_np] * self.gamma
        last_obss_np = np.array(self.last_obss[-batch:])
        actions_np = np.array(self.actions[-batch:])
        return torch.from_numpy(last_obss_np).to(torch.float32).to(self.device), \
               torch.from_numpy(actions_np).to(torch.float32).to(self.device), \
               torch.from_numpy(ref_values).to(torch.float32).to(self.device)

if __name__ == '__main__':
    pass