import gym
import torch
import numpy as np
from utility import get_device


class OnPolicyBuffer(object):
    def __init__(self,
                 env: gym.Env,
                 actor,
                 critic,
                 device="auto",
                 reward_step: int = 5,
                 gae_lambda: float = 1,
                 gamma: float = 0.99):
        self.env = env
        self.actor = actor
        self.critic = critic
        self.device = get_device(device)
        assert reward_step > 0, "reward_step must greater than 0"
        self.reward_step = reward_step
        self.gae_lambda = gae_lambda
        self.gamma = gamma

        self.last_obss = []
        self.actions = []
        self.ref_values = []
        self.new_obss = []
        self.dones = []

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

    def action_transfer(self, action):
        """
        transfer action from a NN, to env-style
        :param action:
        :return:
        """
        action = action.detach().cpu().numpy()
        if isinstance(self.env.action_space, gym.spaces.Box):
            action = self.tanh2bound(action, self.env.action_space.low, self.env.action_space.high)
        else:
            # TODO
            raise NotImplementedError
        return action


    def setup_collect(self):
        """
        collect initial "last_obss" and "reward" of "reward_step" steps
        :return: none
        """
        for i in range(self.reward_step):
            obs_tensor = torch.as_tensor(self._last_obs).to(self.device).to(torch.float32)
            action, _ = self.actor(obs_tensor)
            action = self.action_transfer(action)
            new_obs, reward, done, info = self.env.step(action)
            self.last_obss.append(self._last_obs)
            self.actions.append(action)
            self.ref_values.append(reward)
            for j in range(i):
                self.ref_values[j] += reward * self.gamma ** (i - j)
            self.dones.append(done)

            self._last_obs = new_obs

            if done:
                self._last_obs = self.env.reset()
                self.sum_reward = 0
                self.sum_steps = 0

    def is_ever_done(self, n_steps, reward_step_before):
        for i in range(reward_step_before):
            if self.dones[n_steps + i]:
                return True
        return False

    def collect(self, batch, now_steps, logger):
        """
        collect a batch of data from gym-env.
        :param batch: batch-size of data
        :param now_steps: use for log
        :param logger: logger object
        :return: none
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # setup some buffer when "collect" was first called
        if len(self.last_obss) == 0:
            self.setup_collect()

        n_steps = 0
        self.last_obss = self.last_obss[-self.reward_step:]
        self.actions = self.last_obss[-self.reward_step:]
        self.dones = self.dones[-self.reward_step:]
        self.ref_values = self.ref_values[-self.reward_step:]
        self.new_obss = []


        while n_steps < batch:
            with torch.no_grad():
                # Convert to pytorch tensor
                obs_tensor = torch.as_tensor(self._last_obs).to(self.device).to(torch.float32)
                action, _ = self.actor(obs_tensor)
                value = self.critic(obs_tensor)

            action = self.action_transfer(action)
            new_obs, reward, done, info = self.env.step(action)

            # render env, use for debug
            # self.env.render()
            self.sum_reward += reward
            self.sum_steps += 1

            # there should be "reward_step" elements in "last_obss", "actions", "ref_values" and "not_dones"
            self.last_obss.append(self._last_obs)
            self.actions.append(action)
            self.dones.append(done)
            self.new_obss.append(new_obs)
            self.ref_values.append(reward)
            if not self.is_ever_done(n_steps, self.reward_step):
                self.ref_values[n_steps] += value * self.gamma ** self.reward_step
            for j in range(self.reward_step - 1):
                if not self.is_ever_done(n_steps, self.reward_step - 1 - j):
                    self.ref_values[n_steps + j + 1] += reward * self.gamma ** (self.reward_step - j - 1)

            self._last_obs = new_obs

            if done:
                self._last_obs = self.env.reset()
                logger.log_var('run/steps', self.sum_steps, now_steps + n_steps)
                logger.log_var('run/reward', self.sum_reward, now_steps + n_steps)
                self.sum_reward = 0
                self.sum_steps = 0

            n_steps += 1

        return torch.Tensor(self.last_obss[-batch:]).to(torch.float32).to(self.device), \
               torch.Tensor(self.actions[-batch:]).to(torch.float32).to(self.device), \
               torch.Tensor(self.ref_values[-batch:]).to(torch.float32).to(self.device)

if __name__ == '__main__':
    pass