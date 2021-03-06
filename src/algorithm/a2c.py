import torch
import torch.nn.functional as F
import gym
import math
import numpy as np

from algorithm.on_policy_base import OnPolicyBase
from algorithm.constant import *


class A2C(OnPolicyBase):
    def __int__(self,
                env: gym.Env,
                batch_size: int = 32,
                network_config=NETWORK_CONFIG,
                learning_rate_actor: float = 1e-5,
                learning_rate_critic: float = 1e-3,
                gae_lambda: float = 1,
                gamma: float = 0.99,
                entropy_beta: float = 1e-3,
                log_path: str = '',
                log_prefix: str = 'a2c',
                log_freq: int = 50,
                device="auto",):
        super(A2C, self).__init__(env,
                                  batch_size,
                                  network_config,
                                  learning_rate_actor,
                                  learning_rate_critic,
                                  gae_lambda=gae_lambda,
                                  gamma=gamma,
                                  entropy_beta=entropy_beta,
                                  log_path=log_path,
                                  log_prefix=log_prefix,
                                  log_freq=log_freq,
                                  device=device)

    def learn(self, total_steps):
        super(A2C, self).learn(total_steps)

    @staticmethod
    def calc_logprob(mu, log_var, actions):
        p1 = - torch.pow(mu - actions, 2) / (2 * torch.exp(log_var).clamp(min=1e-3))
        p2 = - torch.log(torch.sqrt(2 * math.pi * torch.exp(log_var)))
        return p1 + p2

    def train(self):
        """
        "train" function for A2C algorithm, which should be a part of "learn" function
        :return: none
        """
        obss, actions, ref_values = self.buffer.collect(self.batch_size, self.now_steps, self.logger)

        # train for critic
        self.critic_optimizer.zero_grad()
        obs_values = self.critic(obss)
        value_lass = F.mse_loss(obs_values.squeeze(-1), ref_values)
        value_lass.backward()
        self.critic_optimizer.step()

        # train for actor
        self.actor_optimizer.zero_grad()
        mu, log_var = self.actor(obss)
        advantage = ref_values.unsqueeze(dim=-1) - obs_values.detach()
        # calculate pi(action|mu, std), pi is a gaussian distribution
        log_prob = advantage * self.calc_logprob(mu, log_var, actions)
        policy_loss = - log_prob.mean()
        # entropy = (log(2 * pi * sigma ** 2) + 1)/2
        entropy_loss = self.entropy_beta * (-(torch.log(2 * np.pi * torch.exp(log_var)) + 1) / 2).mean()
        actor_total_loss = policy_loss + entropy_loss
        actor_total_loss.backward()
        self.actor_optimizer.step()

        self.logger.track('train/advantage', advantage)
        self.logger.track('train/obs_value', obs_values)
        self.logger.track('train/ref_values', ref_values)
        self.logger.track('loss/entropy_loss', entropy_loss)
        self.logger.track('loss/policy_loss', policy_loss)
        self.logger.track('loss/value_lass', value_lass)
        self.logger.track('loss/actor_total_loss', actor_total_loss)
        self.logger.write(self.now_steps)

if __name__ == '__main__':
    env = gym.make("Humanoid-v3")
    model = A2C(env)

    model.learn(1000)

    # obs = env.reset()
    # while True:
    #     action = model.predict(obs)
    #     obs, reward, done, info = env.step(action)
    #     env.render()
    #     if done:
    #         obs = env.reset()