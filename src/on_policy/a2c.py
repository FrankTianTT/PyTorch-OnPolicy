import torch
import torch.nn.functional as F
import gym
import numpy as np

from on_policy.on_policy_base import OnPolicyBase
from on_policy.constant import *


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
                device="auto",):
        super(A2C, self).__init__(env,
                                  batch_size,
                                  network_config,
                                  learning_rate_actor,
                                  learning_rate_critic,
                                  gae_lambda,
                                  gamma,
                                  entropy_beta,
                                  device)

    def learn(self, total_steps):
        super(A2C, self).learn(total_steps)

    @staticmethod
    def calc_logprob(mu, logstd, actions):
        p1 = - ((mu - actions) ** 2) / (2 * torch.exp(logstd).clamp(min=1e-3))
        p2 = - torch.log(torch.sqrt(2 * np.pi * torch.exp(logstd)))
        return p1 + p2

    def train(self):
        self.buffer.collect(self.batch_size)
        obss, actions, ref_values = self.buffer.get(self.batch_size)

        self.critic_optimizer.zero_grad()
        obs_values = self.critic(self.features_extractor(obss))
        value_lass = F.mse_loss(obs_values.squeeze(-1), ref_values)
        value_lass.backward()
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        mu = self.actor(self.features_extractor(obss))
        advantage = ref_values.unsqueeze(dim=-1) - obs_values.detach()
        log_prob = advantage * self.calc_logprob(mu, self.actor.log_std, actions)
        policy_loss = -log_prob.mean()
        entropy_loss = self.entropy_beta * (-(torch.log(2 * np.pi * torch.exp(self.actor.log_std)) + 1) / 2).mean()
        actor_total_loss = policy_loss + entropy_loss
        actor_total_loss.backward()
        self.actor_optimizer.step()

        self.logger.log_var('train/advantage', advantage, self.now_steps)
        self.logger.log_var('train/obs_value', obs_values, self.now_steps)
        self.logger.log_var('train/ref_values', ref_values, self.now_steps)
        self.logger.log_var('train/entropy_loss', entropy_loss, self.now_steps)
        self.logger.log_var('train/policy_loss', policy_loss, self.now_steps)
        self.logger.log_var('train/value_lass', value_lass, self.now_steps)
        self.logger.log_var('train/actor_total_loss', actor_total_loss, self.now_steps)

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