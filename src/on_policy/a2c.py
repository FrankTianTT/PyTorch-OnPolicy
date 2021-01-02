import torch
import gym
import numpy as np
from on_policy.on_policy_base import OnPolicyBase


class A2C(OnPolicyBase):
    def __int__(self,
                 env: gym.Env,
                 critic_network_config=None,
                 actor_network_config=None,
                 device="auto",):
        super(A2C, self).__init__(env, critic_network_config, actor_network_config, device)

    def train(self):
        pass


if __name__ == '__main__':
    env = gym.make("Humanoid-v3")
    model = A2C(env)

    model.learn()

    # obs = env.reset()
    # while True:
    #     action = model.predict(obs)
    #     obs, reward, done, info = env.step(action)
    #     env.render()
    #     if done:
    #         obs = env.reset()