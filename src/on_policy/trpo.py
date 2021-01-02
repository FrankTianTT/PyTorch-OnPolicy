import torch
import gym
import numpy as np
from on_policy.on_policy_base import OnPolicyBase


class TRPO(OnPolicyBase):
    def __int__(self,
                env: gym.Env,
                critic_network_config=None,
                actor_network_config=None,
                device="auto", ):
        super(TRPO, self).__init__(env, critic_network_config, actor_network_config, device)

    def train(self):
        pass


if __name__ == '__main__':
    env = gym.make("Humanoid-v3")
    model = TRPO(env)

    obs = env.reset()
    while True:
        action = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()