import gym
import torch
import numpy as np
from network.actor import ActorBase
from network.critic import CriticBase


class OnPolicyBase(object):
    def __init__(self,
                 env: gym.Env,
                 critic_network_config=None,
                 actor_network_config=None,
                 device="auto",):
        self.env = env
        self.critic_network_config = critic_network_config
        self.actor_network_config = actor_network_config
        self.device = device

        self.critic_net = self.make_critic_net()
        self.actor_net = self.make_actor_net()
        # TODO

    def make_critic_net(self):
        observation_space = self.env.observation_space
        return CriticBase(observation_space, self.critic_network_config, self.device)

    def make_actor_net(self):
        observation_space = self.env.observation_space
        action_space = self.env.action_space
        return ActorBase(observation_space, action_space, self.actor_network_config, self.device)

    def predict(self, observation):
        return self.actor_net.predict(observation)


if __name__ == '__main__':
    # env = gym.make("Humanoid-v3")
    # TODO
    pass