import numpy as np
import gym
import torch
import torch.nn as nn
from network.network_base import Network

class CriticBase(Network):
    def __init__(self,
                 observation_space: gym.spaces.Space,
                 network_config=None,
                 device="auto"):
        obs_size = observation_space.shape[0]
        if network_config is None:
            network_config = {
                "network_sizes": [128, 128],
                "activation_function": ["relu", "identity"]
            }
        super(CriticBase, self).__init__(network_config, obs_size, 1, device)

    def forward(self, x):
        return super(CriticBase, self).forward(x)


if __name__ == "__main__":
    env = gym.make("Humanoid-v3")
    net_conf = {
        "network_sizes": [3, 3, 3],
        "activation_function": ["relu", "tanh"]
    }
    actor = CriticBase(env.observation_space,  net_conf)
