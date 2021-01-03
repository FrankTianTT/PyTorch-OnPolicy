import numpy as np
import gym
import torch
import torch.nn as nn
from network.network_base import Network
from utility import get_device


class CriticBase(nn.Module):
    def __init__(self,
                 observation_space: gym.spaces.Space,
                 network_config=None,
                 device="auto"):
        super(CriticBase, self).__init__()

        self.device = get_device(device)
        obs_size = observation_space.shape[0]
        self.value = nn.Sequential(
            nn.Linear(obs_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        ).to(self.device)
        # if network_config is None:
        #     network_config = {
        #         "network_sizes": [128, 128],
        #         "activation_function": ["relu", "identity"]
        #     }
        # super(CriticBase, self).__init__(network_config, obs_size, 1, device)

    def forward(self, x):
        return self.value(x)