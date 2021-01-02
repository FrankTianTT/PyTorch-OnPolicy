import numpy as np
import gym
import torch
import torch.nn as nn
from network.network_base import Network

class CriticBase(Network):
    def __init__(self,
                 feature_dim: int,
                 network_config=None,
                 device="auto"):
        obs_size = feature_dim
        if network_config is None:
            network_config = {
                "network_sizes": [128, 128],
                "activation_function": ["relu", "identity"]
            }
        super(CriticBase, self).__init__(network_config, obs_size, 1, device)

    def forward(self, x):
        return super(CriticBase, self).forward(x)