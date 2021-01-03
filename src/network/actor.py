import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
import gym
import torch
import torch.nn as nn
from network.network_base import Network


class ActorBase(nn.Module):
    def __init__(self,
                 observation_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 network_config=None,
                 device="auto"):
        super(ActorBase, self).__init__()
        obs_size = observation_space.shape[0]
        if isinstance(action_space, gym.spaces.Box):
            act_size = action_space.shape[0]
        else:
            raise NotImplemented

        self.mu = nn.Sequential(
            nn.Linear(obs_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, act_size),
            nn.Tanh(),
        )
        self.log_std = nn.Parameter(torch.zeros(act_size))
        # if network_config is None:
        #     network_config = {
        #         "network_sizes": [128, 128],
        #         "activation_function": ["relu", "tanh"]
        #     }
        # super(ActorBase, self).__init__(network_config, feature_dim, act_size, device)
        # self.log_std = nn.Parameter(torch.randn(act_size)).to(self.device)

    def forward(self, x):
        return self.mu(x)

    #
    # def log_std(self):
    #     return