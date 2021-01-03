import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
import gym
import torch
import torch.nn as nn
from network.network_base import Network
from utility import get_device



class ActorBase(nn.Module):
    def __init__(self,
                 observation_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 network_config=None,
                 device="auto"):
        super(ActorBase, self).__init__()

        self.device = get_device(device)
        obs_size = observation_space.shape[0]
        if isinstance(action_space, gym.spaces.Box):
            act_size = action_space.shape[0]
        else:
            raise NotImplemented

        self.hidden = nn.Sequential(
            nn.Linear(obs_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        ).to(self.device)
        self.mu = nn.Sequential(
            nn.Linear(64, act_size),
            nn.Tanh(),
        ).to(self.device)
        self.log_var = nn.Sequential(
            nn.Linear(64, 1),
        ).to(self.device)
        # if network_config is None:
        #     network_config = {
        #         "network_sizes": [128, 128],
        #         "activation_function": ["relu", "tanh"]
        #     }
        # super(ActorBase, self).__init__(network_config, feature_dim, act_size, device)
        # self.log_std = nn.Parameter(torch.randn(act_size)).to(self.device)

    def forward(self, x):
        return self.mu(self.hidden(x)), self.log_var(self.hidden(x))

    #
    # def log_std(self):
    #     return