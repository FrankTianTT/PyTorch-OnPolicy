import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
import gym
import torch
import torch.nn as nn
from network.network_base import Network


class ActorBase(Network):
    def __init__(self,
                 feature_dim: int,
                 action_space: gym.spaces.Space,
                 network_config=None,
                 device="auto"):
        act_size = action_space.shape[0]
        if network_config is None:
            network_config = {
                "network_sizes": [128, 128],
                "activation_function": ["relu", "tanh"]
            }
        super(ActorBase, self).__init__(network_config, feature_dim, act_size, device)
        self.log_std = nn.Parameter(torch.zeros(act_size)).to(self.device)

    def forward(self, x):
        return super(ActorBase, self).forward(x)
