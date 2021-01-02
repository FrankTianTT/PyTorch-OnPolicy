import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
import gym
import torch
import torch.nn as nn
from network.network_base import Network


class ActorBase(Network):
    def __init__(self,
                 observation_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 network_config=None,
                 device="auto"):
        obs_size = observation_space.shape[0]
        act_size = action_space.shape[0]
        if network_config is None:
            network_config = {
                "network_sizes": [128, 128],
                "activation_function": ["relu", "tanh"]
            }
        super(ActorBase, self).__init__(network_config, obs_size, act_size, device)
        self.log_std = nn.Parameter(torch.zeros(act_size)).to(self.device)

    def forward(self, x):
        return super(ActorBase, self).forward(x)

    def predict(self, observation):
        observation = torch.from_numpy(observation).to(torch.float32).to(self.device)
        action = self.networks(observation)
        action = action.detach().cpu().numpy()
        return action


if __name__ == "__main__":
    env = gym.make("Humanoid-v3")
    net_conf = {
        "network_sizes": [128, 128],
        "activation_function": ["relu", "tanh"]
    }
    actor = ActorBase(env.observation_space, env.action_space, net_conf)
    print(actor.networks)

    obs = env.reset()
    while True:
        action = actor.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()
