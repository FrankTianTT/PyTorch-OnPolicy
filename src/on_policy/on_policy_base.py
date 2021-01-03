import gym
from abc import ABC
from abc import abstractmethod
import torch
import torch.optim as optim
import numpy as np
from typing import Union
from itertools import chain

from network.actor import ActorBase
from network.critic import CriticBase
from buffer.on_policy_buffer import OnPolicyBuffer
from utility import get_device
from on_policy.constant import *
from logger import Logger

class OnPolicyBase(ABC):
    def __init__(self,
                 env: gym.Env,
                 batch_size: int = 32,
                 network_config=NETWORK_CONFIG,
                 learning_rate_actor: float = 1e-5,
                 learning_rate_critic: float = 1e-3,
                 gae_lambda: float = 1,
                 gamma: float = 0.99,
                 entropy_beta: float = 1e-3,
                 log_path: str = '',
                 log_prefix: str = 'on-policy',
                 log_freq: int = 50,
                 device="auto",):
        self.env = env
        self.batch_size = batch_size
        self.network_config = network_config
        self.learning_rate_actor = learning_rate_actor
        self.learning_rate_critic = learning_rate_critic
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.entropy_beta = entropy_beta
        self.log_path = log_path
        self.log_prefix = log_prefix
        self.log_freq = log_freq
        self.device = get_device(device)

        self.features_extractor = None
        self.actor = None
        self.critic = None
        self.actor_optimizer = None
        self.critic_optimizer = None
        self._last_obs = None
        self._last_dones = None
        self.buffer = None
        self.now_steps = 0

        self.build_network()
        self.build_buffer()
        self.logger = Logger(self.log_path, prefix=self.log_prefix, log_freq=self.log_freq)

    def build_network(self):
        critic_network_config = self.network_config['critic_network_config']
        actor_network_config = self.network_config['actor_network_config']
        observation_space = self.env.observation_space
        action_space = self.env.action_space
        self.critic = CriticBase(observation_space, critic_network_config, self.device)
        self.actor = ActorBase(observation_space, action_space, actor_network_config, self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.learning_rate_critic)

    def build_buffer(self):
        self.buffer = OnPolicyBuffer(self.env, self.actor, self.critic, self.device,
                                     self.gae_lambda, self.gamma)

    def predict(self, observation):
        with torch.no_grad():
            observation = torch.as_tensor(observation).to(self.device).to(torch.float32)
            action, _ = self.actor(observation)
        action = action.cpu().numpy()
        return action

    @abstractmethod
    def train(self):
        raise NotImplementedError

    def learn(self, total_steps):
        this_learn_steps = 0
        while this_learn_steps < total_steps:
            this_learn_steps += self.batch_size
            self.now_steps += self.batch_size
            self.train()

    def __str__(self):
        return 'actor:\n{}\ncritic\n{}'\
            .format(str(self.actor), str(self.critic))

if __name__ == '__main__':
    env = gym.make("Humanoid-v3")
    algro = OnPolicyBase(env)