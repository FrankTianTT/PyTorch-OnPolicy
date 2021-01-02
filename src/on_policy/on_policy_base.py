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
from network.features_extractor import FeaturesExtractor
from buffer.on_policy_buffer import OnPolicyBuffer
from utility import get_device
from on_policy.constant import *


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
                 device="auto",):
        self.env = env
        self.batch_size = batch_size
        self.network_config = network_config
        self.learning_rate_actor = learning_rate_actor
        self.learning_rate_critic = learning_rate_critic
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.entropy_beta = entropy_beta
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

    def build_network(self):
        features_extractor_config = self.network_config['features_extractor_config']
        feature_dim = self.network_config['feature_dim']
        critic_network_config = self.network_config['critic_network_config']
        actor_network_config = self.network_config['actor_network_config']
        observation_space = self.env.observation_space
        action_space = self.env.action_space
        self.features_extractor = FeaturesExtractor(observation_space, feature_dim, features_extractor_config,
                                                    self.device)
        self.critic = CriticBase(feature_dim, critic_network_config, self.device)
        self.actor = ActorBase(feature_dim, action_space, actor_network_config, self.device)

        self.actor_optimizer = optim.Adam(params=chain(self.actor.parameters(), self.features_extractor.parameters()),
                                          lr=self.learning_rate_actor)
        self.critic_optimizer = optim.Adam(params=chain(self.critic.parameters(), self.features_extractor.parameters()),
                                           lr=self.learning_rate_critic)

    def build_buffer(self):
        self.buffer = OnPolicyBuffer(self.env, self.features_extractor, self.actor, self.critic, self.device,
                                     self.gae_lambda, self.gamma)

    def predict(self, observation):
        with torch.no_grad():
            observation = torch.as_tensor(observation).to(self.device).to(torch.float32)
            feature = self.features_extractor(observation)
            action = self.actor(feature)
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
        return 'features_extractor:\n{}\nactor:\n{}\ncritic\n{}'\
            .format(str(self.features_extractor), str(self.actor), str(self.critic))

if __name__ == '__main__':
    env = gym.make("Humanoid-v3")
    algro = OnPolicyBase(env)