import gym
from abc import ABC
from abc import abstractmethod
import torch
import numpy as np
from network.actor import ActorBase
from network.critic import CriticBase
from network.features_extractor import FeaturesExtractor
from typing import Union

NETWORK_CONFIG = {
    'features_extractor_config': {
        "network_sizes": [128, 128],
        "activation_function": ["relu", "relu"]
    },
    'feature_dim': 128,
    'critic_network_config': {
        "network_sizes": [128],
        "activation_function": ["relu", "relu"]
    },
    'actor_network_config': {
        "network_sizes": [128],
        "activation_function": ["relu", "tanh"]
    }
}

class OnPolicyBase(ABC):
    def __init__(self,
                 env: gym.Env,
                 n_steps: int = 2048,
                 network_config=NETWORK_CONFIG,
                 device="auto",):
        self.env = env
        self.n_steps = n_steps
        self.network_config = network_config
        self.device = self.get_device(device)

        self.features_extractor = None
        self.actor = None
        self.critic = None
        self._last_obs = None
        self._last_dones = None

        self.num_timesteps = 0

        self.build_network(self.network_config)
        # TODO

    def build_network(self, network_config):
        features_extractor_config = network_config['features_extractor_config']
        feature_dim = network_config['feature_dim']
        critic_network_config = network_config['critic_network_config']
        actor_network_config = network_config['actor_network_config']
        observation_space = self.env.observation_space
        action_space = self.env.action_space
        self.features_extractor = FeaturesExtractor(observation_space, feature_dim, features_extractor_config,
                                                    self.device)
        self.critic = CriticBase(feature_dim, critic_network_config, self.device)
        self.actor = ActorBase(feature_dim, action_space, actor_network_config, self.device)


    def predict(self, observation):
        with torch.no_grad():
            observation = torch.as_tensor(observation).to(self.device).to(torch.float32)
            feature = self.features_extractor(observation)
            action = self.actor(feature)
        action = action.cpu().numpy()
        return action

    def collect_rollouts(self, env, buffer, n_rollout_steps: int):
        assert self._last_obs is not None, "No previous observation was provided"
        n_steps = 0
        buffer.reset()

        while n_steps < n_rollout_steps:
            with torch.no_grad():
                # Convert to pytorch tensor
                obs_tensor = torch.as_tensor(self._last_obs).to(self.device)
                features = self.features_extractor(obs_tensor)
                actions = self.actor(features)
                values = self.critic(features)
            actions = actions.cpu().numpy()

            if isinstance(self.env.action_space, gym.spaces.Box):
                actions = np.clip(actions, self.env.action_space.low, self.env.action_space.high)

            new_obs, rewards, dones, infos = env.step(actions)

            self.num_timesteps += env.num_envs
            n_steps += 1

            if isinstance(self.env.action_space, gym.spaces.Discrete):
                actions = actions.reshape(-1, 1)
            buffer.add(self._last_obs, actions, rewards, self._last_dones, values)
            self._last_obs = new_obs
            self._last_dones = dones

        with torch.no_grad():
            # Compute value for the last timestep
            obs_tensor = torch.as_tensor(new_obs).to(self.device)
            values = self.critic(self.features_extractor(obs_tensor))

        buffer.compute_returns_and_advantage(last_values=values, dones=dones)

    @staticmethod
    def get_device(device: Union[torch.device, str] = "auto") -> torch.device:
        # Cuda by default
        if device == "auto":
            device = "cuda"
        # Force conversion to th.device
        device = torch.device(device)

        # Cuda not available
        if device.type == torch.device("cuda").type and not torch.cuda.is_available():
            return torch.device("cpu")

        return device

    @abstractmethod
    def train(self):
        pass

    def learn(self):
        pass

    def __str__(self):
        return 'features_extractor:\n{}\nactor:\n{}\ncritic\n{}'\
            .format(str(self.features_extractor), str(self.actor), str(self.critic))

if __name__ == '__main__':
    env = gym.make("Humanoid-v3")
    algro = OnPolicyBase(env)