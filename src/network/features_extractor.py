import gym
from network.network_base import Network

class FeaturesExtractor(Network):
    def __init__(self,
                 observation_space: gym.spaces.Space,
                 feature_dim: int = 128,
                 network_config=None,
                 device="auto"):
        obs_size = observation_space.shape[0]
        if network_config is None:
            network_config = {
                "network_sizes": [128, 128],
                "activation_function": ["relu", "relu"]
            }
        super(FeaturesExtractor, self).__init__(network_config, obs_size, feature_dim, device)

    def forward(self, x):
        return super(FeaturesExtractor, self).forward(x)
