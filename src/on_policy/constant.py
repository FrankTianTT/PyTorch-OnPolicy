NETWORK_CONFIG = {
    'features_extractor_config': {
        "network_sizes": [128],
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