import numpy as np
import torch
import torch.nn as nn
from utility import get_device

class Network(nn.Module):
    def __init__(self,
                 network_config,
                 input_size,
                 output_size,
                 device="auto"):
        super(Network, self).__init__()
        self.device = get_device(device)

        if type(input_size) is int:
            sample_input = torch.rand(1, input_size)
        else:
            sample_input = torch.rand(1, *input_size)

        self.networks = nn.Sequential(*self.construct_networks(network_config, sample_input, output_size)).\
            to(self.device)

    @staticmethod
    def get_network(param, is_conv=False):
        if is_conv:
            assert len(param) == 4, "unrecognized parameter of network: {}, which should be in_channels, kernel_size," \
                                    " stride, out_channels.".format(param)
            in_channels, kernel_size, stride, out_channels = param
            return torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        else:
            assert len(param) == 2, "unrecognized parameter of network: {}, which should be in_dim, out_dim.".\
                format(param)
            in_dim, out_dim = param
            return torch.nn.Linear(in_dim, out_dim)

    @staticmethod
    def get_activation_fn(name):
        name = name.lower()
        if name == "tanh":
            activation_fn = torch.nn.Tanh
        elif name == "sigmoid":
            activation_fn = torch.nn.Sigmoid
        elif name == 'relu':
            activation_fn = torch.nn.ReLU
        elif name == 'identity':
            activation_fn = torch.nn.Identity
        else:
            raise NotImplementedError("activation function {} not implemented.".format(name))

        return activation_fn

    def construct_networks(self, network_config, sample_input, out_size):
        networks_list = []
        # load config from network configs
        network_sizes = network_config['network_sizes']
        hidden_activation_fn_name, out_activation_fn_name = network_config['activation_function']
        hidden_activation_fn = self.get_activation_fn(hidden_activation_fn_name)
        out_activation_fn = self.get_activation_fn(out_activation_fn_name)
        # memorize in_channel for conv and dim for linear layer
        for network_param in network_sizes:
            if type(network_param) == list:
                # conv net case
                temp = sample_input.shape[1]
                expanded_param = [temp] + network_param
                curr_network = self.get_network(expanded_param)
            elif type(network_param) == int:
                # flatten sample_input if from image to vector
                if len(sample_input.shape) == 4:
                    sample_input = sample_input.view(1, -1)
                temp = sample_input.shape[1]
                expanded_param = [temp, network_param]
                curr_network = self.get_network(expanded_param)
            else:
                raise NotImplementedError("unrecognized parameter of network.")
            sample_input = curr_network(sample_input)
            networks_list.append(curr_network)
            networks_list.append(hidden_activation_fn())
        # construct output layer
        assert (len(sample_input.shape) == 2)
        out_network_param = [sample_input.shape[1], out_size]
        out_layer = self.get_network(out_network_param)
        networks_list.append(out_layer)
        networks_list.append(out_activation_fn())
        return networks_list

    def forward(self, x):
        return self.networks(x)

    def __str__(self):
        return str(self.networks)


if __name__ == "__main__":
    net_conf = {
        "network_sizes": [3, 3, 3],
        "activation_function": ["relu", "tanh"]
    }
    net = Network(net_conf, [10], 10)
