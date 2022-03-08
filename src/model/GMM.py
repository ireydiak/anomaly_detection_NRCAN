from typing import List, Tuple

import torch.nn as nn
import torch


class GMM(nn.Module):
    def __init__(self, layers):
        super(GMM, self).__init__()
        self.net = self.create_network(layers)
        self.K = layers[-1][1]

    def create_network(self, layers: List[Tuple]):
        net_layers = []
        for in_neuron, out_neuron, act_fn in layers:
            if in_neuron and out_neuron:
                net_layers.append(nn.Linear(in_neuron, out_neuron))
            net_layers.append(act_fn)
        return nn.Sequential(*net_layers)

    def forward(self, x: torch.Tensor):
        return self.net(x)
