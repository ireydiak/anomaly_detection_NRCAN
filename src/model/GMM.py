from typing import List, Tuple

import torch.nn as nn
import torch


class GMM(nn.Module):
    def __init__(self, layers=None):
        super(GMM, self).__init__()
        default_layers = [(3, 10, nn.Tanh()), (None, None, nn.Dropout(0.5)), (10, 4, nn.Softmax(dim=-1))]
        self.net = self.create_network(layers or default_layers)

    def create_network(self, layers: List[Tuple]):
        net_layers = []
        for in_neuron, out_neuron, act_fn in layers:
            if in_neuron and out_neuron:
                net_layers.append(nn.Linear(in_neuron, out_neuron))
            net_layers.append(act_fn)
        return nn.Sequential(*net_layers)

    def forward(self, x: torch.Tensor):
        return self.net(x)
