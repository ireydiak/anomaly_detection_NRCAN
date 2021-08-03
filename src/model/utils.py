from typing import List, Tuple
import torch.nn as nn


def create_network(layers: List[Tuple]):
    net_layers = []
    for in_neuron, out_neuron, act_fn in layers:
        if in_neuron and out_neuron:
            net_layers.append(nn.Linear(in_neuron, out_neuron))
        net_layers.append(act_fn)
    return nn.Sequential(*net_layers)