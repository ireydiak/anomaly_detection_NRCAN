from typing import List, Tuple
import torch.nn as nn

def create_network(layers: List[Tuple]) -> nn.Sequential:
    net_layers = []
    for layer in layers:
        in_neuron, out_neuron, act_fn = layer[0], layer[1], layer[2]
        net_layers.append(nn.Linear(in_neuron, out_neuron))
        net_layers.append(act_fn)
    return nn.Sequential(*net_layers)
