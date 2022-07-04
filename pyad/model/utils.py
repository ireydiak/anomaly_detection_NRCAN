import torch
from typing import List
from torch import nn

activation_map = {
    "relu": nn.ReLU(),
    "tanh": nn.Tanh(),
    "sigmoid": nn.Sigmoid(),
    "leakyrelu": nn.LeakyReLU(),
    "gelu": nn.GELU()
}


def weights_init_xavier(m):
    # Copied from https://github.com/JohnEfan/PyTorch-ALAD/blob/6e7c4a9e9f327b5b08936376f59af2399d10dc9f/utils/utils.py#L4
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 and classname != 'Conv':
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find("Linear") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        m.bias.data.zero_()


def create_network(layers: List[List]):
    """
    This function builds a linear model whose units and layers depend on
    the passed @layers argument
    :param layers: a list of tuples indicating the layers architecture (in_neuron, out_neuron, activation_function)
    :return: a fully connected neural net (Sequentiel object)
    """
    net_layers = []
    for in_neuron, out_neuron, act_fn in layers:
        net_layers.append(nn.Linear(in_neuron, out_neuron))
        if act_fn:
            net_layers.append(act_fn)
    return nn.Sequential(*net_layers)
