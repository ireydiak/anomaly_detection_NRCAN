from typing import Tuple, List

import torch.nn as nn


class AutoEncoder(nn.Module):
    """
    Implements a Deep Auto Encoder
    """

    def __init__(self, enc_layers, dec_layers):
        super(AutoEncoder, self).__init__()
        self.encoder = create_network(enc_layers)
        self.decoder = create_network(dec_layers)

        self.L = dec_layers[0][0]
        self.code_shape = enc_layers[-1][1]
        self.encoder = self._make_linear(enc_layers)
        self.decoder = self._make_linear(dec_layers)

    def _make_linear(self, layers: List[Tuple]):
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

    def encode(self, x):
        return self.encoder(x)

    def encode(self, X):
        return self.encoder(X)

    def decode(self, X):
        return self.decoder(X)
