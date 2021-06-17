import torch.nn as nn
import torch.functional as F


class AutoEncoder(nn.Module):
    """
    this class implements an deep auto encoder using
    fullyconnected neural net
    """

    def __init__(self, input_size, layers_unit=None, last_layer_activation=False, fa='relu'):
        super(AutoEncoder, self).__init__()

        if layers_unit is None:
            layers_unit = [60, 30, 20, 10, 5, 2]

        self.encoder = self._make_linear(input_size, layers_unit,
                                         last_layer_activation=last_layer_activation,
                                         fa=fa)
        code_shape = layers_unit[-1]
        layers_unit.reverse()
        self.decoder = self._make_linear(code_shape, layers_unit[1:] + [input_size],
                                         last_layer_activation=last_layer_activation,
                                         fa=fa)

    def _make_linear(self, input_size, layers, last_layer_activation=False, fa='relu'):
        """
        This function builds a linear model whose units and layers depend on
        the passed @layers argument
        :param layers: a list of unit per layer
        :return: a fully connected neural net (Sequentiel object)
        """
        activations = dict(relu=nn.ReLU(True), tanh=nn.Tanh())
        net_layers = [nn.Linear(input_size, layers[0]), activations[fa]]
        for i in range(1, len(layers)):
            net_layers.append(nn.Linear(layers[i - 1], layers[i]))

            if i != len(layers) - 1:
                net_layers.append(activations[fa])
            else:
                if last_layer_activation:
                    net_layers.append(activations[fa])
        return nn.Sequential(*net_layers)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        """
        This function compute the output of the network in the forward pass
        :param x: input
        :return: output of the model
        """
        output = self.encoder(x)
        output = self.decoder(output)
        return output
