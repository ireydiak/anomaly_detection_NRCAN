import torch.nn as nn
from .utils import create_network


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

    def encode(self, X):
        return self.encoder(X)

    def decode(self, X):
        return self.decoder(X)

    def forward(self, X):
        z = self.encoder(X)
        x_prime = self.decoder(z)
        return x_prime, z
