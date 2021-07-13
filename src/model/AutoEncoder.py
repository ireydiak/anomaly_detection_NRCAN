from typing import List, Tuple

import torch.nn as nn
from src.model import create_network
import torch


class AutoEncoder(nn.Module):
    def __init__(self, enc_layers: List[Tuple], dec_layers: List[Tuple]):
        super(AutoEncoder, self).__init__()
        self.encoder = create_network(enc_layers)
        self.decoder = create_network(dec_layers)

    def forward(self, X: torch.Tensor):
        return self.decode(self.encode(X))

    def encode(self, X):
        return self.encoder(X)

    def decode(self, X):
        return self.decoder(X)
