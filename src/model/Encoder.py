from typing import List, Tuple
import torch
import torch.nn as nn
from . import create_network


class Encoder(nn.Module):
    def __init__(self, layers: List[Tuple]):
        super(Encoder, self).__init__()
        self.enc = create_network(layers)

    def forward(self, X: torch.Tensor):
        self.enc(X)
