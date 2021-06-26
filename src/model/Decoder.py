from typing import List, Tuple
import torch
import torch.nn as nn
from . import create_network


class Decoder(nn.Module):
    """
    Copies essentially the same structure as an Encoder but we want different names when building models.
    """
    def __init__(self, layers: List[Tuple]):
        super(Decoder, self).__init__()
        self.dec = create_network(layers)

    def forward(self, X: torch.Tensor):
        self.dec(X)
