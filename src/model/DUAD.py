import numpy as np
import torch
import torch.nn as nn
from minisom import MiniSom


from .AutoEncoder import AutoEncoder as AE
from .BaseModel import BaseModel


class DUAD(BaseModel):
    def __init__(self, input_size, r, p0=.35, p=.30, ae_layers=None, **kwargs):
        super(DUAD, self).__init__()

        if not ae_layers:
            enc_layers = [(input_size, 60, nn.Tanh()), (60, 30, nn.Tanh()), (30, 10, nn.Tanh()), (10, 1, None)]
            dec_layers = [(1, 10, nn.Tanh()), (10, 30, nn.Tanh()), (30, 60, nn.Tanh()), (60, input_size, None)]
        else:
            enc_layers = ae_layers[0]
            dec_layers = ae_layers[1]

        self.ae = AE(enc_layers, dec_layers)
        self.cosim = nn.CosineSimilarity()

        self.p0 = p0
        self.p = p
        self.r = r

    def encode(self, x):
        return self.ae.encoder(x)

    def decode(self, code):
        return self.ae.decoder(code)

    def forward(self, x):
        code = self.ae.encoder(x)
        x_prime = self.ae.decoder(code)
        h_x = self.cosim(x, x_prime)
        return code, x_prime, h_x

    def get_params(self) -> dict:
        params = dict(
            duad_p=self.p,
            duad_p0=self.p0,
            duad_r=self.r
        )
        return params
