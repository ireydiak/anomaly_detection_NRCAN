import torch.nn as nn
from .reconstruction import AutoEncoder as AE
from .base import BaseModel


class DUAD(BaseModel):
    def __init__(self, r=10, p0=.35, p=.30, **kwargs):
        self.p0 = p0
        self.p = p
        self.r = r
        self.latent_dim = kwargs.get('ae_latent_dim', 1)
        self.name = "DUAD"
        self.ae = None
        super(DUAD, self).__init__(**kwargs)
        self.cosim = nn.CosineSimilarity()

    def resolve_params(self, dataset_name: str):
        enc_layers = [
            (self.in_features, 60, nn.Tanh()),
            (60, 30, nn.Tanh()),
            (30, 10, nn.Tanh()),
            (10, self.latent_dim, None)
        ]
        dec_layers = [
            (self.latent_dim, 10, nn.Tanh()),
            (10, 30, nn.Tanh()),
            (30, 60, nn.Tanh()),
            (60, self.in_features, None)
        ]
        self.ae = AE(enc_layers, dec_layers).to(self.device)

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
        return {
            "duad_p": self.p,
            "duad_p0": self.p0,
            "duad_r": self.r
        }
