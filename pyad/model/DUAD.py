import torch.nn as nn
from pyad.model.reconstruction import AutoEncoder as AE
from pyad.model.base import BaseModel
from pyad.model.utils import activation_mapper


class DUAD(BaseModel):
    name = "DUAD"

    def __init__(
            self,
            r: int,
            p0: float,
            p: float,
            n_clusters: int,
            act_fn: str,
            n_layers: int,
            compression_factor: int,
            latent_dim: int,
            **kwargs
    ):
        super(DUAD, self).__init__(**kwargs)
        self.p0 = p0
        self.p = p
        self.r = r
        self.n_clusters = n_clusters
        self.latent_dim = latent_dim
        self.n_layers = n_layers
        self.compression_factor = compression_factor
        self.latent_dim = latent_dim
        self.ae = None
        self.cosim = nn.CosineSimilarity()
        self.act_fn = act_fn
        self._build_network()

    @staticmethod
    def get_args_desc():
        # TODO: better description
        return [
            ("p0", float, 35., "Variance threshold of re-evaluation selection"),
            ("p", float, 30., "p parameter"),
            ("r", int, 10, "r parameter"),
            ("p_s", float, 35., "Variance threshold of initial selection"),
            ("n_clusters", int, 20, "number of clusters"),
            ("act_fn", str, "tanh", "activation function of the AE network"),
            ("latent_dim", int, 10, "latent dimension of the AE network"),
            ("compression_factor", int, 2, "compression factor of the AE network"),
            ("n_layers", int, 4, "number of layers for the AE network")
        ]

    def _build_network(self):
        self.ae = AE(
            n_instances=self.n_instances,
            in_features=self.in_features,
            latent_dim=self.latent_dim,
            act_fn=self.act_fn,
            n_layers=self.n_layers,
            compression_factor=self.compression_factor
        ).to(self.device)

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
            "p": self.p,
            "p0": self.p0,
            "r": self.r,
            "act_fn": self.act_fn,
            "compression_factor": self.compression_factor,
            "latent_dim": self.latent_dim,
            "n_layers": self.n_layers,
            "n_clusters": self.n_clusters
        }
