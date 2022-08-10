import torch
import torch.nn as nn
from pytorch_lightning.utilities.cli import MODEL_REGISTRY
from pyad.lightning.base import BaseLightningModel


def create_network(
        input_dim: int, hidden_dims: list, bias=True, act_fn: nn.Module = nn.ReLU
) -> list:
    net_layers = []
    for i in range(len(hidden_dims) - 1):
        net_layers.append(
            nn.Linear(input_dim, hidden_dims[i], bias=bias)
        )
        net_layers.append(
            act_fn()
        )
        input_dim = hidden_dims[i]

    net_layers.append(
        nn.Linear(input_dim, hidden_dims[-1], bias=bias)
    )
    return net_layers


@MODEL_REGISTRY
class DUAD(BaseLightningModel):
    is_legacy = True

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

    def score(self, X: torch.Tensor, y: torch.Tensor = None):
        raise Exception("This is a placeholder model. Methods should not be called from here")


@MODEL_REGISTRY
class SOMDAGMM(BaseLightningModel):
    is_legacy = True

    def __init__(
            self,
            n_soms: int,
            n_mixtures: int,
            latent_dim: int,
            reg_covar: float,
            n_layers: int,
            compression_factor: int,
            lambda_1: float,
            lambda_2: float,
            ae_act_fn="relu",
            gmm_act_fn="tanh",
            **kwargs):
        super(SOMDAGMM, self).__init__(**kwargs)

    def score(self, X: torch.Tensor, y: torch.Tensor = None):
        raise Exception("This is a placeholder model. Methods should not be called from here")
