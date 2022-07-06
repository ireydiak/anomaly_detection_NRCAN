import torch
import numpy as np
from pytorch_lightning.utilities.cli import MODEL_REGISTRY
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.optim.lr_scheduler import StepLR
from pyad.lightning.base import BaseLightningModel
from pyad.loss.EntropyLoss import EntropyLoss
from pyad.lightning.base import create_net_layers
from pyad.model.memory_module import MemoryUnit
from torch import nn
from typing import List, Any
from ray import tune as ray_tune


@MODEL_REGISTRY
class LitMemAE(BaseLightningModel):
    def __init__(
            self,
            mem_dim: int,
            latent_dim: int,
            enc_hidden_dims: List[int],
            shrink_thresh: float,
            alpha: float,
            activation="relu",
            **kwargs
    ):
        super(LitMemAE, self).__init__(**kwargs)
        # encoder-decoder network
        self.encoder = nn.Sequential(*create_net_layers(
            in_dim=self.in_features,
            out_dim=self.hparams.latent_dim,
            hidden_dims=self.hparams.enc_hidden_dims,
            activation=self.hparams.activation
        ))
        # xavier_init(self.encoder)
        self.decoder = nn.Sequential(*create_net_layers(
            in_dim=self.hparams.latent_dim,
            out_dim=self.in_features,
            hidden_dims=list(reversed(self.hparams.enc_hidden_dims)),
            activation=self.hparams.activation
        ))
        # xavier_init(self.decoder)
        # memory module
        self.mem_rep = MemoryUnit(
            self.hparams.mem_dim,
            self.hparams.latent_dim,
            self.hparams.shrink_thresh,
        )
        # loss modules
        self.recon_loss_fn = nn.MSELoss()
        self.entropy_loss_fn = EntropyLoss()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("litmemae")
        parser.add_argument(
            "--shrink_thresh", type=float, default=0.0025, help="Shrink threshold for hard shrinking relu"
        )
        parser.add_argument(
            "--latent_dim", type=int, default=1, help="Latent dimension of the encoder network"
        )
        parser.add_argument(
            "--enc_hidden_dims", type=List[int], help="hidden dimensions of the encoder-decoder architecture"
        )
        parser.add_argument(
            "--alpha", type=float, default=2e-4, help="coefficient for the entropy loss"
        )
        parser.add_argument(
            "--activation", type=str, default="relu", help="activation function"
        )
        return parser

    def compute_loss(self, X: torch.Tensor, mode: str = "train"):
        X_hat, W_hat = self.forward(X)
        R = self.recon_loss_fn(X, X_hat)
        E = self.entropy_loss_fn(W_hat)
        loss = R + (self.hparams.alpha * E)
        self.log(mode + "_loss", loss)
        return loss

    def forward(self, X: torch.Tensor) -> Any:
        f_e = self.encoder(X)
        f_mem, att = self.mem_rep(f_e)
        f_d = self.decoder(f_mem)
        return f_d, att

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        X, _, _ = batch
        X = X.float()
        return self.compute_loss(X)

    def score(self, X: torch.Tensor, y: torch.Tensor = None):
        X_hat, _ = self.forward(X)
        return torch.sum((X - X_hat) ** 2, dim=1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        scheduler = StepLR(optimizer, step_size=20, gamma=0.9)
        return [optimizer], [scheduler]


@MODEL_REGISTRY
class LitAutoEncoder(BaseLightningModel):
    def __init__(self,
                 hidden_dims: List[int],
                 latent_dim: int,
                 reg=0.5,
                 activation="relu",
                 **kwargs):
        super(LitAutoEncoder, self).__init__(**kwargs)
        self.encoder = nn.Sequential(
            *create_net_layers(
                in_dim=self.in_features,
                out_dim=self.hparams.latent_dim,
                hidden_dims=self.hparams.hidden_dims,
                activation=self.hparams.activation
            )
        )
        self.decoder = nn.Sequential(
            *create_net_layers(
                in_dim=self.hparams.latent_dim,
                out_dim=self.in_features,
                hidden_dims=list(reversed(self.hparams.hidden_dims)),
                activation=self.hparams.activation
            )
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("AutoEncoder")
        parser.add_argument("--hidden_dims", type=List[int])
        parser.add_argument("--latent_dim", type=int, default=1)
        parser.add_argument("--reg", type=float, default=0.5)
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--weight_decay", type=float, default=1e-4)

        return parent_parser

    @staticmethod
    def get_ray_config(in_features: int, n_instances: int) -> dict:
        # read parent config
        parent_cfg = BaseLightningModel.get_ray_config(in_features, n_instances)
        # used to set the maximum number of layers where every consecutive layer is compressing the previous layer by
        # a factor of 2
        depth = int(np.floor(np.log2(in_features)))
        # latent_dim options
        latent_dim_opts = (2 ** np.arange(0, depth + 1)).tolist()
        # construct the different layer options
        hidden_dims_opts = [
            [in_features // 2]
        ]
        for layer in range(1, depth):
            last_feature_dim = hidden_dims_opts[-1][-1]
            hidden_dims_opts.append(
                hidden_dims_opts[-1] + [max(1, last_feature_dim // 2)]
            )
        # options for this class
        child_cfg = {
            "hidden_dims": ray_tune.choice(hidden_dims_opts),
            "latent_dim": ray_tune.choice(latent_dim_opts),
            "reg": ray_tune.choice([0.1, 0.25, 0.5]),
            "activation": "relu"
        }
        return dict(
            **parent_cfg,
            **child_cfg
        )

    def forward(self, X: torch.Tensor, **kwargs) -> Any:
        X, y_true, full_labels = X
        X = X.float()
        scores = self.score(X)
        return scores, y_true, full_labels

    def score(self, X: torch.Tensor, y: torch.Tensor = None):
        emb = self.encoder(X)
        X_hat = self.decoder(emb)
        return ((X - X_hat) ** 2).sum(axis=-1)

    def compute_loss(self, X: torch.Tensor, X_hat: torch.Tensor, emb: torch.Tensor):
        l2_emb = emb.norm(2, dim=1).mean()
        loss = ((X - X_hat) ** 2).sum(axis=-1).mean() + self.hparams.reg * l2_emb
        return loss

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        X, _, _ = batch
        X = X.float()
        emb = self.encoder(X)
        X_hat = self.decoder(emb)
        loss = self.compute_loss(X, X_hat, emb)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr
        )
        scheduler = StepLR(optimizer, step_size=20, gamma=0.9)
        return [optimizer], [scheduler]
