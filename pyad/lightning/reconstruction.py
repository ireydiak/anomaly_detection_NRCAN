import numpy as np
import torch
from pytorch_lightning.utilities.cli import MODEL_REGISTRY
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.optim.lr_scheduler import StepLR

from pyad.loss.EntropyLoss import EntropyLoss
from pyad.model.memory_module import MemoryUnit
from pyad.model.utils import activation_mapper
from torch import nn
import pytorch_lightning as pl
from typing import List, Any
from pyad.utils import metrics


def create_net_layers(in_dim, out_dim, hidden_dims, activation="relu", bias=True):
    layers = []
    for i in range(len(hidden_dims)):
        layers.append(
            nn.Linear(in_dim, hidden_dims[i], bias=bias)
        )
        layers.append(
            activation_mapper[activation]
        )
        in_dim = hidden_dims[i]
    layers.append(
        nn.Linear(hidden_dims[-1], out_dim, bias=bias)
    )
    return layers


@MODEL_REGISTRY
class LitMemAE(pl.LightningModule):
    def __init__(
            self,
            in_features: int,
            mem_dim: int,
            latent_dim: int,
            enc_hidden_dims: List[int],
            shrink_thresh: float,
            alpha: float,
            activation="relu",
            lr: float = 1e-3,
            weight_decay: float = 0
    ):
        super(LitMemAE, self).__init__()
        self.save_hyperparameters(ignore=["in_features"])
        self.in_features = in_features
        # encoder-decoder network
        self.encoder = nn.Sequential(*create_net_layers(
            in_dim=in_features,
            out_dim=self.hparams.latent_dim,
            hidden_dims=self.hparams.enc_hidden_dims,
            activation=self.hparams.activation
        ))
        # xavier_init(self.encoder)
        self.decoder = nn.Sequential(*create_net_layers(
            in_dim=self.hparams.latent_dim,
            out_dim=in_features,
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

    def score(self, X: torch.Tensor):
        X_hat, _ = self.forward(X)
        return torch.sum((X - X_hat) ** 2, axis=1)

    def test_step(self, batch, batch_idx):
        X, y_true, labels = batch
        X = X.float()
        scores = self.score(X)
        return {
            "scores": scores,
            "y_true": y_true,
            "labels": labels
        }

    def test_epoch_end(self, outputs) -> None:
        scores, y_true, labels = np.array([]), np.array([]), np.array([])
        for output in outputs:
            scores = np.append(scores, output["scores"].cpu().detach().numpy())
            y_true = np.append(y_true, output["y_true"].cpu().detach().numpy())
            labels = np.append(labels, output["labels"].cpu().detach().numpy())
        # results, _ = metrics.score_recall_precision_w_threshold(scores, y_true)
        results = metrics.estimate_optimal_threshold(scores, y_true)
        for k, v in results.items():
            self.log(k, v)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        scheduler = StepLR(optimizer, step_size=20, gamma=0.9)
        return [optimizer], [scheduler]


@MODEL_REGISTRY
class LitAutoEncoder(pl.LightningModule):
    def __init__(self,
                 in_features: int,
                 hidden_dims: List[int],
                 latent_dim: int,
                 lr=1e-3,
                 reg=0.5,
                 weight_decay=1e-4,
                 activation="relu"):
        super(LitAutoEncoder, self).__init__()
        # call this to save hyper-parameters to the checkpoint
        self.save_hyperparameters(
            "hidden_dims", "latent_dim", "lr", "reg", "activation"
        )
        self.in_features = in_features
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.latent_dim = latent_dim
        self.weight_decay = weight_decay
        self.reg = reg
        self.lr = lr
        self.encoder = nn.Sequential(
            *create_net_layers(in_dim=in_features, out_dim=latent_dim, hidden_dims=hidden_dims, activation=activation)
        )
        self.decoder = nn.Sequential(
            *create_net_layers(
                in_dim=latent_dim, out_dim=in_features, hidden_dims=list(reversed(hidden_dims)), activation=activation
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

    def forward(self, X: torch.Tensor, **kwargs) -> Any:
        X, y_true, full_labels = X
        X = X.float()
        scores = self.score(X)
        return scores, y_true, full_labels

    def score(self, X: torch.Tensor):
        emb = self.encoder(X)
        X_hat = self.decoder(emb)
        return ((X - X_hat) ** 2).sum(axis=-1)

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        X, _, _ = batch
        X = X.float()
        emb = self.encoder(X)
        X_hat = self.decoder(emb)
        l2_emb = emb.norm(2, dim=1).mean()
        loss = ((X - X_hat) ** 2).sum(axis=-1).mean() + self.reg * l2_emb
        return loss

    def test_step(self, batch, batch_idx):
        X, y_true, labels = batch
        X = X.float()
        scores = self.score(X)

        return {
            "scores": scores,
            "y_true": y_true,
            "labels": labels
        }

    def test_epoch_end(self, outputs) -> None:
        scores, y_true, labels = np.array([]), np.array([]), np.array([])
        for output in outputs:
            scores = np.append(scores, output["scores"].cpu().detach().numpy())
            y_true = np.append(y_true, output["y_true"].cpu().detach().numpy())
            labels = np.append(labels, output["labels"].cpu().detach().numpy())
        results, _ = metrics.score_recall_precision_w_threshold(scores, y_true)
        for k, v in results.items():
            self.log(k, v)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr
        )
        scheduler = StepLR(optimizer, step_size=20, gamma=0.9)
        return [optimizer], [scheduler]
