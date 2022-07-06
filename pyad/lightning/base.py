import pytorch_lightning as pl
import torch
import numpy as np
from torch.utils.data import DataLoader
from pyad.utils import metrics
from ray import tune
from torch import nn
from typing import List
from pyad.model.utils import activation_map


def create_net_layers(in_dim, out_dim, hidden_dims, activation="relu", bias=True, dropout=0.):
    layers = []
    assert 0. <= dropout <= 1., "`dropout` must be inclusively between 0 and 1"
    for i in range(len(hidden_dims)):
        layers.append(
            nn.Linear(in_dim, hidden_dims[i], bias=bias)
        )
        if dropout > 0.:
            layers.append(
                nn.Dropout(dropout)
            )
        layers.append(
            activation_map[activation]
        )
        in_dim = hidden_dims[i]
    layers.append(
        nn.Linear(hidden_dims[-1], out_dim, bias=bias)
    )
    return layers


class BaseLightningModel(pl.LightningModule):

    def __init__(
            self,
            weight_decay: float,
            lr: float,
            in_features: int = -1,
            n_instances: int = -1,
            batch_size: int = -1,
            threshold: float = None,
    ):
        """

        Parameters
        ----------
        weight_decay: float
            L2 penalty
        lr: float
            learning rate
        in_features: int
            number of features in the dataset
        n_instances: int
            number of instances/samples in the dataset
        threshold: float
            anomaly ratio in the dataset
        """
        super(BaseLightningModel, self).__init__()
        if threshold:
            assert 0. <= threshold <= 100., "`threshold` must be inclusively between 0 and 1"
        else:
            self.threshold = None
        self.in_features = in_features
        self.n_instances = n_instances
        self.threshold = threshold
        # call this to save hyper-parameters to the checkpoint
        # will save children parameters as well
        self.save_hyperparameters(
            ignore=["in_features", "n_instances", "threshold"]
        )

    def before_train(self, dataloader: DataLoader):
        """
        Optional hook to pretrain model or estimate parameters before training
        """
        pass

    @staticmethod
    def get_ray_config(in_features: int, n_instances: int) -> dict:
        batch_size_opts = [bs_opt for bs_opt in [32, 64, 128, 1024] if n_instances // bs_opt > 0]
        return {
            "lr": tune.loguniform(1e-2, 1e-4),
            "weight_decay": tune.choice([0, 1e-4, 1e-6]),
            "batch_size": tune.choice(batch_size_opts)
        }

    def score(self, X: torch.Tensor, y: torch.Tensor = None):
        raise NotImplementedError

    def test_epoch_end(self, outputs) -> None:
        scores, y_true, labels = np.array([]), np.array([]), np.array([])
        for output in outputs:
            scores = np.append(scores, output["scores"].cpu().detach().numpy())
            y_true = np.append(y_true, output["y_true"].cpu().detach().numpy())
            labels = np.append(labels, output["labels"].cpu().detach().numpy())
        results, _ = metrics.score_recall_precision_w_threshold(scores, y_true, threshold=self.threshold)
        self.results = results
        self.log_dict(results)

    def on_test_end(self):
        return self.results

    def test_step(self, batch, batch_idx):
        X, y_true, labels = batch
        X = X.float()
        scores = self.score(X)

        return {
            "scores": scores,
            "y_true": y_true,
            "labels": labels
        }


class SimpleMLP(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            hidden_dims: List[int],
            activation: str = "relu"
    ):
        super(SimpleMLP).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_dims = hidden_dims
        self.activation = activation
        self._build_network()

    def _build_network(self):
        self.net = nn.Sequential(
            *create_net_layers(
                in_dim=self.in_features,
                out_dim=self.out_features,
                hidden_dims=self.hidden_dims,
                activation=self.activation
            )
        )

    def forward(self, X: torch.Tensor):
        return self.net(X)


class AutoEncoder(nn.Module):
    def __init__(
            self,
            in_features: int,
            hidden_dims: List[int],
            latent_dim: int = 1,
            reg: float = 0.5,
            activation: str = "relu"
    ):
        super(AutoEncoder, self).__init__()
        self.in_features = in_features
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.reg = reg
        self.activation = activation
        self._build_network()

    def _build_network(self):
        self.encoder = nn.Sequential(
            *create_net_layers(
                in_dim=self.in_features,
                out_dim=self.latent_dim,
                hidden_dims=self.hidden_dims,
                activation=self.activation
            )
        )
        self.decoder = nn.Sequential(
            *create_net_layers(
                in_dim=self.latent_dim,
                out_dim=self.in_features,
                hidden_dims=list(reversed(self.hidden_dims)),
                activation=self.activation
            )
        )

    def forward(self, X: torch.Tensor):
        emb = self.encoder(X)
        X_hat = self.decoder(emb)
        return emb, X_hat
