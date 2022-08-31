import pytorch_lightning as pl
import torch
import numpy as np
from torch.utils.data import DataLoader
from pyad.utils import metrics
from ray import tune
from torch import nn
from typing import List, Tuple, Optional, Callable, Union
from pyad.lightning.utils import activation_map


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


def layer_options_helper(in_features: int, max_layers: int = 4) -> Tuple[List[List[int]], List[int]]:
    # used to set the maximum number of layers where every consecutive layer is compressing the previous layer by
    # a factor of 2
    depth = int(np.floor(np.log2(in_features)))
    if max_layers > 0:
        depth = min(depth, max_layers)
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
    return hidden_dims_opts, latent_dim_opts


class BaseLightningModel(pl.LightningModule):
    is_nn = True
    is_legacy = False

    def __init__(
            self,
            lr: float,
            weight_decay: float = 0.0,
            in_features: int = -1,
            n_instances: int = -1,
            batch_size: int = -1,
            threshold: float = None,
            normal_str_label: str = "0",
            **kwargs
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
        normal_str_label: str
            the normal label string representation (used to compute per-class accuracy during test)
        """
        super(BaseLightningModel, self).__init__()
        if threshold:
            assert 0. <= threshold <= 100., "`threshold` must be inclusively between 0 and 1"
            self.threshold = threshold
        else:
            self.threshold = None
        # number of features and instances of the dataset
        # useful to build neural networks
        self.in_features = in_features
        self.n_instances = n_instances

        # call this to save hyper-parameters to the checkpoint
        # will save children parameters as well
        self.save_hyperparameters(
            ignore=["in_features", "n_instances", "threshold", "normal_str_label"]
        )
        # Performance metrics placeholder
        self.results = None
        self.normal_str_label = normal_str_label

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
        # TODO: fix bottleneck with labels
        for output in outputs:
            scores = np.append(scores, output["scores"])
            y_true = np.append(y_true, output["y_true"])
            labels = np.append(labels, output["labels"])
        # compute binary classification results
        results, y_pred = metrics.estimate_optimal_threshold(scores, y_true)
        # results, y_pred = metrics.score_recall_precision_w_threshold(
        #     scores, y_true, threshold=self.threshold
        # )
        # evaluate multi-class if labels contain over two distinct values
        if len(np.unique(labels)) > 2:
            pcacc = metrics.per_class_accuracy(y_true, y_pred, labels, normal_label="Benign")
            results = dict(**results, **pcacc)
        # log results
        self.results = results
        self.log_dict(results)

    def on_test_end(self):
        return self.results

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        X, y, labels = batch
        X = X.float()
        return self.score(X, y).cpu().detach().numpy()

    def test_step(self, batch, batch_idx):
        X, y_true, labels = batch
        X = X.float()
        scores = self.score(X)
        assert torch.isnan(scores).any().item() is False, "found NaN values in the final scores, aborting evaluation"

        if type(labels) == torch.Tensor:
            labels = labels.cpu().detach().numpy()
        else:
            labels = np.array(labels)

        return {
            "scores": scores.cpu().detach().numpy(),
            "y_true": y_true.cpu().detach().numpy(),
            "labels": labels
        }

    def inspect_gradient_wrt_input(self, dataloader, all_labels):
        # TODO: complete and test
        self.model.eval()
        y_true, scores, labels = [], [], []
        y_grad_wrt_X, label_grad_wrt_X, = {0: [], 1: []}, {label: [] for label in all_labels}
        losses = []
        for row in dataloader:
            X, y, label = row
            X = X.float()
            # TODO: put in dataloader
            label = np.array(label)
            X = X.to(self.device).float()
            X.requires_grad = True
            self.optimizer.zero_grad()

            loss = self.train_iter(X)
            loss.backward()

            for y_c in [0, 1]:
                dsdx = X.grad[y == y_c].mean(dim=0).cpu().numpy()
                if len(X.grad[y == y_c]) > 0:
                    y_grad_wrt_X[y_c].append(dsdx)

            for y_c in all_labels:
                dsdx = X.grad[label == y_c].cpu().numpy()
                if len(X.grad[label == y_c]) > 0:
                    label_grad_wrt_X[y_c].append(dsdx)

            losses.append(loss.item())
            y_true.extend(y.cpu().tolist())
            labels.extend(list(label))
        self.model.train()
        y_grad_wrt_X[0], y_grad_wrt_X[1] = np.asarray(y_grad_wrt_X[0]), np.asarray(y_grad_wrt_X[1])
        for y_c in all_labels:
            label_grad_wrt_X[y_c] = np.concatenate(label_grad_wrt_X[y_c])

        return {
            "y_true": np.array(y_true),
            "labels": np.array(labels),
            "y_grad_wrt_X": y_grad_wrt_X,
            "label_grad_wrt_X": label_grad_wrt_X,
            "losses": np.asarray(losses),
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
