from typing import Union, Optional, Callable, Any, List
from pytorch_lightning.utilities.cli import MODEL_REGISTRY, DATAMODULE_REGISTRY

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR

from pyad.loss.TripletCenterLoss import TripletCenterLoss
from pyad.model.base import BaseModel
import pytorch_lightning as pl
from pytorch_lightning.core.optimizer import LightningOptimizer

from pyad.utils import metrics


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


class NeuTraLAD(BaseModel):
    name = "NeuTraLAD"

    def __init__(
            self,
            n_transforms: int,
            trans_type: str,
            temperature: float,
            trans_hidden_dims: list,
            enc_hidden_dims: list,
            **kwargs
    ):
        super(NeuTraLAD, self).__init__(**kwargs)
        self.n_transforms = n_transforms
        self.temperature = temperature
        self.trans_type = trans_type
        self.cosim = nn.CosineSimilarity()
        # Encoder and Transformation layers
        self.enc_hidden_dims = enc_hidden_dims
        self.trans_hidden_dims = trans_hidden_dims
        # Encoder
        self.enc = nn.Sequential(
            *create_network(self.in_features, self.enc_hidden_dims)
        ).to(self.device)
        # Transforms
        self.masks = self._create_masks()
        # self._build_network()

    @staticmethod
    def get_args_desc():
        return [
            ("temperature", float, 0.07, "temperature parameter"),
            ("trans_type", str, "mul", "transformation type (choose between 'res' or 'mul')"),
            ("n_transforms", int, 11, "number of transformations"),
            ("trans_hidden_dims", str, "200", "dimensions of the hidden layers"),
            ("enc_hidden_dims", list, "64,64,64,64,32", "dimensions of the encoder layers")
        ]

    def _create_masks(self):
        masks = []
        for k_i in range(self.n_transforms):
            layers = create_network(self.in_features, self.trans_hidden_dims, bias=False)
            layers.append(nn.Sigmoid())
            masks.append(
                nn.Sequential(*layers).to(self.device)
            )
        return masks

    @staticmethod
    def load_from_ckpt(ckpt):
        model = NeuTraLAD(
            in_features=ckpt["in_features"],
            n_instances=ckpt["n_instances"],
            n_transforms=ckpt["n_transforms"],
            temperature=ckpt["temperature"],
            trans_type=ckpt["trans_type"],
            enc_hidden_dims=ckpt["enc_hidden_dims"],
            trans_hidden_dims=ckpt["trans_hiddem_dims"]
        )
        return model

    def get_params(self) -> dict:
        params = dict(
            n_transforms=self.n_transforms,
            temperature=self.temperature,
            trans_type=self.trans_type,
            enc_hidden_dims=self.enc_hidden_dims,
            trans_hidden_dims=self.trans_hiddem_dims
        )
        return dict(
            **super(NeuTraLAD, self).get_params(),
            **params
        )

    def score(self, X: torch.Tensor):
        Xk = self._computeX_k(X)
        Xk = Xk.permute((1, 0, 2))
        Zk = self.enc(Xk)
        Zk = F.normalize(Zk, dim=-1)
        Z = self.enc(X)
        Z = F.normalize(Z, dim=-1)
        Hij = self._computeBatchH_ij(Zk)
        Hx_xk = self._computeBatchH_x_xk(Z, Zk)

        mask_not_k = (~torch.eye(self.n_transforms, dtype=torch.bool, device=self.device)).float()
        numerator = Hx_xk
        denominator = Hx_xk + (mask_not_k * Hij).sum(dim=2)
        scores_V = numerator / denominator
        score_V = (-torch.log(scores_V)).sum(dim=1)

        return score_V

    def _computeH_ij(self, Z):
        hij = F.cosine_similarity(Z.unsqueeze(1), Z.unsqueeze(0), dim=2)
        exp_hij = torch.exp(
            hij / self.temperature
        )
        return exp_hij

    def _computeBatchH_ij(self, Z):
        hij = F.cosine_similarity(Z.unsqueeze(2), Z.unsqueeze(1), dim=3)
        exp_hij = torch.exp(
            hij / self.temperature
        )
        return exp_hij

    def _computeH_x_xk(self, z, zk):

        hij = F.cosine_similarity(z.unsqueeze(0), zk)
        exp_hij = torch.exp(
            hij / self.temperature
        )
        return exp_hij

    def _computeBatchH_x_xk(self, z, zk):

        hij = F.cosine_similarity(z.unsqueeze(1), zk, dim=2)
        exp_hij = torch.exp(
            hij / self.temperature
        )
        return exp_hij

    def _computeX_k(self, X):
        X_t_s = []

        def transform(type):
            if type == 'res':
                return lambda mask, X: mask(X) + X
            else:
                return lambda mask, X: mask(X) * X

        t_function = transform(self.trans_type)
        for k in range(self.n_transforms):
            X_t_k = t_function(self.masks[k], X)
            X_t_s.append(X_t_k)
        X_t_s = torch.stack(X_t_s, dim=0)

        return X_t_s

    def forward(self, X: torch.Tensor):
        return self.score(X)


class GOAD(BaseModel):
    name = "GOAD"

    def __init__(
            self,
            n_transforms: int,
            feature_space: int,
            num_hidden_nodes: int,
            n_layers: int = 0,
            eps: float = 0,
            lamb: float = 0.1,
            margin: float = 1.,
            **kwargs
    ):
        super().__init__(**kwargs)
        assert n_layers >= 0, "n_layers must be greater or equal to 0"
        self.n_transforms = n_transforms
        self.margin = margin
        self.feature_space = feature_space
        self.num_hidden_nodes = num_hidden_nodes
        self.lamb = lamb
        self.eps = eps
        self.n_layers = n_layers
        self.build_network()

    def build_network(self):
        trunk_layers = [
            nn.Conv1d(self.feature_space, self.num_hidden_nodes, kernel_size=1, bias=False)
        ]
        for i in range(0, self.n_layers):
            trunk_layers.append(
                nn.Conv1d(self.num_hidden_nodes, self.num_hidden_nodes, kernel_size=1, bias=False),
            )
            if i < self.n_layers - 1:
                trunk_layers.append(
                    nn.LeakyReLU(0.2, inplace=True),
                )
            else:
                trunk_layers.append(
                    nn.Conv1d(self.num_hidden_nodes, self.num_hidden_nodes, kernel_size=1, bias=False),
                )
        self.trunk = nn.Sequential(
            *trunk_layers
        )
        self.head = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(self.num_hidden_nodes, self.n_transforms, kernel_size=1, bias=True),
        )

    def forward(self, X: torch.Tensor):
        # (batch_size, num_hidden_nodes, n_transforms)
        tc = self.trunk(X)
        # (batch_size, n_transforms, n_transforms)
        logits = self.head(tc)
        return tc, logits

    @staticmethod
    def get_args_desc():
        return [
            ("n_transforms", int, 256, "number of affine transformations"),
            ("margin", float, 1., "margin used in the objective function to regularize the distance between clusters"),
            ("num_hidden_nodes", int, 8, "number of hidden nodes in the neural network"),
            ("lamb", float, 0.1, "trade-off between triplet and cross-entropy losses"),
            ("n_layers", int, 0, "number of hidden layers"),
            ("feature_space", int, 32, "dimension of the feature space learned by the neural network"),
            ("eps", float, 0.,
             "small value added to the anomaly score to ensure equal probabilities for uncertain regions")
        ]

    def get_params(self) -> dict:
        parent_params = super(GOAD, self).get_params()
        return dict(
            n_transforms=self.n_transforms,
            margin=self.margin,
            feature_space=self.feature_space,
            lamb=self.lamb,
            n_layers=self.n_layers,
            **parent_params
        )


def h_func(x_k, x_l, temp=0.1):
    mat = F.cosine_similarity(x_k, x_l)

    return torch.exp(
        mat / temp
    )
