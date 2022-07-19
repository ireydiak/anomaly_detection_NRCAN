from typing import List
from pytorch_lightning.utilities.cli import MODEL_REGISTRY

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from pyad.loss.TripletCenterLoss import TripletCenterLoss
from pyad.lightning.base import BaseLightningModel
from ray import tune


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
class LitNeuTraLAD(BaseLightningModel):
    def __init__(
            self,
            n_transforms: int,
            trans_type: str,
            temperature: float,
            trans_hidden_dims: List[int],
            enc_hidden_dims: List[int],
            **kwargs
    ):
        super(LitNeuTraLAD, self).__init__(**kwargs)
        # Model parameters
        self.cosim = nn.CosineSimilarity()
        # Encoder and Transformation layers
        if self.hparams.trans_hidden_dims[-1] != self.in_features:
            self.hparams.trans_hidden_dims.append(self.in_features)
        # Encoder
        self.enc = nn.Sequential(
            *create_network(self.in_features, self.hparams.enc_hidden_dims, bias=False)
        )
        # Transforms
        self.masks = self._create_masks()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("ntl")
        parser.add_argument("--trans_hidden_dims", type=List[int])
        parser.add_argument("--enc_hidden_dims", type=List[int])
        parser.add_argument("--n_transforms", type=str, default=11, help="number of transformations")
        parser.add_argument("--temperature", type=float, default=0.1)
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--weight_decay", type=float, default=1e-4)

        return parent_parser

    @staticmethod
    def get_ray_config(in_features: int, n_instances: int):
        enc_dim = in_features // 2 if in_features > 10 else in_features * 3
        latent_dim = enc_dim // 2 if in_features > 10 else in_features
        enc_hidden_dims_choices = [
            [enc_dim, enc_dim, enc_dim, enc_dim, latent_dim],  # 5 layers
            [enc_dim, enc_dim, latent_dim],  # 3 layers
            [enc_dim, latent_dim]  # 2 layers
        ]
        trans_hidden_dims_choices = [
            [100],
            [200],
            [400]
        ]
        parent_choices = BaseLightningModel.get_ray_config(in_features, n_instances)
        child_choices = {
            "n_transforms": tune.choice([5, 11, 21, 63]),
            "trans_type": tune.choice(["res", "mul"]),
            "temperature": tune.loguniform(1e-1, 1e-3),
            "trans_hidden_dims": tune.choice(trans_hidden_dims_choices),
            "enc_hidden_dims": tune.choice(enc_hidden_dims_choices)
        }
        return dict(
            **parent_choices,
            **child_choices
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        scheduler = StepLR(optimizer, step_size=20, gamma=0.9)
        return [optimizer], [scheduler]

    def _create_masks(self):
        masks = nn.ModuleList()
        for k_i in range(self.hparams.n_transforms):
            layers = create_network(self.in_features, self.hparams.trans_hidden_dims, bias=False)
            layers.append(nn.Sigmoid())
            masks.append(
                nn.Sequential(*layers)
            )
        return masks

    def forward(self, X: torch.Tensor, **kwargs):
        X, y_true, full_labels = X
        X = X.float()
        scores = self.score(X)
        return scores, y_true, full_labels

    def compute_loss(self, X: torch.Tensor, mode="train"):
        loss = self.score(X).mean()
        self.log(mode + "_loss", loss)
        return loss

    def training_step(self, batch, batch_idx):
        X, _, _ = batch
        X = X.float()
        loss = self.compute_loss(X)

        return loss

    def score(self, X: torch.Tensor, y: torch.Tensor = None):
        Xk = self._computeX_k(X)
        Xk = Xk.permute((1, 0, 2))
        Zk = self.enc(Xk)
        Zk = F.normalize(Zk, dim=-1)
        Z = self.enc(X)
        Z = F.normalize(Z, dim=-1)
        Hij = self._computeBatchH_ij(Zk)
        Hx_xk = self._computeBatchH_x_xk(Z, Zk)

        mask_not_k = (~torch.eye(self.hparams.n_transforms, dtype=torch.bool, device=self.device)).float()
        numerator = Hx_xk
        denominator = Hx_xk + (mask_not_k * Hij).sum(dim=2)
        scores_V = numerator / denominator
        score_V = (-torch.log(scores_V)).sum(dim=1)

        return score_V

    def _computeH_ij(self, Z):
        hij = F.cosine_similarity(Z.unsqueeze(1), Z.unsqueeze(0), dim=2)
        exp_hij = torch.exp(
            hij / self.hparams.temperature
        )
        return exp_hij

    def _computeBatchH_ij(self, Z):
        hij = F.cosine_similarity(Z.unsqueeze(2), Z.unsqueeze(1), dim=3)
        exp_hij = torch.exp(
            hij / self.hparams.temperature
        )
        return exp_hij

    def _computeH_x_xk(self, z, zk):
        hij = F.cosine_similarity(z.unsqueeze(0), zk)
        exp_hij = torch.exp(
            hij / self.hparams.temperature
        )
        return exp_hij

    def _computeBatchH_x_xk(self, z, zk):
        hij = F.cosine_similarity(z.unsqueeze(1), zk, dim=2)
        exp_hij = torch.exp(
            hij / self.hparams.temperature
        )
        return exp_hij

    def _computeX_k(self, X):
        X_t_s = []

        def transform(trans_type, x):
            if trans_type == 'res':
                return lambda mask, X: mask(X) + X
            else:
                return lambda mask, X: mask(X) * X

        t_function = transform(self.hparams.trans_type, X)
        for k in range(self.hparams.n_transforms):
            X_t_k = t_function(self.masks[k], X)
            X_t_s.append(X_t_k)
        X_t_s = torch.stack(X_t_s, dim=0)

        return X_t_s


@MODEL_REGISTRY
class LitGOAD(BaseLightningModel):
    def __init__(self,
                 n_transforms: int,
                 feature_dim: int,
                 num_hidden_nodes: int,
                 batch_size: int,
                 n_layers: int = 0,
                 eps=0,
                 lamb=0.1,
                 margin=1,
                 **kwargs):
        super(LitGOAD, self).__init__(**kwargs)
        self.save_hyperparameters(
            "n_transforms", "feature_dim", "num_hidden_nodes", "eps", "lamb", "margin", "n_layers"
        )
        self.build_network()
        # Cross entropy loss
        self.ce_loss = nn.CrossEntropyLoss()
        # Triplet loss
        self.tc_loss = TripletCenterLoss(margin=self.hparams.margin)
        # Transformation matrix
        trans_matrix = torch.randn(
            (self.hparams.n_transforms, self.in_features, self.hparams.feature_dim),
        )
        self.register_buffer("trans_matrix", trans_matrix)
        # Hypersphere centers
        centers = torch.zeros((self.hparams.feature_dim, self.hparams.n_transforms))
        self.register_buffer("centers", centers)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("goad")
        parser.add_argument("--n_transforms", type=str, default=11, help="number of transformations")
        parser.add_argument("--feature_dim", type=int, help="output dimension of transformation")
        parser.add_argument("--n_layers", type=int, help="number of layers")
        parser.add_argument("--num_hidden_nodes", type=int, help="number of hidden nodes")
        parser.add_argument("--enc_hidden_dims", type=str, default="64,64,64,64,32")
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--weight_decay", type=float, default=1e-4)
        parser.add_argument("--eps", type=float, default=0.)
        parser.add_argument("--lamb", type=float, default=0.1)
        parser.add_argument("--margin", type=float, default=1.)

        return parent_parser

    @staticmethod
    def get_ray_config(in_features: int, n_instances: int):
        if n_instances < 1_000_000:
            feature_dim_choices = n_transforms_choices = [32, 64, 128, 256]
        else:
            feature_dim_choices = [32, 64]
            n_transforms_choices = [32]

        if n_instances < 5_000:
            n_layers_choices = [0, 1]
            num_hidden_nodes_choices = [8, 16, 32]
        else:
            n_layers_choices = [2, 3, 5]
            num_hidden_nodes_choices = [8, 32, 64, 128]
        parent_choices = BaseLightningModel.get_ray_config(in_features, n_instances)
        child_choices = {
            "n_transforms": tune.choice(n_transforms_choices),
            "feature_dim": tune.choice(feature_dim_choices),
            "num_hidden_nodes": tune.choice(num_hidden_nodes_choices),
            "eps": 0,
            "lamb": tune.loguniform(1e-1, 1.),
            "margin": 1,
            "n_layers": tune.choice(n_layers_choices)
        }
        return dict(
            **parent_choices,
            **child_choices
        )

    def build_network(self):
        trunk_layers = [
            nn.Conv1d(self.hparams.feature_dim, self.hparams.num_hidden_nodes, kernel_size=1, bias=False)
        ]
        for i in range(0, self.hparams.n_layers):
            trunk_layers.append(
                nn.Conv1d(self.hparams.num_hidden_nodes, self.hparams.num_hidden_nodes, kernel_size=1, bias=False),
            )
            if i < self.hparams.n_layers - 1:
                trunk_layers.append(
                    nn.LeakyReLU(0.2, inplace=True),
                )
            else:
                trunk_layers.append(
                    nn.Conv1d(self.hparams.num_hidden_nodes, self.hparams.num_hidden_nodes, kernel_size=1, bias=False),
                )
        self.trunk = nn.Sequential(
            *trunk_layers
        )
        self.head = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(self.hparams.num_hidden_nodes, self.hparams.n_transforms, kernel_size=1, bias=True),
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr
        )
        scheduler = StepLR(optimizer, step_size=20, gamma=0.9)

        return [optimizer], [scheduler]

    def forward(self, X: torch.Tensor):
        # (batch_size, num_hidden_nodes, n_transforms)
        tc = self.trunk(X)
        # (batch_size, n_transforms, n_transforms)
        logits = self.head(tc)
        return tc, logits

    def score(self, X: torch.Tensor, y: torch.Tensor = None):
        diffs = ((X.unsqueeze(2) - self.centers) ** 2).sum(-1)
        diffs_eps = self.hparams.eps * torch.ones_like(diffs)
        diffs = torch.max(diffs, diffs_eps)
        logp_sz = torch.nn.functional.log_softmax(-diffs, dim=2)
        score = -torch.diagonal(logp_sz, 0, 1, 2).sum(dim=1)
        return score

    def test_step(self, batch, batch_idx):
        X, y_true, labels = batch
        X = X.float()
        # Apply affine transformations
        X_augmented = torch.vstack(
            [X @ t for t in self.trans_matrix]
        ).reshape(X.shape[0], self.hparams.feature_dim, self.hparams.n_transforms)
        # Forward pass & reshape
        zs, fs = self.forward(X_augmented)
        zs = zs.permute(0, 2, 1)
        # Compute anomaly score
        scores = self.score(zs)
        # val_probs_rots[idx] = -torch.diagonal(score, 0, 1, 2).cpu().data.numpy()
        return {
            "scores": scores.cpu().detach().numpy(),
            "y_true": y_true.cpu().detach().numpy(),
            "labels": labels
        }

    def training_step(self, batch, batch_idx):
        X, _, _ = batch
        X = X.float()

        # transformation labels
        labels = torch.arange(
            self.hparams.n_transforms
        ).unsqueeze(0).expand((len(X), self.hparams.n_transforms)).long().to(self.device)

        # Apply affine transformations
        X_augmented = torch.vstack(
            [X @ t for t in self.trans_matrix]
        ).reshape(X.shape[0], self.hparams.feature_dim, self.hparams.n_transforms).to(self.device)
        # Forward pass
        tc_zs, logits = self.forward(X_augmented)
        # Update enters estimates
        self.centers += tc_zs.mean(0)
        # Update batch count for computing centers means
        self.n_batch += 1
        # Compute losses
        loss = self.compute_loss(logits, labels, tc_zs)

        return loss

    def compute_loss(self, logits, labels, tc_zs, mode="train"):
        ce_loss = self.ce_loss(logits, labels)
        tc_loss = self.tc_loss(tc_zs)
        loss = self.hparams.lamb * tc_loss + ce_loss
        self.log(mode + "_loss", loss)
        return loss

    def on_train_epoch_start(self) -> None:
        self.centers = torch.zeros((self.hparams.num_hidden_nodes, self.hparams.n_transforms)).to(self.device)
        self.n_batch = 0

    def on_train_epoch_end(self) -> None:
        self.centers = (self.centers.mT / self.n_batch).unsqueeze(0).to(self.device)
