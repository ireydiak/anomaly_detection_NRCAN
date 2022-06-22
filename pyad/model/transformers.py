import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pyad.loss.TripletCenterLoss import TripletCenterLoss
from pyad.model.base import BaseModel
import pytorch_lightning as pl

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
            # fc_1_out: int,
            # fc_last_out: int,
            # compression_unit: int,
            n_transforms: int,
            # n_layers: int,
            trans_type: str,
            temperature: float,
            # trans_fc_in: int,
            # trans_fc_out: int,
            trans_hidden_dims: list,
            enc_hidden_dims: list,
            **kwargs
    ):
        super(NeuTraLAD, self).__init__(**kwargs)
        # self.compression_unit = compression_unit
        # self.fc_1_out = fc_1_out
        # self.fc_last_out = fc_last_out
        # self.n_layers = n_layers
        self.n_transforms = n_transforms
        self.temperature = temperature
        self.trans_type = trans_type
        # self.trans_fc_in = trans_fc_in if trans_fc_in and trans_fc_in > 0 else self.in_features
        # self.trans_fc_out = trans_fc_out if trans_fc_out and trans_fc_out > 0 else self.in_features
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

    #
    @staticmethod
    def get_args_desc():
        return [
            ("fc_1_out", int, 90, "output dim of first hidden layer"),
            ("fc_last_out", int, 32, "output dim of the last layer"),
            ("compression_unit", int, 20, "used to set output dim of next layer (in_feature - compression_unit)"),
            ("temperature", float, 0.07, "temperature parameter"),
            ("trans_type", str, "mul", "transformation type (choose between 'res' or 'mul')"),
            ("n_layers", int, 4, "number of layers"),
            ("n_transforms", int, 11, "number of transformations"),
            ("trans_hidden_dims", list, [200], "dimensions of the hidden layers"),
            ("enc_hidden_dims", list, [64, 32], "dimensions of the encoder layers")
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

    def _create_masks_legacy(self) -> list:
        masks = [None] * self.n_transforms
        out_dims = self.trans_layers
        for K_i in range(self.n_transforms):
            net_layers = create_network(self.in_features, out_dims, bias=False)
            net_layers[-1] = nn.Sigmoid()
            masks[K_i] = nn.Sequential(*net_layers).to(self.device)
        return masks

    def _build_network(self):
        out_dims = [0] * self.n_layers
        out_features = self.fc_1_out
        for i in range(self.n_layers - 1):
            out_dims[i] = out_features
            out_features -= self.compression_unit
        out_dims[-1] = self.fc_last_out
        self.trans_layers = [self.trans_fc_in, self.trans_fc_out]

        # Encoder
        enc_layers = create_network(self.in_features, out_dims)[:-1]  # removes ReLU from the last layer
        self.enc = nn.Sequential(*enc_layers).to(self.device)
        # Masks / Transformations
        self.masks = self._create_masks_legacy()

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


class LitGOAD(pl.LightningModule):
    def __init__(self,
                 in_features: int,
                 n_transforms: int,
                 feature_dim: int,
                 num_hidden_nodes: int,
                 batch_size: int,
                 n_layers: int = 0,
                 lr=1e-4,
                 eps=0,
                 lamb=0.1,
                 margin=1):
        super(LitGOAD, self).__init__()
        self.save_hyperparameters(
            "n_transforms", "feature_dim", "num_hidden_nodes", "eps", "lamb", "margin", "n_layers"
        )
        self.lr = lr
        self.batch_size = batch_size
        self.in_features = in_features
        self.n_transforms = n_transforms
        self.margin = margin
        self.feature_dim = feature_dim
        self.num_hidden_nodes = num_hidden_nodes
        self.lamb = lamb
        self.eps = eps
        self.n_layers = n_layers
        self.build_network()
        # Cross entropy loss
        self.ce_loss = nn.CrossEntropyLoss()
        # Triplet loss
        self.tc_loss = TripletCenterLoss(margin=self.margin)
        # Transformation matrix
        self.trans_matrix = torch.randn(
            (self.n_transforms, self.in_features, feature_dim),
            device=self.device
        )
        # Hypersphere centers
        self.centers = torch.zeros((self.feature_dim, self.n_transforms), device=self.device)

    def build_network(self):
        trunk_layers = [
            nn.Conv1d(self.feature_dim, self.num_hidden_nodes, kernel_size=1, bias=False)
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

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr
        )
        return optimizer

    def forward(self, X: torch.Tensor):
        # (batch_size, num_hidden_nodes, n_transforms)
        tc = self.trunk(X)
        # (batch_size, n_transforms, n_transforms)
        logits = self.head(tc)
        return tc, logits

    def score(self, sample: torch.Tensor):
        diffs = ((sample.unsqueeze(2) - self.centers) ** 2).sum(-1)
        diffs_eps = self.eps * torch.ones_like(diffs)
        diffs = torch.max(diffs, diffs_eps)
        logp_sz = torch.nn.functional.log_softmax(-diffs, dim=2)
        score = -torch.diagonal(logp_sz, 0, 1, 2).sum(dim=1)
        return score

    def on_fit_start(self):
        # Transfer modules to appropriate device
        self.trans_matrix = self.trans_matrix.to(self.device)
        self.centers = self.centers.to(self.device)

    def test_step(self, batch, batch_idx):
        X, y_true, labels = batch
        X = X.float()
        # Apply affine transformations
        X_augmented = torch.vstack(
            [X @ t for t in self.trans_matrix]
        ).reshape(X.shape[0], self.feature_dim, self.n_transforms)
        # Forward pass & reshape
        zs, fs = self.forward(X_augmented)
        zs = zs.permute(0, 2, 1)
        # Compute anomaly score
        scores = self.score(zs)
        # val_probs_rots[idx] = -torch.diagonal(score, 0, 1, 2).cpu().data.numpy()
        return {
            "scores": scores,
            "y_true": y_true,
            "labels": labels
        }

    def training_step(self, batch, batch_idx):
        X, _, _ = batch
        X = X.float()

        # transformation labels
        labels = torch.arange(
            self.n_transforms
        ).unsqueeze(0).expand((len(X), self.n_transforms)).long().to(self.device)

        # Apply affine transformations
        X_augmented = torch.vstack(
            [X @ t for t in self.trans_matrix]
        ).reshape(X.shape[0], self.feature_dim, self.n_transforms).to(self.device)
        # Forward pass
        tc_zs, logits = self.forward(X_augmented)
        # Update enters estimates
        self.centers += tc_zs.mean(0)
        # Update batch count for computing centers means
        self.n_batch += 1
        # Compute losses
        ce_loss = self.ce_loss(logits, labels)
        tc_loss = self.tc_loss(tc_zs)
        loss = self.lamb * tc_loss + ce_loss

        return loss

    def on_train_epoch_start(self) -> None:
        self.centers = torch.zeros((self.num_hidden_nodes, self.n_transforms)).to(self.device)
        self.n_batch = 0

    def on_train_epoch_end(self) -> None:
        self.centers = (self.centers.mT / self.n_batch).unsqueeze(0).to(self.device)

    def test_epoch_end(self, outputs) -> None:
        scores, y_true, labels = np.array([]), np.array([]), np.array([])
        for output in outputs:
            scores = np.append(scores, output["scores"].cpu().detach().numpy())
            y_true = np.append(y_true, output["y_true"].cpu().detach().numpy())
            labels = np.append(labels, output["labels"].cpu().detach().numpy())
        results, _ = metrics.score_recall_precision_w_threshold(scores, y_true)
        for k, v in results.items():
            self.log(k, v)


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
