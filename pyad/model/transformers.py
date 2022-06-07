import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pyad.model.base import BaseModel


def create_network(D: int, out_dims: np.array, bias=True) -> list:
    net_layers = []
    previous_dim = D
    for dim in out_dims:
        net_layers.append(nn.Linear(previous_dim, dim, bias=bias))
        net_layers.append(nn.ReLU())
        previous_dim = dim
    return net_layers


class NeuTraLAD(BaseModel):
    name = "NeuTraLAD"

    def __init__(
            self,
            fc_1_out: int,
            fc_last_out: int,
            compression_unit: int,
            n_transforms: int,
            n_layers: int,
            trans_type: str,
            temperature: float,
            trans_fc_in: int,
            trans_fc_out: int,
            **kwargs
    ):
        super(NeuTraLAD, self).__init__(**kwargs)
        self.compression_unit = compression_unit
        self.fc_1_out = fc_1_out
        self.fc_last_out = fc_last_out
        self.n_layers = n_layers
        self.n_transforms = n_transforms
        self.temperature = temperature
        self.trans_type = trans_type
        self.trans_fc_in = trans_fc_in if trans_fc_in and trans_fc_in > 0 else self.in_features
        self.trans_fc_out = trans_fc_out if trans_fc_out and trans_fc_out > 0 else self.in_features
        self.cosim = nn.CosineSimilarity()
        self._build_network()

    @staticmethod
    def get_args_desc():
        # TODO: better description
        return [
            ("fc_1_out", int, 90, "output dim of first hidden layer"),
            ("fc_last_out", int, 32, "output dim of the last layer"),
            ("compression_unit", int, 20, "used to set output dim of next layer (in_feature - compression_unit)"),
            ("temperature", float, 0.07, "temperature parameter"),
            ("trans_type", str, "mul", "transformation type (choose between 'res' or 'mul')"),
            ("n_layers", int, 4, "number of layers"),
            ("n_transforms", int, 11, "number of transformations"),
            ("trans_fc_in", int, 200, "input dim of transformer layer"),
            ("trans_fc_out", int, -1, "output dim of transformer layer"),
        ]

    def _create_masks(self) -> list:
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
        self.masks = self._create_masks()

    @staticmethod
    def load_from_ckpt(ckpt):
        model = NeuTraLAD(
            in_features=ckpt["in_features"],
            n_instances=ckpt["n_instances"],
            fc_1_out=ckpt["fc_1_out"],
            fc_last_out=ckpt["fc_last_out"],
            compression_unit=ckpt["compression_unit"],
            n_transforms=ckpt["n_transforms"],
            n_layers=ckpt["n_layers"],
            trans_type=ckpt["trans_type"],
            temperature=ckpt["temperature"],
            trans_fc_in=ckpt["trans_fc_in"],
            trans_fc_out=ckpt["trans_fc_out"]
        )
        return model

    def get_params(self) -> dict:
        params = dict(
            fc_1_out=self.fc_1_out,
            fc_last_out=self.fc_last_out,
            compression_unit=self.compression_unit,
            n_transforms=self.n_transforms,
            n_layers=self.n_layers,
            trans_type=self.trans_type,
            temperature=self.temperature,
            trans_fc_in=self.trans_fc_in,
            trans_fc_out=self.trans_fc_out,
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
        tc = self.trunk(X)
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
            ("eps", float, 0., "small value added to the anomaly score to ensure equal probabilities for uncertain regions")
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
