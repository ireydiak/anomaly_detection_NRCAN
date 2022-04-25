import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .base import BaseModel
from .utils import weights_init_xavier


# learning_rate = 1e-5
# batch_size = 50
# latent_dim = 32
# init_kernel = tf.contrib.layers.xavier_initializer()

def create_network(D: int, out_dims: np.array, bias=True) -> list:
    net_layers = []
    previous_dim = D
    for dim in out_dims:
        net_layers.append(nn.Linear(previous_dim, dim, bias=bias))
        net_layers.append(nn.ReLU())
        previous_dim = dim
    return net_layers


class NeuTraLAD(BaseModel):
    def __init__(self, n_layers=3,
                 trans_type='res', temperature: float = 0.07,
                 **kwargs
                 ):

        self.n_layers = n_layers
        super(NeuTraLAD, self).__init__(**kwargs)

        self.temperature = temperature
        self.trans_type = trans_type
        self.cosim = nn.CosineSimilarity()
        self.name = "NeuTraLAD"

    def _create_masks(self) -> list:
        masks = [None] * self.K
        out_dims = self.trans_layers or np.array([self.in_features] * self.n_layers)
        for K_i in range(self.K):
            net_layers = create_network(self.in_features, out_dims, bias=False)
            net_layers[-1] = nn.Sigmoid()
            masks[K_i] = nn.Sequential(*net_layers).to(self.device)
        return masks

    def _build_network(self):
        # Encoder
        enc_layers = create_network(self.in_features, self.emb_out_dims)[:-1]  # remove ReLU from the last layer
        self.enc = nn.Sequential(*enc_layers).to(self.device)
        # Masks / Transformations
        self.masks = self._create_masks()

    def resolve_params(self, dataset: str):
        K, Z = 7, 32
        # out_dims = np.linspace(self.D, Z, self.n_layers, dtype=np.int32)
        out_dims = [90, 70, 50] + [Z]
        trans_layers = [24, 6]
        if dataset == 'Thyroid':
            Z = 24
            K = 11
            out_dims = [24] * 4 + [Z]
            trans_layers = [24, 6]
        elif dataset == 'Arrhythmia':
            K = 11
            out_dims = [64] * 4 + [Z]
            trans_layers = [200, self.in_features]
            # out_dims[:-1] *= 2
        else:
            self.trans_type = 'mul'
            K = 11
            out_dims = [64] * 4 + [Z]
            trans_layers = [200, self.in_features]
        self.K, self.Z, self.emb_out_dims, self.trans_layers = K, Z, out_dims, trans_layers
        self._build_network()

        return K, Z, out_dims, trans_layers

    def get_params(self) -> dict:
        return {
            'D': self.in_features,
            'K': self.K,
            'temperature': self.temperature
        }

    def score(self, X: torch.Tensor):
        Xk = self._computeX_k(X)
        Xk = Xk.permute((1, 0, 2))
        Zk = self.enc(Xk)
        Zk = F.normalize(Zk, dim=-1)
        Z = self.enc(X)
        Z = F.normalize(Z, dim=-1)
        Hij = self._computeBatchH_ij(Zk)
        Hx_xk = self._computeBatchH_x_xk(Z, Zk)

        mask_not_k = (~torch.eye(self.K, dtype=torch.bool, device=self.device)).float()
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
        for k in range(self.K):
            X_t_k = t_function(self.masks[k], X)
            X_t_s.append(X_t_k)
        X_t_s = torch.stack(X_t_s, dim=0)

        return X_t_s

    def forward(self, X: torch.Tensor):
        return self.score(X)


def h_func(x_k, x_l, temp=0.1):
    mat = F.cosine_similarity(x_k, x_l)

    return torch.exp(
        mat / 0.1
    )
