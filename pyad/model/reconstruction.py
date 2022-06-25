import math

import numpy as np
import torch
from minisom import MiniSom
from pytorch_lightning.utilities.cli import MODEL_REGISTRY
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.optim.lr_scheduler import StepLR

from pyad.loss.EntropyLoss import EntropyLoss
from pyad.model.base import BaseModel
from pyad.model.GMM import GMM
from pyad.model.memory_module import MemoryUnit
from pyad.model import utils
from pyad.model.utils import activation_mapper
from torch import nn
import pytorch_lightning as pl
from typing import List, Any

from pyad.utils import metrics


def xavier_init(model):
    for name, param in model.named_parameters():
        if name.endswith(".bias"):
            param.data.fill_(0)
        else:
            bound = math.sqrt(6) / math.sqrt(param.shape[0] + param.shape[1])
            param.data.uniform_(-bound, bound)


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


class AutoEncoder(BaseModel):
    """
    Implements a basic Deep Auto Encoder
    """
    name = "AE"

    def __init__(
            self,
            latent_dim: int,
            act_fn: str,
            n_layers: int,
            compression_factor: int,
            reg: float = 0.5,
            **kwargs):
        super(AutoEncoder, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.act_fn = activation_mapper[act_fn]
        self.n_layers = n_layers
        self.compression_factor = compression_factor
        self.encoder, self.decoder = None, None
        self.reg = reg
        self._build_network()

    @staticmethod
    def load_from_ckpt(ckpt):
        model = AutoEncoder(
            in_features=ckpt["in_features"],
            n_instances=ckpt["n_instances"],
            latent_dim=ckpt["latent_dim"],
            act_fn=str(ckpt["act_fn"]).lower().replace("()", ""),
            n_layers=ckpt["n_layers"],
            compression_factor=ckpt["compression_factor"],
            reg=ckpt["reg"],
        )
        model.load_state_dict(ckpt["model_state_dict"])
        return model

    def _build_network(self):
        # Create the ENCODER layers
        enc_layers = []
        in_features = self.in_features
        compression_factor = self.compression_factor
        assert compression_factor * self.n_layers < in_features, "invalid parameters, too many layers for the number of available attributes"
        for _ in range(self.n_layers - 1):
            out_features = in_features // compression_factor
            enc_layers.append(
                [in_features, out_features, self.act_fn]
            )
            in_features = out_features
        enc_layers.append(
            [in_features, self.latent_dim, None]
        )
        # Create DECODER layers by simply reversing the encoder
        dec_layers = [[b, a, c] for a, b, c in reversed(enc_layers)]
        # Add and remove activation function from the first and last layer
        dec_layers[0][-1] = self.act_fn
        dec_layers[-1][-1] = None
        # Create networks
        self.encoder = utils.create_network(enc_layers)
        self.decoder = utils.create_network(dec_layers)

    @staticmethod
    def get_args_desc():
        return [
            ("latent_dim", int, 3, "Latent dimension of the AE network"),
            ("n_layers", int, 4, "Number of layers for the AE network"),
            ("compression_factor", int, 2, "Compression factor for the AE network"),
            ("act_fn", str, "relu", "Activation function of the AE network"),
            ("reg", float, 0.5, "Regularization term during training")
        ]

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        """
        This function compute the output of the network in the forward pass
        :param x: input
        :return: output of the model
        """
        output = self.encoder(x)
        output = self.decoder(output)
        return x, output

    def get_params(self) -> dict:
        params = {
            "latent_dim": self.latent_dim,
            "act_fn": str(self.act_fn).lower().replace("()", ""),
            "n_layers": self.n_layers,
            "compression_factor": self.compression_factor,
            "reg": self.reg
        }

        return dict(
            super().get_params(),
            **params
        )


class DAGMM(BaseModel):
    """
    This class proposes an unofficial implementation of the DAGMM architecture proposed in
    https://sites.cs.ucsb.edu/~bzong/doc/iclr18-dagmm.pdf.
    Simply put, it's an end-to-end trained auto-encoder network complemented by a distinct gaussian mixture network.
    """
    name = "DAGMM"

    def __init__(
            self,
            n_mixtures: int,
            latent_dim: int,
            lambda_1: float,
            lambda_2: float,
            reg_covar: float,
            n_layers: int,
            compression_factor: int,
            ae_act_fn="relu",
            gmm_act_fn="tanh",
            **kwargs
    ):
        super(DAGMM, self).__init__(**kwargs)
        self.n_mixtures = n_mixtures
        self.latent_dim = latent_dim
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.reg_covar = reg_covar
        self.ae_n_layers = n_layers
        self.ae_compression_factor = compression_factor
        self.ae_act_fn = ae_act_fn
        self.gmm_act_fn = activation_mapper[gmm_act_fn]
        self.cosim = nn.CosineSimilarity()
        self.softmax = nn.Softmax(dim=-1)
        self.ae = None
        self.gmm = None
        self._build_network()

    @staticmethod
    def get_args_desc():
        return [
            ("n_mixtures", int, 4, "Number of mixtures for the GMM network."),
            ("latent_dim", int, 1, "Latent dimension of the AE network."),
            ("lambda_1", float, 0.1, "Coefficient for the energy loss."),
            ("lambda_2", float, 0.005, "Coefficient of the penalization for degenerate covariance matrices."),
            ("reg_covar", float, 1e-12,
             "Small epsilon value added to covariance matrix to ensure it remains invertible."),
            ("n_layers", int, 4, "Number of layers for the AE network."),
            ("compression_factor", int, 2, "Compression factor for the AE network."),
            ("ae_act_fn", str, "relu", "Activation function of the AE network."),
            ("gmm_act_fn", str, "tanh", "Activation function of the GMM network."),
        ]

    def _build_network(self):
        # Create GMM layers
        gmm_layers = [
            [self.latent_dim + 2, 10, self.gmm_act_fn],
            [None, None, nn.Dropout(0.5)],
            [10, self.n_mixtures, nn.Softmax(dim=-1)]
        ]
        # Create the sub-networks (AE and GMM)
        self.ae = AutoEncoder(
            latent_dim=self.latent_dim,
            act_fn=self.ae_act_fn,
            n_layers=self.ae_n_layers,
            compression_factor=self.ae_compression_factor,
            in_features=self.in_features,
            device=self.device,
            n_instances=self.n_instances
        )
        self.gmm = GMM(layers=gmm_layers)

    @staticmethod
    def load_from_ckpt(ckpt):
        model = DAGMM(
            in_features=ckpt["in_features"],
            n_instances=ckpt["n_instances"],
            n_mixtures=ckpt["n_mixtures"],
            latent_dim=ckpt["latent_dim"],
            lambda_1=ckpt["lambda_1"],
            lambda_2=ckpt["lambda_2"],
            reg_covar=ckpt["reg_covar"],
            n_layers=ckpt["n_layers"],
            compression_factor=ckpt["compression_factor"],
            ae_act_fn=ckpt["ae_act_fn"],
            gmm_act_fn=ckpt["gmm_act_fn"]
        )
        model.load_state_dict(ckpt["model_state_dict"])
        return model

    def forward(self, x: torch.Tensor):
        """
        This function compute the output of the network in the forward pass
        :param x: input
        :return: output of the model
        """

        # computes the z vector of the original paper (p.4), that is
        # :math:`z = [z_c, z_r]` with
        #   - :math:`z_c = h(x; \theta_e)`
        #   - :math:`z_r = f(x, x')`
        code = self.ae.encoder(x)
        x_prime = self.ae.decoder(code)
        rel_euc_dist = self.relative_euclidean_dist(x, x_prime)
        cosim = self.cosim(x, x_prime)
        z_r = torch.cat([code, rel_euc_dist.unsqueeze(-1), cosim.unsqueeze(-1)], dim=1)

        # compute gmm net output, that is
        #   - p = MLN(z, \theta_m) and
        #   - \gamma_hat = softmax(p)
        gamma_hat = self.gmm.forward(z_r)
        # gamma = self.softmax(output)

        return code, x_prime, cosim, z_r, gamma_hat

    def forward_end_dec(self, x: torch.Tensor):
        """
        This function compute the output of the network in the forward pass
        :param x: input
        :return: output of the model
        """

        # computes the z vector of the original paper (p.4), that is
        # :math:`z = [z_c, z_r]` with
        #   - :math:`z_c = h(x; \theta_e)`
        #   - :math:`z_r = f(x, x')`
        code = self.ae.encoder(x)
        x_prime = self.ae.decoder(code)
        rel_euc_dist = self.relative_euclidean_dist(x, x_prime)
        cosim = self.cosim(x, x_prime)
        z_r = torch.cat([code, rel_euc_dist.unsqueeze(-1), cosim.unsqueeze(-1)], dim=1)

        return code, x_prime, cosim, z_r

    def forward_estimation_net(self, z_r: torch.Tensor):
        """
        This function compute the output of the network in the forward pass
        :param z_r: input
        :return: output of the model
        """

        # compute gmm net output, that is
        #   - p = MLN(z, \theta_m) and
        #   - \gamma_hat = softmax(p)
        gamma_hat = self.gmm.forward(z_r)

        return gamma_hat

    def relative_euclidean_dist(self, x, x_prime):
        return (x - x_prime).norm(2, dim=1) / x.norm(2, dim=1)

    def compute_params(self, z: torch.Tensor, gamma: torch.Tensor):
        r"""
        Estimates the parameters of the GMM.
        Implements the following formulas (p.5):
            :math:`\hat{\phi_k} = \sum_{i=1}^N \frac{\hat{\gamma_{ik}}}{N}`
            :math:`\hat{\mu}_k = \frac{\sum{i=1}^N \hat{\gamma_{ik} z_i}}{\sum{i=1}^N \hat{\gamma_{ik}}}`
            :math:`\hat{\Sigma_k} = \frac{
                \sum{i=1}^N \hat{\gamma_{ik}} (z_i - \hat{\mu_k}) (z_i - \hat{\mu_k})^T}
                {\sum{i=1}^N \hat{\gamma_{ik}}
            }`

        The second formula was modified to use matrices instead:
            :math:`\hat{\mu}_k = (I * \Gamma)^{-1} (\gamma^T z)`

        Parameters
        ----------
        z: N x D matrix (n_samples, n_features)
        gamma: N x K matrix (n_samples, n_mixtures)


        Returns
        -------

        """
        N = z.shape[0]
        K = gamma.shape[1]

        # K
        gamma_sum = torch.sum(gamma, dim=0)
        phi = gamma_sum / N

        # phi = torch.mean(gamma, dim=0)

        # K x D
        # :math: `\mu = (I * gamma_sum)^{-1} * (\gamma^T * z)`
        mu = torch.sum(gamma.unsqueeze(-1) * z.unsqueeze(1), dim=0) / gamma_sum.unsqueeze(-1)
        # mu = torch.linalg.inv(torch.diag(gamma_sum)) @ (gamma.T @ z)

        mu_z = z.unsqueeze(1) - mu.unsqueeze(0)
        cov_mat = mu_z.unsqueeze(-1) @ mu_z.unsqueeze(-2)
        cov_mat = gamma.unsqueeze(-1).unsqueeze(-1) * cov_mat
        cov_mat = torch.sum(cov_mat, dim=0) / gamma_sum.unsqueeze(-1).unsqueeze(-1)

        # ==============
        K, N, D = gamma.shape[1], z.shape[0], z.shape[1]
        # (K,)
        gamma_sum = torch.sum(gamma, dim=0)
        # prob de x_i pour chaque cluster k
        phi_ = gamma_sum / N

        # K x D
        # :math: `\mu = (I * gamma_sum)^{-1} * (\gamma^T * z)`
        mu_ = torch.sum(gamma.unsqueeze(-1) * z.unsqueeze(1), dim=0) / gamma_sum.unsqueeze(-1)
        # Covariance (K x D x D)

        self.phi = phi.data
        self.mu = mu.data
        self.cov_mat = cov_mat
        # self.covs = covs
        # self.cov_mat = covs

        return phi, mu, cov_mat

    def estimate_sample_energy(self, z, phi=None, mu=None, cov_mat=None, average_energy=True):
        if phi is None:
            phi = self.phi
        if mu is None:
            mu = self.mu
        if cov_mat is None:
            cov_mat = self.cov_mat

        # jc_res = self.estimate_sample_energy_js(z, phi, mu)

        # Avoid non-invertible covariance matrix by adding small values (eps)
        d = z.shape[1]
        eps = self.reg_covar
        cov_mat = cov_mat + (torch.eye(d)).to(self.device) * eps
        # N x K x D
        mu_z = z.unsqueeze(1) - mu.unsqueeze(0)

        # scaler
        inv_cov_mat = torch.cholesky_inverse(torch.linalg.cholesky(cov_mat))
        # inv_cov_mat = torch.linalg.inv(cov_mat)
        det_cov_mat = torch.linalg.cholesky(2 * np.pi * cov_mat)
        det_cov_mat = torch.diagonal(det_cov_mat, dim1=1, dim2=2)
        det_cov_mat = torch.prod(det_cov_mat, dim=1)

        exp_term = torch.matmul(mu_z.unsqueeze(-2), inv_cov_mat)
        exp_term = torch.matmul(exp_term, mu_z.unsqueeze(-1))
        exp_term = - 0.5 * exp_term.squeeze()

        # Applying log-sum-exp stability trick
        # https://www.xarg.org/2016/06/the-log-sum-exp-trick-in-machine-learning/
        max_val = torch.max(exp_term.clamp(min=0), dim=1, keepdim=True)[0]
        exp_result = torch.exp(exp_term - max_val)

        log_term = phi * exp_result
        log_term /= det_cov_mat
        log_term = log_term.sum(axis=-1)

        # energy computation
        energy_result = - max_val.squeeze() - torch.log(log_term + eps)

        if average_energy:
            energy_result = energy_result.mean()

        # penalty term
        cov_diag = torch.diagonal(cov_mat, dim1=1, dim2=2)
        pen_cov_mat = (1 / cov_diag).sum()

        return energy_result, pen_cov_mat

    def compute_loss(self, x, x_prime, energy, pen_cov_mat):
        """

        """
        rec_err = ((x - x_prime) ** 2).mean()
        loss = rec_err + self.lambda_1 * energy + self.lambda_2 * pen_cov_mat

        return loss

    def get_params(self) -> dict:
        params = dict(
            n_mixtures=self.n_mixtures,
            latent_dim=self.latent_dim,
            lambda_1=self.lambda_1,
            lambda_2=self.lambda_2,
            reg_covar=self.reg_covar,
            n_layers=self.ae_n_layers,
            compression_factor=self.ae_compression_factor,
            ae_act_fn=str(self.ae_act_fn).lower().replace("()", ""),
            gmm_act_fn=str(self.gmm_act_fn).lower().replace("()", ""),
        )
        return dict(
            **super(DAGMM, self).get_params(),
            **params
        )


default_som_args = {
    "x": 32,
    "y": 32,
    "lr": 0.6,
    "neighborhood_function": "bubble",
    "n_epoch": 500,
    "n_som": 1
}


class SOMDAGMM(BaseModel):
    name = "SOMDAGMM"

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
        self.n_mixtures = n_mixtures
        self.latent_dim = latent_dim
        self.reg_covar = reg_covar
        self.n_layers = n_layers
        self.compression_factor = compression_factor
        self.ae_act_fn = ae_act_fn
        self.gmm_act_fn = gmm_act_fn
        self.n_som = n_soms
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.som_args = None
        self.dagmm = None
        self.soms = None
        self._build_network()

    @staticmethod
    def get_args_desc():
        return [
            ("n_soms", int, 1, "Number of SOM"),
            ("n_mixtures", int, 4, "Number of mixtures for the GMM network"),
            ("latent_dim", int, 1, "Latent dimension of the AE network"),
            ("lambda_1", float, 0.005, "Lambda 1 parameter used during optimization"),
            ("lambda_2", float, 0.1, "Lambda 2 parameter used during optimization"),
            ("reg_covar", float, 1e-12,
             "Small epsilon value added to covariance matrix to ensure it remains reversible."),
            ("n_layers", int, 4, "Number of layers for the AE network"),
            ("compression_factor", int, 2, "Compression factor for the AE network"),
            ("ae_act_fn", str, "relu", "Activation function of the AE network"),
            ("gmm_act_fn", str, "tanh", "Activation function of the GMM network"),
        ]

    def _build_network(self):
        # set these values according to the used dataset
        # Use 0.6 for KDD; 0.8 for IDS2018 with babel as neighborhood function as suggested in the paper.
        grid_length = int(np.sqrt(5 * np.sqrt(self.n_instances))) // 2
        grid_length = 32 if grid_length > 32 else grid_length
        self.som_args = {
            "x": grid_length,
            "y": grid_length,
            "lr": 0.6,
            "neighborhood_function": "bubble",
            "n_epoch": 8000,
            "n_som": self.n_som
        }
        self.soms = [MiniSom(
            self.som_args['x'], self.som_args['y'], self.in_features,
            neighborhood_function=self.som_args['neighborhood_function'],
            learning_rate=self.som_args['lr']
        )] * self.som_args.get('n_som', 1)
        # DAGMM
        self.dagmm = DAGMM(
            in_features=self.in_features,
            n_instances=self.n_instances,
            device=self.device,
            n_mixtures=self.n_mixtures,
            latent_dim=self.latent_dim,
            lambda_1=self.lambda_1,
            lambda_2=self.lambda_2,
            reg_covar=self.reg_covar,
            n_layers=self.n_layers,
            compression_factor=self.compression_factor,
            ae_act_fn=self.ae_act_fn,
            gmm_act_fn=self.gmm_act_fn,
        )
        # Replace DAGMM's GMM network
        gmm_input = self.n_som * 2 + self.dagmm.latent_dim + 2
        gmm_layers = [
            (gmm_input, 10, nn.Tanh()),
            (None, None, nn.Dropout(0.5)),
            (10, self.n_mixtures, nn.Softmax(dim=-1))
        ]
        self.dagmm.gmm = GMM(gmm_layers).to(self.device)

    def train_som(self, X: torch.Tensor):
        # SOM-generated low-dimensional representation
        for i in range(len(self.soms)):
            self.soms[i].train(X, self.som_args["n_epoch"])

    def forward(self, X):
        # DAGMM's latent feature, the reconstruction error and gamma
        code, X_prime, cosim, z_r = self.dagmm.forward_end_dec(X)
        # Concatenate SOM's features with DAGMM's
        z_r_s = []
        for i in range(len(self.soms)):
            z_s_i = [self.soms[i].winner(x) for x in X.cpu()]
            z_s_i = [[x, y] for x, y in z_s_i]
            z_s_i = torch.from_numpy(np.array(z_s_i)).to(z_r.device)  # / (default_som_args.get('x')+1)
            z_r_s.append(z_s_i)
        z_r_s.append(z_r)
        Z = torch.cat(z_r_s, dim=1)

        # estimation network
        gamma = self.dagmm.forward_estimation_net(Z)

        return code, X_prime, cosim, Z, gamma

    def compute_params(self, Z, gamma):
        return self.dagmm.compute_params(Z, gamma)

    def estimate_sample_energy(self, Z, phi, mu, Sigma, average_energy=True):
        return self.dagmm.estimate_sample_energy(Z, phi, mu, Sigma, average_energy=average_energy)

    def compute_loss(self, X, X_prime, energy, Sigma):
        rec_loss = ((X - X_prime) ** 2).mean()
        sample_energy = self.lambda_1 * energy
        penalty_term = self.lambda_2 * Sigma

        return rec_loss + sample_energy + penalty_term

    def get_params(self) -> dict:
        params = self.dagmm.get_params()
        parent_params = super(SOMDAGMM, self).get_params()
        for k, v in self.som_args.items():
            params[f'SOM-{k}'] = v
        return dict(
            **params,
            **parent_params
        )


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
        #xavier_init(self.encoder)
        self.decoder = nn.Sequential(*create_net_layers(
            in_dim=self.hparams.latent_dim,
            out_dim=in_features,
            hidden_dims=list(reversed(self.hparams.enc_hidden_dims)),
            activation=self.hparams.activation
        ))
        #xavier_init(self.decoder)
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


class MemAutoEncoder(BaseModel):
    name = "MemAE"

    def __init__(
            self,
            mem_dim: int,
            latent_dim: int,
            shrink_thres: float,
            n_layers: int,
            compression_factor: int,
            alpha: float,
            act_fn="relu",
            **kwargs
    ):
        """
        Implements model Memory AutoEncoder as described in the paper
        `Memorizing Normality to Detect Anomaly: Memory-augmented Deep Autoencoder (MemAE) for Unsupervised Anomaly Detection`.
        A few adjustments were made to train the model on tabular data instead of images.
        This version is not meant to be trained on image datasets.

        - Original github repo: https://github.com/donggong1/memae-anomaly-detection
        - Paper citation:
            @inproceedings{
            gong2019memorizing,
            title={Memorizing Normality to Detect Anomaly: Memory-augmented Deep Autoencoder for Unsupervised Anomaly Detection},
            author={Gong, Dong and Liu, Lingqiao and Le, Vuong and Saha, Budhaditya and Mansour, Moussa Reda and Venkatesh, Svetha and Hengel, Anton van den},
            booktitle={IEEE International Conference on Computer Vision (ICCV)},
            year={2019}
            }

        Parameters
        ----------
        dataset_name: Name of the dataset (used to set the parameters)
        in_features: Number of variables in the dataset
        """
        super(MemAutoEncoder, self).__init__(**kwargs)
        self.alpha = alpha
        self.latent_dim = latent_dim
        self.act_fn = activation_mapper[act_fn]
        self.shrink_thres = shrink_thres
        self.n_layers = n_layers
        self.compression_factor = compression_factor
        self.mem_dim = mem_dim
        self.encoder = None
        self.decoder = None
        self.mem_rep = None
        self._build_network()

    @staticmethod
    def load_from_ckpt(ckpt):
        model = MemAutoEncoder(
            in_features=ckpt["in_features"],
            n_instances=ckpt["n_instances"],
            mem_dim=ckpt["mem_dim"],
            latent_dim=ckpt["latent_dim"],
            shrink_thres=ckpt["shrink_thres"],
            n_layers=ckpt["n_layers"],
            compression_factor=ckpt["compression_factor"],
            alpha=ckpt["alpha"],
            act_fn=ckpt["act_fn"],
        )
        model.load_state_dict(ckpt["model_state_dict"])
        return model

    @staticmethod
    def get_args_desc():
        return [
            ("shrink_thres", float, 0.0025, "Shrink threshold for hard shrinking relu"),
            ("latent_dim", int, 1, "Latent dimension of the AE network"),
            ("mem_dim", int, 50, "Number of memory units"),
            ("n_layers", int, 4, "Number of layers for the AE network"),
            ("alpha", float, 2e-4, "Coefficient for the entropy loss"),
            ("compression_factor", int, 2, "Compression factor for the AE network"),
            ("act_fn", str, "relu", "Activation function of the AE network"),
        ]

    def _build_network(self):
        # Create the ENCODER layers
        enc_layers = []
        in_features = self.in_features
        compression_factor = self.compression_factor
        for _ in range(self.n_layers - 1):
            out_features = in_features // compression_factor
            enc_layers.append(
                [in_features, out_features, self.act_fn]
            )
            in_features = out_features
        enc_layers.append(
            [in_features, self.latent_dim, None]
        )
        # Create DECODER layers by simply reversing the encoder
        dec_layers = [[b, a, c] for a, b, c in reversed(enc_layers)]
        # Add and remove activation function from the first and last layer
        dec_layers[0][-1] = self.act_fn
        dec_layers[-1][-1] = None
        # Create networks
        self.encoder = utils.create_network(enc_layers)
        self.decoder = utils.create_network(dec_layers)
        self.mem_rep = MemoryUnit(self.mem_dim, self.latent_dim, self.shrink_thres).to(self.device)

    def forward(self, x):
        f_e = self.encoder(x)
        f_mem, att = self.mem_rep(f_e)
        f_d = self.decoder(f_mem)
        return f_d, att

    def get_params(self):
        params = {
            "latent_dim": self.latent_dim,
            "shrink_thres": self.shrink_thres,
            "compression_factor": self.compression_factor,
            "n_layers": self.n_layers,
            "mem_dim": self.mem_dim,
            "alpha": self.alpha,
            "act_fn": str(self.act_fn).lower().replace("()", "")
        }
        return dict(
            **super(MemAutoEncoder, self).get_params(),
            **params
        )
