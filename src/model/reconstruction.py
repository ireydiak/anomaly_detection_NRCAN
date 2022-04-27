import gzip
import pickle

import numpy as np
import torch
from minisom import MiniSom
from .base import BaseModel
from .GMM import GMM
from .memory_module import MemoryUnit
from torch import nn
from typing import Tuple, List


class AutoEncoder(nn.Module):
    """
    Implements a basic Deep Auto Encoder
    """

    def __init__(self, enc_layers: list = None, dec_layers: list = None, **kwargs):
        latent_dim = kwargs.get("ae_latent_dim", 1)
        cond = (enc_layers and dec_layers) or (kwargs.get("dataset_name", None) and kwargs.get("in_features", None))
        assert cond, "please provide either the name of the dataset and the number of features or specify the encoder and decoder layers"
        super(AutoEncoder, self).__init__()
        if not enc_layers or not dec_layers:
            enc_layers, dec_layers = AutoEncoder.resolve_layers(kwargs.get("in_features"),
                                                                kwargs.get("dataset_name"),
                                                                latent_dim=latent_dim)
        self.latent_dim = dec_layers[0][0]
        self.in_features = enc_layers[-1][1]
        self.encoder = self._make_linear(enc_layers)
        self.decoder = self._make_linear(dec_layers)
        self.name = "AutoEncoder"

    @staticmethod
    def from_dataset(in_features: int, dataset_name: str):
        enc_layers, dec_layers = AutoEncoder.resolve_layers(in_features, dataset_name)
        return AutoEncoder(enc_layers, dec_layers)

    @staticmethod
    def resolve_layers(in_features: int, dataset_name: str, latent_dim=1):
        if dataset_name == "Arrhythmia":
            enc_layers = [
                (in_features, 10, nn.Tanh()),
                (10, latent_dim, None)
            ]
            dec_layers = [
                (latent_dim, 10, nn.Tanh()),
                (10, in_features, None)
            ]
        elif dataset_name == "Thyroid":
            enc_layers = [
                (in_features, 12, nn.Tanh()),
                (12, 4, nn.Tanh()),
                (4, latent_dim, None)
            ]
            dec_layers = [
                (latent_dim, 4, nn.Tanh()),
                (4, 12, nn.Tanh()),
                (12, in_features, None)
            ]
        elif "kdd" in dataset_name.lower():
            enc_layers = [
                (in_features, 60, nn.Tanh()),
                (60, 30, nn.Tanh()),
                (30, 10, nn.Tanh()),
                (10, latent_dim, None)
            ]
            dec_layers = [
                (latent_dim, 10, nn.Tanh()),
                (10, 30, nn.Tanh()),
                (30, 60, nn.Tanh()),
                (60, in_features, None)]
        else:
            enc_layers = [
                (in_features, in_features // 2, nn.ReLU()),
                (in_features // 2, in_features // 4, nn.ReLU()),
                (in_features // 4, in_features // 6, nn.ReLU()),
                (in_features // 6, latent_dim, None)
            ]
            dec_layers = [
                (latent_dim, in_features // 6, nn.ReLU()),
                (in_features // 6, in_features // 4, nn.ReLU()),
                (in_features // 4, in_features // 2, nn.ReLU()),
                (in_features // 2, in_features, None)]
        return enc_layers, dec_layers

    def _make_linear(self, layers: List[Tuple]):
        """
        This function builds a linear model whose units and layers depend on
        the passed @layers argument
        :param layers: a list of tuples indicating the layers architecture (in_neuron, out_neuron, activation_function)
        :return: a fully connected neural net (Sequentiel object)
        """
        net_layers = []
        for in_neuron, out_neuron, act_fn in layers:
            net_layers.append(nn.Linear(in_neuron, out_neuron))
            if act_fn:
                net_layers.append(act_fn)
        return nn.Sequential(*net_layers)

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
        return {
            "in_features": self.in_features,
            "latent_dim": self.latent_dim
        }

    def reset(self):
        self.apply(self.weight_reset)

    def weight_reset(self, m):
        reset_parameters = getattr(m, "reset_parameters", None)
        if callable(reset_parameters):
            m.reset_parameters()

    @staticmethod
    def load(filename):
        # Load model from file (.pklz)
        with gzip.open(filename, 'rb') as f:
            model = pickle.load(f)
        assert isinstance(model, BaseModel)
        return model

    def save(self, filename):
        torch.save(self.state_dict(), filename)


class DAGMM(BaseModel):
    """
    This class proposes an unofficial implementation of the DAGMM architecture proposed in
    https://sites.cs.ucsb.edu/~bzong/doc/iclr18-dagmm.pdf.
    Simply put, it's an end-to-end trained auto-encoder network complemented by a distinct gaussian mixture network.
    """

    def __init__(self, lambda_1=0.005, lambda_2=0.1, reg_covar=1e-12, **kwargs):
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.reg_covar = reg_covar
        self.ae = None
        self.gmm = None
        self.K = None
        self.latent_dim = None
        self.name = "DAGMM"
        super(DAGMM, self).__init__(**kwargs)
        self.cosim = nn.CosineSimilarity()
        self.softmax = nn.Softmax(dim=-1)

    def resolve_params(self, dataset_name: str):
        # defaults to parameters described in section 4.3 of the paper
        # https://sites.cs.ucsb.edu/~bzong/doc/iclr18-dagmm.pdf.
        latent_dim = self.latent_dim or 1
        if dataset_name == 'Arrhythmia':
            K = 2
            gmm_layers = [
                (latent_dim + 2, 10, nn.Tanh()),
                (None, None, nn.Dropout(0.5)),
                (10, K, nn.Softmax(dim=-1))
            ]
        elif dataset_name == "Thyroid":
            K = 2
            gmm_layers = [
                (latent_dim + 2, 10, nn.Tanh()),
                (None, None, nn.Dropout(0.5)),
                (10, K, nn.Softmax(dim=-1))
            ]
        else:
            K = 4
            gmm_layers = [
                (latent_dim + 2, 10, nn.Tanh()),
                (None, None, nn.Dropout(0.5)),
                (10, K, nn.Softmax(dim=-1))
            ]
        self.latent_dim = latent_dim
        self.K = K
        self.ae = AutoEncoder.from_dataset(self.in_features, dataset_name)
        self.gmm = GMM(gmm_layers)

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
        inv_cov_mat = torch.cholesky_inverse(torch.cholesky(cov_mat))
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
        return {
            "lambda_1": self.lambda_1,
            "lambda_2": self.lambda_2,
            "latent_dim": self.ae.latent_dim,
            "K": self.gmm.K
        }


default_som_args = {
    "x": 32,
    "y": 32,
    "lr": 0.6,
    "neighborhood_function": "bubble",
    "n_epoch": 500,
    "n_som": 1
}


class SOMDAGMM(BaseModel):
    def __init__(self, n_som: int = 1, lambda_1: float = 0.1, lambda_2: float = 0.005, **kwargs):
        self.som_args = None
        self.dagmm = None
        self.soms = None
        self.n_som = n_som
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.name = "SOMDAGMM"
        super(SOMDAGMM, self).__init__(**kwargs)

    def resolve_params(self, dataset_name: str):
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
            dataset_name=dataset_name,
            in_features=self.in_features,
            n_instances=self.n_instances,
            device=self.device,
        )
        # Replace DAGMM's GMM network
        gmm_input = self.n_som * 2 + self.dagmm.latent_dim + 2
        if dataset_name == "Arrhythmia":
            gmm_layers = [(gmm_input, 10, nn.Tanh()), (None, None, nn.Dropout(0.5)), (10, 2, nn.Softmax(dim=-1))]
        else:
            gmm_layers = [(gmm_input, 10, nn.Tanh()), (None, None, nn.Dropout(0.5)), (10, 4, nn.Softmax(dim=-1))]
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
        for k, v in self.som_args.items():
            params[f'SOM-{k}'] = v
        return params


class MemAutoEncoder(BaseModel):

    def __init__(self, **kwargs):
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
        self.name = "MemAE"
        self.latent_dim = None
        self.encoder = None
        self.decoder = None
        self.mem_rep = None
        super(MemAutoEncoder, self).__init__(**kwargs)

    def resolve_params(self, dataset_name: str):
        mem_dim = 50
        shrink_thres = 0.0025
        if dataset_name == 'Arrhythmia':
            enc_layers = [
                nn.Linear(self.in_features, self.in_features // 2),
                nn.Tanh(),
                nn.Linear(self.in_features // 2, self.in_features // 4),
                nn.Tanh(),
                nn.Linear(self.in_features // 4, self.in_features // 6),
                nn.Tanh(),
                nn.Linear(self.in_features // 6, 10)
            ]
            dec_layers = [
                nn.Linear(10, self.in_features // 6),
                nn.Tanh(),
                nn.Linear(self.in_features // 6, self.in_features // 4),
                nn.Tanh(),
                nn.Linear(self.in_features // 4, self.in_features // 2),
                nn.Tanh(),
                nn.Linear(self.in_features // 2, self.in_features),
            ]
        elif dataset_name == 'Thyroid':
            enc_layers = [
                nn.Linear(self.in_features, 4),
                nn.Tanh(),
                nn.Linear(4, 2),
                nn.Tanh(),
                nn.Linear(2, 1)
            ]
            dec_layers = [
                nn.Linear(1, 2),
                nn.Tanh(),
                nn.Linear(2, 4),
                nn.Tanh(),
                nn.Linear(4, self.in_features)
            ]
        else:
            enc_layers = [
                nn.Linear(self.in_features, 60),
                nn.Tanh(),
                nn.Linear(60, 30),
                nn.Tanh(),
                nn.Linear(30, 10),
                nn.Tanh(),
                nn.Linear(10, 3)
            ]
            dec_layers = [
                nn.Linear(3, 10),
                nn.Tanh(),
                nn.Linear(10, 30),
                nn.Tanh(),
                nn.Linear(30, 60),
                nn.Tanh(),
                nn.Linear(60, self.in_features)
            ]
        self.latent_dim = enc_layers[-1].out_features
        self.encoder = nn.Sequential(
            *enc_layers
        ).to(self.device)
        self.decoder = nn.Sequential(
            *dec_layers
        ).to(self.device)
        self.mem_rep = MemoryUnit(mem_dim, self.latent_dim, shrink_thres, device=self.device).to(self.device)

    def forward(self, x):
        f_e = self.encoder(x)
        f_mem, att = self.mem_rep(f_e)
        f_d = self.decoder(f_mem)
        return f_d, att

    def get_params(self):
        return {
            "latent_dim": self.latent_dim,
            "in_features": self.in_features
        }
