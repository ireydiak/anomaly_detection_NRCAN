# Contains DAGMM, SOMDAGMM, MemAE


import numpy as np
import torch
import torch.nn as nn

from minisom import MiniSom
from .BaseModel import BaseModel
from .memory_module import MemoryUnit

from model import AutoEncoder as AE
from model import GMM


class DAGMM(BaseModel):
    """
    This class proposes an unofficial implementation of the DAGMM architecture proposed in
    https://sites.cs.ucsb.edu/~bzong/doc/iclr18-dagmm.pdf.
    Simply put, it's an end-to-end trained auto-encoder network complemented by a distinct gaussian mixture network.
    """

    def __init__(self, input_size, ae_layers=None, gmm_layers=None, lambda_1=0.1, lambda_2=0.005, device='cpu',
                 reg_covar=1e-12):
        """
        DAGMM constructor

        Parameters
        ----------
        input_size: Number of lines
        ae_layers: Layers for the Auto Encoder network
        gmm_layers: Layers for the GMM network
        lambda_1: Parameter lambda_1 for the objective function
        lambda_2: Parameter lambda_2 for the objective function
        """
        super(DAGMM, self).__init__()

        # defaults to parameters described in section 4.3 of the paper
        # https://sites.cs.ucsb.edu/~bzong/doc/iclr18-dagmm.pdf.
        if not ae_layers:
            enc_layers = [(input_size, 60, nn.Tanh()), (60, 30, nn.Tanh()), (30, 10, nn.Tanh()), (10, 1, None)]
            dec_layers = [(1, 10, nn.Tanh()), (10, 30, nn.Tanh()), (30, 60, nn.Tanh()), (60, input_size, None)]
        else:
            enc_layers = ae_layers[0]
            dec_layers = ae_layers[1]

        gmm_layers = gmm_layers or [(3, 10, nn.Tanh()), (None, None, nn.Dropout(0.5)), (10, 4, nn.Softmax(dim=-1))]

        self.ae = AE(enc_layers, dec_layers)
        self.gmm = GMM.GMM(gmm_layers)

        self.cosim = nn.CosineSimilarity()
        self.softmax = nn.Softmax(dim=-1)
        self.phi = None
        self.mu = None
        self.cov_mat = None
        self.K = None
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.device = device
        self.reg_covar = reg_covar

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

    def estimate_sample_energy(self, z, phi=None, mu=None, cov_mat=None, average_energy=True, device='cpu'):
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
        cov_mat = cov_mat + (torch.eye(d)).to(device) * eps
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
            "\u03BB_1": self.lambda_1,
            "\u03BB_2": self.lambda_2,
            "L": self.ae.L,
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
    def __init__(self, input_len: int, dagmm: DAGMM, som_args: dict = None, **kwargs):
        super(SOMDAGMM, self).__init__()
        self.som_args = som_args or default_som_args
        # Use 0.6 for KDD; 0.8 for IDS2018 with babel as neighborhood function as suggested in the paper.
        self.soms = [MiniSom(
            self.som_args['x'], self.som_args['y'], input_len,
            neighborhood_function=self.som_args['neighborhood_function'],
            learning_rate=self.som_args['lr']
        )] * self.som_args.get("n_som", 1)
        self.dagmm = dagmm
        self.lamb_1 = kwargs.get('lamb_1', 0.1)
        self.lamb_2 = kwargs.get('lamb_2', 0.005)

        # pretrain som as part of the initialization process

    def train_som(self, X):
        # SOM-generated low-dimensional representation
        for i in range(len(self.soms)):
            self.soms[i].train(X, self.som_args['n_epoch'])

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

    def estimate_sample_energy(self, Z, phi, mu, Sigma, average_energy=True, device='cpu'):
        return self.dagmm.estimate_sample_energy(Z, phi, mu, Sigma, average_energy=average_energy, device=device)

    def compute_loss(self, X, X_prime, energy, Sigma):
        rec_loss = ((X - X_prime) ** 2).mean()
        sample_energy = self.lamb_1 * energy
        penalty_term = self.lamb_2 * Sigma

        return rec_loss + sample_energy + penalty_term

    def get_params(self) -> dict:
        params = self.dagmm.get_params()
        for k, v in self.som_args.items():
            params[f'SOM-{k}'] = v
        return params


class MemAutoEncoder(BaseModel):

    def __init__(self, mem_dim: int, enc_layers: list, dec_layers: list, shrink_thres=0.0025, device='cpu'):
        """
        Implements model Memory AutoEncoder as described in the paper
        `Memorizing Normality to Detect Anomaly: Memory-augmented Deep Autoencoder (MemAE) for Unsupervised Anomaly Detection`.
        A few adjustments were made to train the model on matrices instead of tensors.
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
        D: Feature space dimension
        L: Latent space dimension
        mem_dim: Dimension of the memory matrix
        shrink_thres: The shrink threshold used in the memory module
        device: The Torch-compatible device used during training
        """
        super(MemAutoEncoder, self).__init__()

        self.D = enc_layers[0].in_features
        self.L = enc_layers[-1].out_features
        self.encoder = nn.Sequential(
            *enc_layers
        ).to(device)
        self.mem_rep = MemoryUnit(mem_dim, self.L, shrink_thres, device=device).to(device)
        self.decoder = nn.Sequential(
            *dec_layers
        ).to(device)

    def forward(self, x):
        f_e = self.encoder(x)
        f_mem, att = self.mem_rep(f_e)
        f_d = self.decoder(f_mem)
        return f_d, att

    def get_params(self):
        return {
            'L': self.L,
            'D': self.D
        }

