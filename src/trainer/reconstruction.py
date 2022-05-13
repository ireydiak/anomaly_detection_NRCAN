import torch
import numpy as np
from torch.utils.data.dataloader import DataLoader
from .base import BaseTrainer
from loss.EntropyLoss import EntropyLoss
from torch import nn


class AutoEncoderTrainer(BaseTrainer):

    def score(self, sample: torch.Tensor):
        _, X_prime = self.model(sample)
        return ((sample - X_prime) ** 2).sum(axis=1)

    def train_iter(self, X):
        code, X_prime = self.model(X)
        l2_z = code.norm(2, dim=1).mean()
        reg = 0.5
        loss = ((X - X_prime) ** 2).sum(axis=-1).mean() + reg * l2_z

        return loss


class DAGMMTrainer(BaseTrainer):
    def __init__(self, lamb_1: float = 0.1, lamb_2: float = 0.005, **kwargs) -> None:
        super(DAGMMTrainer, self).__init__(**kwargs)
        self.lamb_1 = lamb_1
        self.lamb_2 = lamb_2
        self.phi = None
        self.mu = None
        self.cov_mat = None
        self.covs = None

    def train_iter(self, sample: torch.Tensor):
        z_c, x_prime, _, z_r, gamma_hat = self.model(sample)
        phi, mu, cov_mat = self.compute_params(z_r, gamma_hat)
        energy_result, pen_cov_mat = self.estimate_sample_energy(
            z_r, phi, mu, cov_mat
        )
        self.phi = phi.data
        self.mu = mu.data
        self.cov_mat = cov_mat
        return self.loss(sample, x_prime, energy_result, pen_cov_mat)

    def loss(self, x, x_prime, energy, pen_cov_mat):
        rec_err = ((x - x_prime) ** 2).mean()
        return rec_err + self.lamb_1 * energy + self.lamb_2 * pen_cov_mat

    def test(self, dataset: DataLoader):
        """
        function that evaluate the model on the test set every iteration of the
        active learning process
        """
        self.model.eval()

        with torch.no_grad():
            scores, y_true = [], []
            for row in dataset:
                X, y = row[0], row[1]
                X = X.to(self.device).float()
                # forward pass
                code, x_prime, cosim, z, gamma = self.model(X)
                sample_energy, pen_cov_mat = self.estimate_sample_energy(
                    z, self.phi, self.mu, self.cov_mat, average_energy=False
                )
                y_true.extend(y)
                scores.extend(sample_energy.cpu().numpy())

        return np.array(y_true), np.array(scores)

    def weighted_log_sum_exp(self, x, weights, dim):
        """
        Inspired by https://discuss.pytorch.org/t/moving-to-numerically-stable-log-sum-exp-leads-to-extremely-large-loss-values/61938

        Parameters
        ----------
        x
        weights
        dim

        Returns
        -------

        """
        m, idx = torch.max(x, dim=dim, keepdim=True)
        return m.squeeze(dim) + torch.log(torch.sum(torch.exp(x - m) * (weights.unsqueeze(2)), dim=dim))

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

        return phi, mu, cov_mat

    def estimate_sample_energy(self, z, phi=None, mu=None, cov_mat=None, average_energy=True, eps=1e-12):
        if phi is None:
            phi = self.phi
        if mu is None:
            mu = self.mu
        if cov_mat is None:
            cov_mat = self.cov_mat

        # Avoid non-invertible covariance matrix by adding small values (eps)
        d = z.shape[1]
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
        if exp_term.ndim == 1:
            exp_term = exp_term.unsqueeze(0)
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

    def score(self, sample: torch.Tensor):
        _, _, _, z, _ = self.model(sample)
        return self.estimate_sample_energy(z)


class MemAETrainer(BaseTrainer):
    def __init__(self, **kwargs) -> None:
        super(MemAETrainer, self).__init__(**kwargs)
        self.alpha = 2e-4
        self.recon_loss_fn = nn.MSELoss().to(self.device)
        self.entropy_loss_fn = EntropyLoss().to(self.device)

    def train_iter(self, sample: torch.Tensor):
        x_hat, w_hat = self.model(sample)
        R = self.recon_loss_fn(sample, x_hat)
        E = self.entropy_loss_fn(w_hat)
        return R + (self.alpha * E)

    def score(self, sample: torch.Tensor):
        x_hat, _ = self.model(sample)
        return torch.sum((sample - x_hat) ** 2, axis=1)


class SOMDAGMMTrainer(BaseTrainer):

    def before_training(self, dataset: DataLoader):
        self.train_som(dataset.dataset.dataset.X)

    def train_som(self, X):
        self.model.train_som(X)

    def train_iter(self, X):
        # SOM-generated low-dimensional representation
        code, X_prime, cosim, Z, gamma = self.model(X)

        phi, mu, Sigma = self.model.compute_params(Z, gamma)
        energy, penalty_term = self.model.estimate_sample_energy(Z, phi, mu, Sigma)

        return self.model.compute_loss(X, X_prime, energy, penalty_term)

    def test(self, dataset: DataLoader):
        """
        function that evaluate the model on the test set every iteration of the
        active learning process
        """
        self.model.eval()

        with torch.no_grad():
            scores, y_true = [], []
            for row in dataset:
                X, y = row[0], row[1]
                X = X.to(self.device).float()

                sample_energy, _ = self.score(X)

                y_true.extend(y)
                scores.extend(sample_energy.cpu().numpy())

            return np.array(y_true), np.array(scores)

    def score(self, sample: torch.Tensor):
        code, x_prime, cosim, z, gamma = self.model(sample)
        phi, mu, cov_mat = self.model.compute_params(z, gamma)
        sample_energy, pen_cov_mat = self.model.estimate_sample_energy(
            z, phi, mu, cov_mat, average_energy=False
        )
        return sample_energy, pen_cov_mat
