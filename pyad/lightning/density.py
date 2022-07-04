import numpy as np
import torch
from pytorch_lightning.utilities.cli import MODEL_REGISTRY
from pyad.lightning.base import BaseLightningModel
from pyad.lightning.base import AutoEncoder, create_net_layers
from pyad.model.utils import relative_euclidean_dist
from torch import nn
from typing import List, Any


@MODEL_REGISTRY
class LitDAGMM(BaseLightningModel):

    def __init__(
            self,
            n_mixtures: int,
            ae_hidden_dims: List[int],
            gmm_hidden_dims: List[int],
            ae_activation: str = "relu",
            gmm_activation: str = "tanh",
            latent_dim: int = 1,
            lamb_1: float = 0.1,
            lamb_2: float = 0.005,
            reg_covar: float = 1e-12,
            dropout_rate: float = 0.5,
            **kwargs: Any
    ):
        super(LitDAGMM, self).__init__(**kwargs)
        self.phi = None
        self.mu = None
        self.cov_mat = None
        self.covs = None
        self.cosim = nn.CosineSimilarity().to(self.device)
        self.softmax = nn.Softmax().to(self.device)
        self._build_network()

    def _build_network(self):
        # GMM network
        gmm_layers = create_net_layers(
            in_dim=self.hparams.latent_dim + 2,
            out_dim=self.hparams.n_mixtures,
            activation=self.hparams.gmm_activation,
            hidden_dims=self.hparams.gmm_hidden_dims,
            dropout=self.hparams.dropout_rate
        )
        self.gmm_net = nn.Sequential(
            *gmm_layers
        )
        # AutoEncoder network
        self.ae_net = AutoEncoder(
            in_features=self.in_features,
            latent_dim=self.hparams.latent_dim,
            activation=self.hparams.ae_activation,
            hidden_dims=self.hparams.ae_hidden_dims
        )

    def forward(self, X: torch.Tensor):
        # computes the z vector of the original paper (p.4), that is
        # :math:`z = [z_c, z_r]` with
        #   - :math:`z_c = h(x; \theta_e)`
        #   - :math:`z_r = f(x, x')`
        emb, X_hat = self.ae_net(X)
        rel_euc_dist = relative_euclidean_dist(X, X_hat)
        cosim = self.cosim(X, X_hat)
        z = torch.cat(
            [emb, rel_euc_dist.unsqueeze(-1), cosim.unsqueeze(-1)],
            dim=1
        ).to(self.device)
        # compute gmm net output, that is
        #   - p = MLP(z, \theta_m) and
        #   - \gamma_hat = softmax(p)
        gamma_hat = self.gmm_net(z)
        gamma_hat = self.softmax(gamma_hat)
        return emb, X_hat, z, gamma_hat

    def training_step(self, batch, batch_idx):
        X, _, _ = batch
        X = X.float()
        # forward pass:
        # - embed and reconstruct original sample
        # - create new feature matrix from embeddings and reconstruction error
        # - pass later input to GMM-MLP
        z_c, x_prime, z, gamma_hat = self.forward(X)
        # estimate phi, mu and covariance matrices
        phi, mu, cov_mat = self.compute_params(z, gamma_hat)
        # estimate energy
        energy_result, pen_cov_mat = self.estimate_sample_energy(
            z, phi, mu, cov_mat
        )
        self.phi = phi.data
        self.mu = mu.data
        self.cov_mat = cov_mat
        loss = self.compute_loss(X, x_prime, energy_result, pen_cov_mat)
        return loss

    def compute_loss(
            self,
            X: torch.Tensor,
            X_hat: torch.Tensor,
            energy: torch.Tensor,
            pen_cov_mat: torch.Tensor
    ):
        rec_err = ((X - X_hat) ** 2).mean()
        loss = rec_err + self.hparams.lamb_1 * energy + self.hparams.lamb_2 * pen_cov_mat
        return loss

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

    def compute_params(self, z: torch.Tensor, gamma_hat: torch.Tensor):
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
        gamma_hat: N x K matrix (n_samples, n_mixtures)


        Returns
        -------

        """
        N = z.shape[0]

        # gamma
        gamma_sum = gamma_hat.sum(dim=0)
        #gamma_sum /= gamma_sum.sum()

        # \phi \in (n_mixtures,)
        phi = gamma_sum / N

        # \mu \in (n_mixtures, z_dim)
        mu = torch.sum(gamma_hat.unsqueeze(-1) * z.unsqueeze(1), dim=0) / gamma_sum.unsqueeze(-1)
        # mu = torch.linalg.inv(torch.diag(gamma_sum)) @ (gamma.T @ z)

        mu_z = z.unsqueeze(1) - mu.unsqueeze(0)
        cov_mat = mu_z.unsqueeze(-1) @ mu_z.unsqueeze(-2)
        cov_mat = gamma_hat.unsqueeze(-1).unsqueeze(-1) * cov_mat
        cov_mat = torch.sum(cov_mat, dim=0) / gamma_sum.unsqueeze(-1).unsqueeze(-1)

        return phi, mu, cov_mat

    def estimate_sample_energy(self, z, phi=None, mu=None, cov_mat=None, average_energy=True):
        if phi is None:
            phi = self.phi
        if mu is None:
            mu = self.mu
        if cov_mat is None:
            cov_mat = self.cov_mat

        # Avoid non-invertible covariance matrix by adding small values (self.reg_covar)
        d = z.shape[1]
        cov_mat = cov_mat + (torch.eye(d)).to(self.device) * self.hparams.reg_covar
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
        if exp_term.ndim == 1:
            exp_term = exp_term.unsqueeze(0)
        max_val = torch.max(exp_term.clamp(min=0), dim=1, keepdim=True)[0]
        exp_result = torch.exp(exp_term - max_val)

        log_term = phi * exp_result
        log_term /= det_cov_mat
        log_term = log_term.sum(axis=-1)

        # energy computation
        energy_result = - max_val.squeeze() - torch.log(log_term + self.hparams.reg_covar)

        if average_energy:
            energy_result = energy_result.mean()

        # penalty term
        cov_diag = torch.diagonal(cov_mat, dim1=1, dim2=2)
        pen_cov_mat = (1 / cov_diag).sum()

        return energy_result, pen_cov_mat

    def score(self, X: torch.Tensor, y: torch.Tensor = None):
        _, _, z, _ = self.forward(X)
        energy, _ = self.estimate_sample_energy(z, average_energy=False)
        return energy

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr
        )
        # scheduler = StepLR(optimizer, step_size=20, gamma=0.9)
        return optimizer  # [optimizer], [scheduler]
