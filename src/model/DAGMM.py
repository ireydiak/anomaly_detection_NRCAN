import torch.nn as nn
import torch
import numpy as np

from src.model import AutoEncoder as AE
from src.model import GMM


class DAGMM(nn.Module):
    """
    This class proposes an unofficial implementation of the DAGMM architecture proposed in
    https://sites.cs.ucsb.edu/~bzong/doc/iclr18-dagmm.pdf.
    Simply put, it's an end-to-end trained auto-encoder network complemented by a distinct gaussian mixture network.
    """

    def __init__(self, input_size, ae_layers=None, gmm_layers=None, lambda_1=0.1, lambda_2=0.005, device='cpu'):
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
        self.gmm = GMM(gmm_layers)

        self.cosim = nn.CosineSimilarity()
        self.softmax = nn.Softmax(dim=-1)
        self.phi = None
        self.mu = None
        self.cov_mat = None
        self.K = None
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.device = device

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
        x_prime, code = self.ae(x)
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
        K = gamma.shape[1]

        # K
        gamma_sum = torch.sum(gamma, dim=0)
        phi = gamma_sum / N

        # K x D
        # :math: `\mu = (I * gamma_sum)^{-1} * (\gamma^T * z)`
        mu = torch.sum(gamma.unsqueeze(-1) * z.unsqueeze(1), dim=0) / gamma_sum.unsqueeze(-1)
        # mu = torch.linalg.inv(torch.diag(gamma_sum)) @ (gamma.T @ z)

        # Covariance (K x D x D)
        covs = []
        for i in range(0, K):
            xm = z - mu[i]
            cov = 1 / gamma_sum[i] * ((gamma[:, i].unsqueeze(-1) * xm).T @ xm)
            cov += 1e-12
            covs.append(cov)

        mu_z = z.unsqueeze(1) - mu.unsqueeze(0)
        cov_mat = mu_z.unsqueeze(-1) @ mu_z.unsqueeze(-2)
        cov_mat = gamma.unsqueeze(-1).unsqueeze(-1) * cov_mat
        cov_mat = torch.sum(cov_mat, dim=0) / gamma_sum.unsqueeze(-1).unsqueeze(-1)

        self.phi = phi.data
        self.mu = mu.data
        self.cov_mat = cov_mat
        self.covs = covs
        # self.cov_mat = covs

        return phi, mu, cov_mat

    def mv_normal_cholesky(self, z: torch.Tensor, mu: torch.Tensor, Sigma: torch.Tensor):
        N, D = z.shape[0], z.shape[1]
        L = torch.linalg.cholesky(Sigma)
        scale = torch.prod(torch.diag(torch.linalg.cholesky(2 * np.pi * Sigma)))
        L_inv_zmu = torch.linalg.inv(L) @ (z - mu).reshape(N, D, 1)
        exp_term = L_inv_zmu.reshape(N, 1, D) @ L_inv_zmu
        return (1.0 / scale) * torch.exp(-0.5 * exp_term)

    def weighted_log_sum_exp(self, x: torch.Tensor, weights: torch.Tensor):
        a = torch.max(x)
        return a + torch.log(torch.sum(torch.exp(x - a) * weights))

    def estimate_sample_energy_js(self, z, phi, mu):
        energies = []
        for k, sigma_k in enumerate(self.covs):
            probs = self.mv_normal_cholesky(z, mu[k], sigma_k).reshape(1024, )
            energies.append(-self.weighted_log_sum_exp(torch.log(probs), phi[k]))
        return torch.sum(torch.Tensor(energies))

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
        eps = 1e-12
        cov_mat = cov_mat + (torch.eye(d)).to(device) * eps
        # N x K x D
        mu_z = z.unsqueeze(1) - mu.unsqueeze(0)

        # scaler
        inv_cov_mat = torch.linalg.inv(cov_mat)
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

    def estimate_sample_energy_backup(self, z, phi=None, mu=None, cov_mat=None, average_it=True, device='cpu'):

        if phi is None:
            phi = self.phi
        if mu is None:
            mu = self.mu
        if cov_mat is None:
            cov_mat = self.cov_mat

        # jc_res = self.estimate_sample_energy_js(z, phi, mu)

        # Avoid non-invertible covariance matrix by adding small values (eps)
        d = z.shape[1]
        eps = 1e-12
        cov_mat = cov_mat + (torch.eye(d)).to(device) * eps
        # N x K x D
        mu_z = z.unsqueeze(1) - mu.unsqueeze(0)

        # scaler
        inv_cov_mat = torch.linalg.inv(cov_mat)
        L = torch.linalg.cholesky(2 * np.pi * cov_mat)
        L_diag = torch.diagonal(L, dim1=1, dim2=2)
        # K
        det_cov_mat = torch.prod(L_diag, dim=1)

        # (NxKx1xD @ KxDxD) @ NxKxDx1 = NxKx1x1
        exp_term = (mu_z.unsqueeze(-2) @ inv_cov_mat) @ mu_z.unsqueeze(-1)
        # NxK
        exp_term = - 0.5 * exp_term.squeeze()

        # Applying log-sum-exp stability trick
        # https://www.xarg.org/2016/06/the-log-sum-exp-trick-in-machine-learning/
        # max_val = torch.max(exp_term.clamp(min=0), dim=1, keepdim=True)[0]
        max_val = torch.max(exp_term)
        # NxK
        exp_result = torch.exp(exp_term - max_val)

        # (1xK .* NxK) ./ det_cov_mat
        log_term = (phi * exp_result) / det_cov_mat
        # Kx1
        log_term = log_term.sum(axis=1)

        # element-wise log
        energy_result = - max_val - torch.log(log_term + eps)

        # if average_it:
        #     energy_result = energy_result.mean()

        # penalty term
        cov_diag = torch.diagonal(cov_mat, dim1=1, dim2=2)
        pen_cov_mat = (1 / cov_diag).sum()

        return energy_result.sum(), pen_cov_mat

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
            "L": self.ae.L
        }
