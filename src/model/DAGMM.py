import torch.nn as nn
import torch.functional as F
from model.AutoEncoder import AutoEncoder as AE
import torch
import numpy as np


class DAGMM(nn.Module):
    """
    This class proposes an unofficial implementation of the DAGMM architecture proposed in
    https://sites.cs.ucsb.edu/~bzong/doc/iclr18-dagmm.pdf.
    Simply put, it's an end-to-end trained auto-encoder network complemented by a distinct gaussian mixture network.
    """

    def __init__(self, input_size,
                 ae_layers_unit=None,
                 last_layer_activation=False,
                 fa='relu',
                 gmm_layers=None,
                 gmm_drop_out_last=True,
                 lambda_1=0.1,
                 lambda_2=0.005):
        """
        @gmm_layer is a list of units for each layer. The number of units in the last layer
        defines the number of gaussian of the model
        """
        super(DAGMM, self).__init__()

        self.ae = self.create_ae(input_size)

        code_shape = self.ae.code_shape + 2  # 2 for the euclidean error and the cosine similarity

        self.cosim = nn.CosineSimilarity()
        self.softmax = nn.Softmax(dim=-1)

        gmm_layers = gmm_layers or [(3, 10, nn.Tanh()), (None, None, nn.Dropout(0.5)), (10, 4, nn.Softmax())]
        self.gmm_predictor = self.create_gmm(code_shape, gmm_layers, fa=fa, gmm_drop_out_last=gmm_drop_out_last)

        self.phi = None
        self.mu = None
        self.cov_mat = None
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2

    def create_ae(self, input_size: int):
        return AE(input_size)

    def create_gmm(self, input_size, layers, last_layer_activation=False, fa='relu', gmm_drop_out_last=False):
        """
        This function builds a linear model whose units and layers depend on
        the passed @layers argument
        :param layers: a list of unit per layer
        :return: a fully connected neural net (Sequentiel object)
        """
        net_layers = []
        for in_neuron, out_neuron, act_fn in layers:
            if in_neuron and out_neuron:
                net_layers.append(nn.Linear(in_neuron, out_neuron))
            net_layers.append(act_fn)
        # activations = dict(relu=nn.ReLU(True), tanh=nn.Tanh())
        # net_layers = [nn.Linear(input_size, layers[0]), activations[fa]]
        # for i in range(1, len(layers)):
        #     # drop out for the last layer
        #     if i == len(layers) - 1:
        #         net_layers.append(nn.Dropout())
        #
        #     net_layers.append(nn.Linear(layers[i - 1], layers[i]))
        #
        #     if i != len(layers) - 1:
        #         net_layers.append(activations[fa])
        #     else:
        #         if last_layer_activation:
        #             net_layers.append(activations[fa])
        return nn.Sequential(*net_layers)

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
        x_hat = self.ae.decoder(code)
        rel_euc_dist = self.relative_euclidean_dist(x, x_hat)
        cosim = self.cosim(x, x_hat)
        z_r = torch.cat([code, rel_euc_dist.unsqueeze(-1), cosim.unsqueeze(-1)], dim=1)

        # compute gmm net output, that is
        #   - p = MLN(z, \theta_m) and
        #   - \gamma_hat = softmax(p)
        gamma_hat = self.gmm_predictor(z_r)
        # gamma = self.softmax(output)

        return code, x_hat, cosim, z_r, gamma_hat

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

    def relative_euclidean_dist(self, x, x_hat):
        return (x - x_hat).norm(2, dim=1) / x.norm(2, dim=1)

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

        # K x D
        # :math: `\mu = (I * gamma_sum)^{-1} * (\gamma^T * z)`
        # mu = torch.sum(gamma.unsqueeze(-1) * z.unsqueeze(1), dim=0) / gamma_sum.unsqueeze(-1)
        mu = torch.matmul(
            torch.inverse(torch.diag(gamma_sum)),
            torch.matmul(gamma.T, z)
        )

        # Covariance (K x D x D)
        mu_z = z.unsqueeze(1) - mu.unsqueeze(0)
        cov_mat = torch.matmul(mu_z.unsqueeze(-1), mu_z.unsqueeze(-2))
        cov_mat = gamma.unsqueeze(-1).unsqueeze(-1) * cov_mat
        cov_mat = torch.sum(cov_mat, dim=0) / gamma_sum.unsqueeze(-1).unsqueeze(-1)

        self.phi = phi.data
        self.mu = mu.data
        self.cov_mat = cov_mat

        return phi, mu, cov_mat

    def estimate_sample_energy(self, z, phi=None, mu=None, cov_mat=None, average_it=True, device='cpu'):

        if phi is None:
            phi = self.phi
        if mu is None:
            mu = self.mu
        if cov_mat is None:
            cov_mat = self.cov_mat

        # Avoid non-invertible covariance matrix by adding small values (eps)
        d = z.shape[1]
        eps = 1e-12
        cov_mat = cov_mat + (torch.eye(d)).to(device) * eps
        mu_z = z.unsqueeze(1) - mu.unsqueeze(0)

        # or more simply
        #
        # scaler
        inv_cov_mat = torch.linalg.inv(cov_mat)
        det_cov_mat = torch.linalg.cholesky(2 * np.pi * cov_mat)
        det_cov_mat = torch.diagonal(det_cov_mat, dim1=1, dim2=2)
        det_cov_mat = torch.prod(det_cov_mat, dim=1)
        det_cov_mat = torch.sqrt(det_cov_mat)

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

        if average_it:
            energy_result = energy_result.mean()

        # penalty term
        cov_diag = torch.diagonal(cov_mat, dim1=1, dim2=2)
        pen_cov_mat = (1 / cov_diag).sum()

        return energy_result, pen_cov_mat

    def compute_loss(self, x, x_prime, energy, pen_cov_mat):
        """

        """
        # rec_err = ((x - x_prime) ** 2).mean()
        # rec_err = ((x - x_prime) ** 2).sum(1)
        N = x.shape[0]
        rec_err = torch.linalg.norm(x - x_prime, 2, dim=1).mean()
        loss = rec_err + (self.lambda_1 / N) * energy + self.lambda_2 * pen_cov_mat

        return loss
