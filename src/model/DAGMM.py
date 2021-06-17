import torch.nn as nn
import torch.functional as F
from model.AutoEncoder import AutoEncoder as AE
import torch
import numpy as np


class DAGMM(nn.Module):
    """
    this class implements an deep auto encoder gaussian mixture model using
    fullyconnected neural net
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

        code_shape = ae_layers_unit[-1] + 2  # 2 for the euclidean error and the cosine similarity
        self.ae = AE(input_size, ae_layers_unit, fa=fa, last_layer_activation=last_layer_activation)

        self.cosim = nn.CosineSimilarity()
        self.softmax = nn.Softmax(dim=-1)

        self.gmm_predictor = self._make_linear(code_shape, gmm_layers, fa=fa, gmm_drop_out_last=gmm_drop_out_last)

        self.phi = None
        self.mu = None
        self.cov_mat = None
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2

    def _make_linear(self, input_size, layers, last_layer_activation=False, fa='relu', gmm_drop_out_last=False):
        """
        This function builds a linear model whose units and layers depend on
        the passed @layers argument
        :param layers: a list of unit per layer
        :return: a fully connected neural net (Sequentiel object)
        """
        activations = dict(relu=nn.ReLU(True), tanh=nn.Tanh())
        net_layers = [nn.Linear(input_size, layers[0]), activations[fa]]
        for i in range(1, len(layers)):

            # drop out for the last layer
            if i == len(layers) - 1:
                net_layers.append(nn.Dropout())

            net_layers.append(nn.Linear(layers[i - 1], layers[i]))

            if i != len(layers) - 1:
                net_layers.append(activations[fa])
            else:
                if last_layer_activation:
                    net_layers.append(activations[fa])
        return nn.Sequential(*net_layers)

    def forward(self, x):
        """
        This function compute the output of the network in the forward pass
        :param x: input
        :return: output of the model
        """

        # print(f'\nnan={np.nonzero(np.isnan(x.data.cpu().numpy()))}')

        code = self.ae.encoder(x)
        x_hat = self.ae.decoder(code)

        # compute similarity measures to concatenate with the encoded x
        rel_euc_dist = self.relative_eucludian_dist(x, x_hat)
        cosim = self.cosim(x, x_hat)

        z_error = torch.cat([code, rel_euc_dist.unsqueeze(-1), cosim.unsqueeze(-1)], dim=1)

        # compute gmm net output
        output = self.gmm_predictor(z_error)
        gamma = self.softmax(output)

        return code, x_hat, cosim, z_error, gamma

    def relative_eucludian_dist(self, x, x_hat):
        return (x - x_hat).norm(2, dim=1) / x.norm(2, dim=1)

    def compute_params(self, z, gamma):

        # gamma : N x K
        # z : N x D

        N = z.shape[0]

        # K
        gamma_sum = torch.sum(gamma, dim=0)
        phi = gamma_sum / N

        # K x D
        mu = torch.sum(gamma.unsqueeze(-1) * z.unsqueeze(1), dim=0) / gamma_sum.unsqueeze(-1)

        mu_z = z.unsqueeze(1) - mu.unsqueeze(0)
        # print(f'mu_z={mu_z}')
        mu_z_t = mu_z.unsqueeze(-1)
        mu_z_ = mu_z.unsqueeze(-2)
        cov_mat = torch.matmul(mu_z.unsqueeze(-1), mu_z.unsqueeze(-2))
        cov_mat = gamma.unsqueeze(-1).unsqueeze(-1) * cov_mat
        # K x D x D
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

        #
        eps = 1e-12
        d = z.shape[1]
        # eye_mat = torch.eye(d) * eps
        # test = cov_mat.unsqueeze(1)
        # t = test + eye_mat

        cov_mat = cov_mat + (torch.eye(d)).to(device) * eps

        mu_z = z.unsqueeze(1) - mu.unsqueeze(0)

        # l = torch.linalg.cholesky(cov_mat)

        inv_cov_mat = torch.linalg.inv(cov_mat)

        # det_cov_mat = 2 * np.pi * torch.linalg.det(cov_mat)
        # det_cov_mat = torch.sqrt(det_cov_mat)

        det_cov_mat = torch.linalg.cholesky(2 * np.pi * cov_mat)
        det_cov_mat = torch.diagonal(det_cov_mat, dim1=1, dim2=2)
        det_cov_mat = torch.prod(det_cov_mat, dim=1)
        # print(f"\ndet:{det_cov_mat}")
        det_cov_mat = torch.sqrt(det_cov_mat)

        exp_term = torch.matmul(mu_z.unsqueeze(-2), inv_cov_mat)
        exp_term = torch.matmul(exp_term, mu_z.unsqueeze(-1))
        exp_term = - 0.5 * exp_term.squeeze()

        # print(f"\nexpterm:{exp_term}")

        # Applying log-sum-exp stability trick
        # https://www.xarg.org/2016/06/the-log-sum-exp-trick-in-machine-learning/
        max_val = torch.max(exp_term.clamp(min=0), dim=1, keepdim=True)[0]

        exp_result = torch.exp(exp_term - max_val)
        # print(f"\nexpterm trick:{exp_result}")
        # exp_result = torch.exp(exp_term)

        log_term = phi * exp_result
        log_term /= det_cov_mat
        log_term = log_term.sum(axis=-1)

        # energy computation
        energy_result = - max_val.squeeze() - torch.log(log_term + eps)
        # energy_result = - torch.log(log_term)
        if average_it:
            energy_result = energy_result.mean()

        # penalty term
        cov_diag = torch.diagonal(cov_mat, dim1=1, dim2=2)
        pen_cov_mat = (1 / cov_diag).sum()

        # energy_result_, pen_cov_mat_ = self.compute_energy(z, phi, mu, cov_mat, average_it,device)
        # print(f'\n{energy_result} -- {energy_result_}, \n{pen_cov_mat} -- {pen_cov_mat_}')
        return energy_result, pen_cov_mat

    def compute_loss(self, x, x_hat, energy, pen_cov_mat):
        """

        """
        rec_err = ((x - x_hat) ** 2).mean()  # .mean(axis=-1)
        loss = rec_err + self.lambda_1 * energy + self.lambda_2 * pen_cov_mat

        return loss
