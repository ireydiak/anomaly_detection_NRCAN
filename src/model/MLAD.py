from typing import Tuple

import torch
from torch import Tensor
import torch.nn as nn

from src.model import AutoEncoder
from src.model import Encoder
from src.model import Decoder


class MLAD(nn.Module):

    def __init__(
            self, D: int, L: int, K: int, **kwargs
    ):
        """

        Parameters
        ----------
        D: Number of features (n_features)
        L: Size of latent space (dimensionality of latent space)
        K: Number of gaussian mixtures
        """
        super(MLAD, self).__init__()
        # Common network
        self.common_net = Encoder([(D, 64, nn.ReLU()), (64, 64, nn.ReLU()), (64, L, nn.Sigmoid())])
        # Error network
        err_net_layers = kwargs.get('error_layers', [(D, 64, nn.ReLU()), (64, 64, nn.ReLU()), (64, L, nn.Sigmoid())])
        self.error_net = Encoder(err_net_layers)
        # Representation network (Decoder)
        repr_net_layers = kwargs.get(
            'representation_layers', [(2 * L, 64, nn.ReLU()), (64, 96, nn.ReLU()), (96, D, nn.Sigmoid())]
        )
        self.repr_net = Decoder(repr_net_layers)
        # Exchanger network (Decoder)
        exchange_net_layers = kwargs.get(
            'exchanger_layers', [(L, 64, nn.ReLU()), (64, 64, nn.ReLU()), (64, D, nn.Sigmoid())]
        )
        self.exchange_net = Decoder(exchange_net_layers)
        # GMM network
        gmm_net_layers = kwargs.get('gmm_layers', [
            ((L, 16, nn.ReLU()), (16, 16, nn.ReLU()), (16, K, nn.Softmax(dim=1))),
            ((K, 16, nn.ReLU()), (16, 16, nn.ReLU()), (16, L, nn.Sigmoid()))
        ])
        self.gmm_net = AutoEncoder(gmm_net_layers[0], gmm_net_layers[1])
        self.lambda_1 = kwargs.get('lambda_1', 1e-04)
        self.lambda_2 = kwargs.get('lambda_2', 0.01)
        self.lambda_3 = kwargs.get('lambda_3', 1e-04)
        self.K = K
        self.L = L
        self.D = D

    def reconstruction_loss(self, X_1_hat: Tensor, X_2_hat: Tensor, X_1: Tensor, X_2: Tensor):
        loss = nn.MSELoss()
        return loss(X_1_hat, X_2) + loss(X_2_hat, X_1)

    def exchanger_loss(self, X_1_hat: Tensor, X_2_hat: Tensor, X_1: Tensor, X_2: Tensor):
        loss = nn.MSELoss()
        return loss(X_1, X_1_hat) + loss(X_2, X_2_hat)

    def gmm_loss(self, common_1: Tensor, common_2: Tensor, gmm_1: Tensor, gmm_2: Tensor):
        loss = nn.MSELoss()
        return loss(common_1, gmm_1) + loss(common_2, gmm_2)

    def common_loss(self, common_1: Tensor, common_2: Tensor):
        loss = nn.MSELoss()
        return loss(common_1, common_2)

    def metric_loss(self, dot_metric, metric_input):
        loss = nn.MSELoss()
        return loss(dot_metric, metric_input)

    def loss(self, com_meta_tup, gmm_meta_tup, dot_metrics, ex_meta_tup, rec_meta_tup, samples_meta_tup, metric_label):
        """
        Computes the loss from the siamese two-input streams

        Parameters
        ----------
        com_meta_tup: A Tuple of two-input Tensors (common)
        gmm_meta_tup: A Tuple of two-input Tensors (gmm)
        dot_metrics: A (n_sample, 1) Tensor
        ex_meta_tup: A Tuple of two-input Tensors (exchanger)
        rec_meta_tup: A Tuple of two-input Tensors (reconstructor)
        samples_meta_tup: A Tuple of two-input Tensors containing the original samples
        metric_label: A (n_sample, 1) Tensor

        Returns
        -------
        The sum of MSE loss for the five sub-networks
        """
        # Common Loss
        common_loss_A = self.common_loss(*com_meta_tup[0])
        common_loss_B = self.common_loss(*com_meta_tup[1])
        # Reconstruction Loss
        rec_loss_A = self.reconstruction_loss(*rec_meta_tup[0], *samples_meta_tup[0])
        rec_loss_B = self.reconstruction_loss(*rec_meta_tup[1], *samples_meta_tup[1])
        # Exchanger Loss
        ex_loss_A = self.exchanger_loss(*ex_meta_tup[0], *samples_meta_tup[0])
        ex_loss_B = self.exchanger_loss(*ex_meta_tup[1], *samples_meta_tup[1])
        # GMM Loss
        gmm_loss_A = self.gmm_loss(*com_meta_tup[0], *gmm_meta_tup[0])
        gmm_loss_B = self.gmm_loss(*com_meta_tup[1], *gmm_meta_tup[1])
        # Metric Loss
        metric_loss = sum([self.metric_loss(dot_metric, metric_label) for dot_metric in dot_metrics])
        # Compute losses
        loss_A = self.lambda_1 * common_loss_A + rec_loss_A + ex_loss_A + self.lambda_2 * (gmm_loss_A + gmm_loss_B)
        loss_B = self.lambda_1 * common_loss_B + rec_loss_B + ex_loss_B + self.lambda_2 * (gmm_loss_B + gmm_loss_B)
        loss_metric = self.lambda_3 * metric_loss
        return loss_A + loss_B + loss_metric

    def forward_one(self, X: Tensor) -> (float, float, float, float, float):
        """
        Performs a forward pass on a single input

        Parameters
        ----------
        X: A (n_inputs, n_features) Tensor

        Returns
        -------
        The output of the five MLAD networks.
        """
        # Common
        common = self.common_net(X)
        # Error
        err = self.error_net(X)
        # GMM, GMM coding
        gmm, gmm_z = self.gmm_net(common)
        # Exchanger
        ex = self.exchange_net(common)

        return common, err, gmm, gmm_z, ex

    def forward_two(self, X_1: Tensor, X_2: Tensor):
        """
        Performs a forward pass on a two-input stream

        Parameters
        ----------
        X_1: A (n_inputs, n_features) Tensor
        X_2: A (n_inputs, n_features) Tensor

        Returns
        -------
        Tuples from the five sub-networks
        """
        common_1, err_1, gmm_1, gmm_z_1, ex_1 = self.forward_one(X_1)
        common_2, err_2, gmm_2, gmm_z_2, ex_2 = self.forward_one(X_2)
        # Concat
        mix_1 = torch.cat((common_1, err_2), dim=1)
        mix_2 = torch.cat((common_2, err_1), dim=1)
        # Decode
        rec_1 = self.repr_net(mix_1)
        rec_2 = self.repr_net(mix_2)
        return (common_1, common_2), (err_1, err_2), (gmm_1, gmm_2), (gmm_z_1, gmm_z_2), (ex_1, ex_2), (rec_1, rec_2)

    def forward(self, X_1: Tensor, X_2: Tensor, Z_1: Tensor, Z_2: Tensor) -> (Tuple, Tuple, Tuple, Tuple, Tuple, Tuple):
        """
        Performs a forward pass on the model.
        Since we are using a siamese two-input stream network, the forward pass is decoupled into multiple
        smaller functions.

        Parameters
        ----------
        X_1: A (n_inputs, n_features) Tensor
        X_2: A (n_inputs, n_features) Tensor
        Z_2: A (n_inputs, n_features) Tensor
        Z_1: A (n_inputs, n_features) Tensor

        Returns
        -------

        """
        common_tup_1, err_tup_1, gmm_tup_1, gmm_tup_z_1, ex_tup_1, rec_tup_1 = self.forward_two(X_1, X_2)
        common_tup_2, err_tup_2, gmm_tup_2, gmm_tup_z_2, ex_tup_2, rec_tup_2 = self.forward_two(Z_1, Z_2)
        dot_metrics = (
            (gmm_tup_z_1[0] * gmm_tup_z_2[0]) @ torch.ones(self.K, 1),
            (gmm_tup_z_1[0] * gmm_tup_z_2[1]) @ torch.ones(self.K, 1),
            (gmm_tup_z_1[1] * gmm_tup_z_2[0]) @ torch.ones(self.K, 1),
            (gmm_tup_z_1[1] * gmm_tup_z_2[1]) @ torch.ones(self.K, 1)
        )

        return (common_tup_1, common_tup_2), \
               (err_tup_1, err_tup_2), \
               (gmm_tup_1, gmm_tup_2), \
               dot_metrics, \
               (ex_tup_1, ex_tup_2), \
               (rec_tup_1, rec_tup_2)

    def common_pass(self, X):
        return self.common_net(X)
