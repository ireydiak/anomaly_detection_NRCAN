from typing import Tuple

import torch
import torch.nn as nn
from src.model.Encoder import Encoder
from src.model.Decoder import Decoder


class MLAD(nn.Module):
    common_net = None
    err_net = None
    repr_net = None
    exchange_net = None
    gmm_net = None

    def __init__(
            self, D: int, L: int, K: int,
            **kwargs
    ):
        """

        Parameters
        ----------
        D: Number of features (dimensionality)
        L: Size of latent space (dimensionality of latent space)
        K: Number of gaussian mixtures
        """
        super(MLAD, self).__init__()
        # Common network
        common_net_layers = kwargs.get('common_layers',
                                       [(D, 64, nn.ReLU()), (64, 64, nn.ReLU()), (64, L, nn.Sigmoid())])
        self.common_net = Encoder(common_net_layers)
        # Error network
        err_net_layers = kwargs.get('error_layers', [(D, 64, nn.ReLU()), (64, 64, nn.ReLU()), (64, L, nn.Sigmoid())])
        self.error_net = Encoder(err_net_layers)
        # Representation network
        repr_net_layers = kwargs.get('representation_layers',
                                     [(2 * L, 64, nn.ReLU()), (64, 96, nn.ReLU()), (96, D, nn.Sigmoid())])
        self.repr_net = Decoder(repr_net_layers)
        # Exchanger network
        exchange_net_layers = kwargs.get('exchanger_layers',
                                         [(L, 64, nn.ReLU()), (64, 64, nn.ReLU()), (64, D, nn.Sigmoid())])
        self.exchange_net = Decoder(exchange_net_layers)
        # GMM network
        gmm_net_layers = kwargs.get('gmm_layers', [
            (L, 16, nn.ReLU()), (16, 16, nn.ReLU()), (16, K, nn.Softmax()), (K, 16, nn.ReLU()),
            (16, 16, nn.ReLU()), (16, L, nn.Sigmoid())
        ])
        self.gmm_net = Encoder(gmm_net_layers)
        self.lambda_1 = kwargs.get('lambda_1', 1e-04)
        self.lambda_2 = kwargs.get('lambda_2', 1e-04)
        self.lambda_3 = kwargs.get('lambda_3', 1e-04)
        self.lambda_4 = kwargs.get('lambda_4', 1e-04)
        self.lambda_5 = kwargs.get('lambda_5', 1e-04)

    def _forward_single_stream(self, X: torch.Tensor):
        pass

    def _forward_two_stream(self, X_1: torch.Tensor, X_2: torch.Tensor) -> (Tuple, Tuple, Tuple, Tuple, Tuple):
        # common pass
        common_1 = self.common_net.forward(X_1)
        common_2 = self.common_net.forward(X_2)
        # error pass
        err_1 = self.error_net.forward(X_1)
        err_2 = self.error_net.forward(X_2)
        # GMM
        gmm_1 = self.gmm_net.forward(common_1)
        gmm_2 = self.gmm_net.forward(common_2)
        # Exchanger
        ex_1 = self.exchange_net.forward(common_1)
        ex_2 = self.exchange_net.forward(common_2)
        # Concat
        mix_1 = torch.cat(common_1, err_2)
        mix_2 = torch.cat(common_2, err_1)
        # Decode (representation)
        rec_1 = self.repr_net(mix_1)
        rec_2 = self.repr_net(mix_2)
        return (common_1, common_2), (err_1, err_2), (gmm_1, gmm_2), (ex_1, ex_2), (rec_1, rec_2)

    def reconstruction_loss(self, X_1: torch.Tensor, X_2: torch.Tensor, rec_tup: Tuple):
        rec_1, rec_2 = rec_tup[0], rec_tup[1]
        return torch.mean((rec_1 - X_1) ** 2) + torch.mean((rec_2 - X_2) ** 2)

    def exchanger_loss(self, X_1: torch.Tensor, X_2: torch.Tensor, ex_tup: Tuple):
        ex_1, ex_2 = ex_tup[0], ex_tup[1]
        return torch.mean((ex_1 - X_1) ** 2) + torch.mean((ex_2 - X_2) ** 2)

    def gmm_loss(self, common_tup: Tuple, gmm_tup: Tuple):
        gmm_1, gmm_2 = gmm_tup[0], gmm_tup[1]
        c_1, c_2 = common_tup[0], common_tup[1]
        return torch.mean((gmm_1 - c_1) ** 2) + torch.mean((gmm_2 - c_2) ** 2)

    def metric_loss(self):
        pass

    def loss(self, common_tup, err_tup, gmm_tup, ex_tup, rec_tup, X_1, X_2):
        rec_loss = self.reconstruction_loss(X_1, X_2, rec_tup)
        gmm_loss = self.gmm_lost(common_tup, gmm_tup)
        mix_loss = 0.0
        ex_loss = self.exchanger_loss(X_1, X_2, ex_tup)
        metric_loss = 0.0

        return self.lambda_1 * rec_loss + \
               self.lambda_2 * gmm_loss + \
               self.lambda_3 * mix_loss + \
               self.lambda_4 * ex_loss + \
               self.lambda_5 * metric_loss

    def forward(self, X_1: torch.Tensor, X_2: torch.Tensor = None):
        """
        Single forward pass dispatcher.
        During training, MLAD follows a two-input stream architecture.
        However testing is done on a single input matrix.

        Parameters
        ----------
        X_1: A torch.Tensor input matrix
        X_2: Another torch.Tensor input matrix

        Returns
        -------

        """
        if not X_2:
            self._forward_single_stream(X_1)
        else:
            return self._forward_two_stream(X_1, X_2)
