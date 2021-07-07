from typing import Tuple

import torch
from torch import Tensor
import torch.nn as nn

from src.loss import mean_square_loss
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
        common_net_layers = kwargs.get(
            'common_layers', [(D, 64, nn.ReLU()), (64, 64, nn.ReLU()), (64, L, nn.Sigmoid())]
        )
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
        # TODO: Viz network
        self.gmm_net = Encoder(gmm_net_layers)
        self.lambda_1 = kwargs.get('lambda_1', 1e-04)
        self.lambda_2 = kwargs.get('lambda_2', 0.01)
        self.lambda_3 = kwargs.get('lambda_3', 0.01)
        self.lambda_4 = kwargs.get('lambda_4', 1e-04)

    def _forward_single_stream(self, X: Tensor):
        raise Warning("Unimplemented")

    def _forward_two_stream(self, X_1: Tensor, X_2: Tensor) -> (Tuple, Tuple, Tuple, Tuple):
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
        return (common_1, common_2), (gmm_1, gmm_2), (ex_1, ex_2), (rec_1, rec_2)

    def reconstruction_loss(self, X_1: Tensor, X_2: Tensor, X_hat_1: Tensor, X_hat_2: Tensor):
        return mean_square_loss(X_hat_1, X_2) + mean_square_loss(X_hat_2, X_1)

    def exchanger_loss(self, X_1: Tensor, X_2: Tensor, X_hat_1: Tensor, X_hat_2: Tensor):
        return mean_square_loss(X_1, X_hat_2) + mean_square_loss(X_2, X_hat_1)

    def gmm_loss(self, common_1: Tensor, common_2: Tensor, gmm_1: Tensor, gmm_2: Tensor):
        return mean_square_loss(common_1, gmm_1) + mean_square_loss(common_2, gmm_2)

    def common_loss(self, common_1: Tensor, common_2: Tensor):
        return mean_square_loss(common_1, common_2)

    def common_res_loss(self):
        # TODO: IMPLEMENT
        # ORIGINAL CODE BELOW
        # loss_common_res1 = tf.reduce_mean(tf.square(common1 - common1_res))
        # loss_common_res2 = tf.reduce_mean(tf.square(common2 - common2_res))
        raise Warning("Common res loss unimplemented")

    def metric_loss(self):
        # TODO: IMPLEMENT
        # ORIGINAL CODE BELOW
        # metric_input = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        # ...
        # dot_metric1 = tf.matmul(common1_gmm_coding * common1_gmm_codingz, tf.ones([K, 1]))
        # ...
        # loss_metric1 = tf.reduce_mean(tf.square(dot_metric1 - metric_input))
        # loss_metric2 = tf.reduce_mean(tf.square(dot_metric2 - metric_input))
        # loss_metric3 = tf.reduce_mean(tf.square(dot_metric3 - metric_input))
        # loss_metric4 = tf.reduce_mean(tf.square(dot_metric4 - metric_input))
        raise Warning("Metric loss unimplemented")

    def loss(self, common_tup, gmm_tup, exchanger_tup, rec_tup, X_1, X_2):
        com_loss = self.common_loss(*common_tup)
        rec_loss = self.reconstruction_loss(X_1, X_2, *rec_tup)
        gmm_loss = self.gmm_loss(*common_tup, *gmm_tup)
        exchanger_loss = self.exchanger_loss(X_1, X_2, *exchanger_tup)
        com_res_loss = 0.0
        metric_loss = 0.0

        return self.lambda_1 * com_loss + \
               rec_loss + \
               exchanger_loss + \
               self.lambda_2 * com_res_loss + \
               self.lambda_3 * gmm_loss + \
               self.lambda_4 * metric_loss

    def forward(self, X_1: Tensor, X_2: Tensor = None):
        """
        Single forward pass dispatcher.
        During training, MLAD follows a two-input stream architecture.
        However testing is done on a single input matrix.

        Parameters
        ----------
        X_1: A Tensor input matrix
        X_2: Another Tensor input matrix

        Returns
        -------

        """
        if not X_2:
            self._forward_single_stream(X_1)
        else:
            return self._forward_two_stream(X_1, X_2)
