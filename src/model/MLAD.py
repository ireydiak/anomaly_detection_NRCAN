import torch
import torch.nn as nn
from src.model.Encoder import Encoder
from src.model.Decoder import Decoder


class MLAD(nn.Module):

    def __init__(
            self, D: int, L: int, K: int,
            common_layers=None, error_layers=None, representation_layers=None, exchanger_layers=None, gmm_layers=None
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
        common_net_layers = common_layers or [(D, 64, nn.ReLU()), (64, 64, nn.ReLU()), (64, L, nn.Sigmoid())]
        self.common_net = Encoder(common_net_layers)
        # Error network
        err_net_layers = error_layers or [(D, 64), (64, 64), (64, L)]
        self.error_net = Encoder(err_net_layers)
        # Representation network
        repr_net_layers = representation_layers or [(2 * L, 64, nn.ReLU()), (64, 96, nn.ReLU()), (96, D, nn.Sigmoid())]
        self.repr_net = Decoder(repr_net_layers)
        # Exchanger network
        exchange_net_layers = exchanger_layers or [(L, 64, nn.ReLU()), (64, 64, nn.ReLU()), (64, D, nn.Sigmoid())]
        self.exchange_net = Decoder(exchange_net_layers)
        # GMM network
        gmm_net_layers = gmm_layers or [
            (L, 16, nn.ReLU()), (16, 16, nn.ReLU()), (16, K, nn.Softmax()), (K, 16, nn.ReLU()),
            (16, 16, nn.ReLU()), (16, L, nn.Sigmoid())
        ]
        self.gmm_net = Encoder(gmm_net_layers)

    def _forward_single_stream(self, X: torch.Tensor):
        pass

    def _forward_two_stream(self, X_1: torch.Tensor, X_2: torch.Tensor):
        # common pass
        common_1 = self.common_net.forward(X_1)
        common_2 = self.common_net.forward(X_2)
        # error pass
        err_1 = self.error_net.forward(X_1)
        err_2 = self.error_net.forward(X_2)

    def forward(self, sample_1: torch.Tensor, sample_2: torch.Tensor = None):
        """
        Single forward pass dispatcher.
        During training, MLAD follows a two-input stream architecture.
        However testing is done on a single input matrix.

        Parameters
        ----------
        sample_1: A torch.Tensor input matrix
        sample_2: Another torch.Tensor input matrix

        Returns
        -------

        """
        if not sample_2:
            self._forward_single_stream(sample_1)
        else:
            self._forward_two_stream(sample_1, sample_2)
