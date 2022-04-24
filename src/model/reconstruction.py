import gzip
import pickle

import numpy as np
import torch
from torch import nn
from minisom import MiniSom
from src.model.base import BaseModel
from src.model.GMM import GMM
from src.model.memory_module import MemoryUnit
from src.model import utils


class AutoEncoder(BaseModel):
    """
    Implements a basic Deep Auto Encoder
    """
    name = "AE"

    def __init__(
            self,
            latent_dim: int,
            act_fn: str,
            n_layers: int,
            compression_factor: int,
            **kwargs):
        super(AutoEncoder, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.act_fn = activation_mapper[act_fn]
        self.n_layers = n_layers
        self.compression_factor = compression_factor
        self.encoder, self.decoder = None, None
        self._build_network()

    def _build_network(self):
        # Create the ENCODER layers
        enc_layers = []
        in_features = self.in_features
        compression_factor = self.compression_factor
        for _ in range(self.n_layers - 1):
            out_features = in_features // compression_factor
            enc_layers.append(
                [in_features, out_features, self.act_fn]
            )
            in_features = out_features
            compression_factor += self.compression_factor
        enc_layers.append(
            [in_features, self.latent_dim, None]
        )
        # Create DECODER layers by simply reversing the encoder
        dec_layers = [[b, a, c] for a, b, c in reversed(enc_layers)]
        # Add and remove activation function from the first and last layer
        dec_layers[0][-1] = self.act_fn
        dec_layers[-1][-1] = None
        # Create networks
        self.encoder = utils.create_network(enc_layers)
        self.decoder = utils.create_network(dec_layers)

    @staticmethod
    def get_args_desc():
        return [
            ("latent_dim", int, 1, "Latent dimension of the AE network"),
            ("n_layers", int, 4, "Number of layers for the AE network"),
            ("compression_factor", int, 2, "Compression factor for the AE network"),
            ("act_fn", str, "relu", "Activation function of the AE network"),
        ]

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        """
        This function compute the output of the network in the forward pass
        :param x: input
        :return: output of the model
        """
        output = self.encoder(x)
        output = self.decoder(output)
        return x, output

    def get_params(self) -> dict:
        return {
            "latent_dim": self.latent_dim,
            "act_fn": self.act_fn.__str__,
            "n_layers": self.n_layers,
            "compression_factor": self.compression_factor
        }

    def reset(self):
        self.apply(self.weight_reset)

    def weight_reset(self, m):
        reset_parameters = getattr(m, "reset_parameters", None)
        if callable(reset_parameters):
            m.reset_parameters()

    @staticmethod
    def load(filename):
        # Load model from file (.pklz)
        with gzip.open(filename, 'rb') as f:
            model = pickle.load(f)
        assert isinstance(model, BaseModel)
        return model

    def save(self, filename):
        torch.save(self.state_dict(), filename)


# TODO: Move elsewhere (bootstrap maybe?)
activation_mapper = {
    "relu": nn.ReLU(),
    "tanh": nn.Tanh(),
    "sigmoid": nn.Sigmoid()
}


class DAGMM(BaseModel):
    """
    This class proposes an unofficial implementation of the DAGMM architecture proposed in
    https://sites.cs.ucsb.edu/~bzong/doc/iclr18-dagmm.pdf.
    Simply put, it's an end-to-end trained auto-encoder network complemented by a distinct gaussian mixture network.
    """
    name = "DAGMM"

    def __init__(
            self,
            n_mixtures: int,
            latent_dim: int,
            lambda_1: float,
            lambda_2: float,
            reg_covar: float,
            n_layers: int,
            compression_factor:int,
            ae_act_fn="relu",
            gmm_act_fn="tanh",
            **kwargs
    ):
        super(DAGMM, self).__init__(**kwargs)
        self.n_mixtures = n_mixtures
        self.latent_dim = latent_dim
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.reg_covar = reg_covar
        self.ae_n_layers = n_layers
        self.ae_compression_factor = compression_factor
        self.ae_act_fn = ae_act_fn
        self.gmm_act_fn = activation_mapper[gmm_act_fn]
        self.cosim = nn.CosineSimilarity()
        self.softmax = nn.Softmax(dim=-1)
        self.ae = None
        self.gmm = None
        self._build_network()

    @staticmethod
    def get_args_desc():
        return [
            ("n_mixtures", int, 4, "Number of mixtures for the GMM network."),
            ("latent_dim", int, 1, "Latent dimension of the AE network."),
            ("lambda_1", float, 0.1, "Coefficient for the energy loss."),
            ("lambda_2", float, 0.005, "Coefficient of the penalization for degenerate covariance matrices."),
            ("reg_covar", float, 1e-12,
             "Small epsilon value added to covariance matrix to ensure it remains invertible."),
            ("n_layers", int, 4, "Number of layers for the AE network."),
            ("compression_factor", int, 2, "Compression factor for the AE network."),
            ("ae_act_fn", str, "relu", "Activation function of the AE network."),
            ("gmm_act_fn", str, "tanh", "Activation function of the GMM network."),
        ]

    def _build_network(self):
        # Create GMM layers
        gmm_layers = [
            [self.latent_dim + 2, 10, self.gmm_act_fn],
            [None, None, nn.Dropout(0.5)],
            [10, self.n_mixtures, nn.Softmax(dim=-1)]
        ]
        # Create the sub-networks (AE and GMM)
        self.ae = AutoEncoder(
            latent_dim=self.latent_dim,
            act_fn=self.ae_act_fn,
            n_layers=self.ae_n_layers,
            compression_factor=self.ae_compression_factor,
            in_features=self.in_features,
            device=self.device,
            n_instances=self.n_instances
        )
        self.gmm = GMM(layers=gmm_layers)

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

    def estimate_sample_energy(self, z, phi=None, mu=None, cov_mat=None, average_energy=True):
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
        cov_mat = cov_mat + (torch.eye(d)).to(self.device) * eps
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
            "lambda_1": self.lambda_1,
            "lambda_2": self.lambda_2,
            "latent_dim": self.ae.latent_dim,
            "n_mixtures": self.gmm.K
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
    name = "SOMDAGMM"

    def __init__(
            self,
            n_soms: int,
            n_mixtures: int,
            latent_dim: int,
            reg_covar: float,
            n_layers: int,
            compression_factor: int,
            lambda_1: float,
            lambda_2: float,
            ae_act_fn="relu",
            gmm_act_fn="tanh",
            **kwargs):
        super(SOMDAGMM, self).__init__(**kwargs)
        self.n_mixtures = n_mixtures
        self.latent_dim = latent_dim
        self.reg_covar = reg_covar
        self.n_layers = n_layers
        self.compression_factor = compression_factor
        self.ae_act_fn = ae_act_fn
        self.gmm_act_fn = gmm_act_fn
        self.n_som = n_soms
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.som_args = None
        self.dagmm = None
        self.soms = None
        self._build_network()

    @staticmethod
    def get_args_desc():
        return [
            ("n_soms", int, 1, "Number of SOM"),
            ("n_mixtures", int, 4, "Number of mixtures for the GMM network"),
            ("latent_dim", int, 1, "Latent dimension of the AE network"),
            ("lambda_1", float, 0.005, "Lambda 1 parameter used during optimization"),
            ("lambda_2", float, 0.1, "Lambda 2 parameter used during optimization"),
            ("reg_covar", float, 1e-12,
             "Small epsilon value added to covariance matrix to ensure it remains reversible."),
            ("n_layers", int, 4, "Number of layers for the AE network"),
            ("compression_factor", int, 2, "Compression factor for the AE network"),
            ("ae_act_fn", str, "relu", "Activation function of the AE network"),
            ("gmm_act_fn", str, "tanh", "Activation function of the GMM network"),
        ]

    def _build_network(self):
        # set these values according to the used dataset
        # Use 0.6 for KDD; 0.8 for IDS2018 with babel as neighborhood function as suggested in the paper.
        grid_length = int(np.sqrt(5 * np.sqrt(self.n_instances))) // 2
        grid_length = 32 if grid_length > 32 else grid_length
        self.som_args = {
            "x": grid_length,
            "y": grid_length,
            "lr": 0.6,
            "neighborhood_function": "bubble",
            "n_epoch": 8000,
            "n_som": self.n_som
        }
        self.soms = [MiniSom(
            self.som_args['x'], self.som_args['y'], self.in_features,
            neighborhood_function=self.som_args['neighborhood_function'],
            learning_rate=self.som_args['lr']
        )] * self.som_args.get('n_som', 1)
        # DAGMM
        self.dagmm = DAGMM(
            in_features=self.in_features,
            n_instances=self.n_instances,
            device=self.device,
            n_mixtures=self.n_mixtures,
            latent_dim=self.latent_dim,
            lambda_1=self.lambda_1,
            lambda_2=self.lambda_2,
            reg_covar=self.reg_covar,
            n_layers=self.n_layers,
            compression_factor=self.compression_factor,
            ae_act_fn=self.ae_act_fn,
            gmm_act_fn=self.gmm_act_fn,
        )
        # Replace DAGMM's GMM network
        gmm_input = self.n_som * 2 + self.dagmm.latent_dim + 2
        gmm_layers = [
            (gmm_input, 10, nn.Tanh()),
            (None, None, nn.Dropout(0.5)),
            (10, self.n_mixtures, nn.Softmax(dim=-1))
        ]
        self.dagmm.gmm = GMM(gmm_layers).to(self.device)

    def train_som(self, X: torch.Tensor):
        # SOM-generated low-dimensional representation
        for i in range(len(self.soms)):
            self.soms[i].train(X, self.som_args["n_epoch"])

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

    def estimate_sample_energy(self, Z, phi, mu, Sigma, average_energy=True):
        return self.dagmm.estimate_sample_energy(Z, phi, mu, Sigma, average_energy=average_energy)

    def compute_loss(self, X, X_prime, energy, Sigma):
        rec_loss = ((X - X_prime) ** 2).mean()
        sample_energy = self.lambda_1 * energy
        penalty_term = self.lambda_2 * Sigma

        return rec_loss + sample_energy + penalty_term

    def get_params(self) -> dict:
        params = self.dagmm.get_params()
        for k, v in self.som_args.items():
            params[f'SOM-{k}'] = v
        return params


class MemAutoEncoder(BaseModel):
    name = "MemAE"

    def __init__(
            self,
            mem_dim: int,
            latent_dim: int,
            shrink_thres: float,
            n_layers: int,
            compression_factor: int,
            act_fn="relu",
            **kwargs
    ):
        """
        Implements model Memory AutoEncoder as described in the paper
        `Memorizing Normality to Detect Anomaly: Memory-augmented Deep Autoencoder (MemAE) for Unsupervised Anomaly Detection`.
        A few adjustments were made to train the model on tabular data instead of images.
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
        dataset_name: Name of the dataset (used to set the parameters)
        in_features: Number of variables in the dataset
        """
        super(MemAutoEncoder, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.act_fn = activation_mapper[act_fn]
        self.shrink_thres = shrink_thres
        self.n_layers = n_layers
        self.compression_factor = compression_factor
        self.mem_dim = mem_dim
        self.encoder = None
        self.decoder = None
        self.mem_rep = None
        self._build_network()

    @staticmethod
    def get_args_desc():
        return [
            ("shrink_thres", float, 0.0025, "Shrink threshold for hard shrinking relu"),
            ("latent_dim", int, 1, "Latent dimension of the AE network"),
            ("mem_dim", int, 50, "Number of memory units"),
            ("n_layers", int, 4, "Number of layers for the AE network"),
            ("alpha", float, 2e-4, "Coefficient for the entropy loss"),
            ("compression_factor", int, 2, "Compression factor for the AE network"),
            ("act_fn", str, "relu", "Activation function of the AE network"),
        ]

    def _build_network(self):
        # Create the ENCODER layers
        enc_layers = []
        in_features = self.in_features
        compression_factor = self.compression_factor
        for _ in range(self.n_layers - 1):
            out_features = in_features // compression_factor
            enc_layers.append(
                [in_features, out_features, self.act_fn]
            )
            in_features = out_features
            compression_factor += self.compression_factor
        enc_layers.append(
            [in_features, self.latent_dim, None]
        )
        # Create DECODER layers by simply reversing the encoder
        dec_layers = [[b, a, c] for a, b, c in reversed(enc_layers)]
        # Add and remove activation function from the first and last layer
        dec_layers[0][-1] = self.act_fn
        dec_layers[-1][-1] = None
        # Create networks
        self.encoder = utils.create_network(enc_layers)
        self.decoder = utils.create_network(dec_layers)
        self.mem_rep = MemoryUnit(self.mem_dim, self.latent_dim, self.shrink_thres, device=self.device).to(self.device)

    def forward(self, x):
        f_e = self.encoder(x)
        f_mem, att = self.mem_rep(f_e)
        f_d = self.decoder(f_mem)
        return f_d, att

    def get_params(self):
        return {
            "latent_dim": self.latent_dim,
            "in_features": self.in_features,
            "shrink_thres": self.shrink_thres,
            "mem_dim": self.mem_dim
        }
