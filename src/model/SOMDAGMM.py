import numpy as np
import torch
import torch.nn as nn
from minisom import MiniSom

from model import DAGMM

default_som_args = {
    "x": 32,
    "y": 32,
    "lr": 0.6,
    "neighborhood_function": "bubble",
    "n_epoch": 500,
    "n_som": 1
}


class SOMDAGMM(nn.Module):
    def __init__(self, input_len: int, dagmm: DAGMM, som_args: dict = None, **kwargs):
        super(SOMDAGMM, self).__init__()
        self.som_args = som_args or default_som_args
        # Use 0.6 for KDD; 0.8 for IDS2018 with babel as neighborhood function as suggested in the paper.
        self.soms = [MiniSom(
            self.som_args['x'], self.som_args['y'], input_len,
            neighborhood_function=self.som_args['neighborhood_function'],
            learning_rate=self.som_args['lr']
        )] * self.som_args.get("n_som", 1)
        self.dagmm = dagmm
        self.lamb_1 = kwargs.get('lamb_1', 0.1)
        self.lamb_2 = kwargs.get('lamb_2', 0.005)

        # pretrain som as part of the initialization process

    def train_som(self, X):
        # SOM-generated low-dimensional representation
        for i in range(len(self.soms)):
            self.soms[i].train(X, self.som_args['n_epoch'])

    def forward(self, X):
        # DAGMM's latent feature, the reconstruction error and gamma
        # _, X_prime, _, z_r = self.dagmm.forward_end_dec(X)
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

        # Z = z_r

        # estimation network
        gamma = self.dagmm.forward_estimation_net(Z)

        return code, X_prime, cosim, Z, gamma

    def compute_params(self, Z, gamma):
        return self.dagmm.compute_params(Z, gamma)

    def estimate_sample_energy(self, Z, phi, mu, Sigma, average_energy=True, device='cpu'):
        return self.dagmm.estimate_sample_energy(Z, phi, mu, Sigma, average_energy=average_energy, device=device)

    def compute_loss(self, X, X_prime, energy, Sigma):
        rec_loss = ((X - X_prime) ** 2).mean()
        sample_energy = self.lamb_1 * energy
        penalty_term = self.lamb_2 * Sigma

        return rec_loss + sample_energy + penalty_term

    def get_params(self) -> dict:
        params = self.dagmm.get_params()
        for k, v in self.som_args.items():
            params[f'SOM-{k}'] = v
        return params
