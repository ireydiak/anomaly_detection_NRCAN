import numpy as np
import torch
import torch.nn as nn
from minisom import MiniSom
from model.DAGMM import DAGMM


default_som_args={
    "x": 2,
    "y": 2,
    "lr": 0.6,
    "neighborhood_function": "bubble"
}


class SOMDAGMM(nn.Module):
    def __init__(self, input_len: int, dagmm: DAGMM, som_args: dict=None, **kwargs):
        super(SOMDAGMM, self).__init__()
        som_args = som_args or default_som_args
        # Use 0.6 for KDD; 0.8 for IDS2018
        self.som = MiniSom(
            som_args['x'], som_args['y'], input_len,
            neighborhood_function=som_args['neighborhood_function'],
            learning_rate=som_args['lr']
        )
        self.dagmm = dagmm
        self.lamb_1 = kwargs.get('lamb_1', 0.1)
        self.lamb_2 = kwargs.get('lamb_2', 0.005)

    def forward(self, X):
        # SOM-generated low-dimensional representation
        self.som.train(X, 200)
        # DAGMM's latent feature, the reconstruction error and gamma
        _, X_prime, _, z_r, gamma = self.dagmm(X)
        # Concatenate SOM's features with DAGMM's
        Z = torch.cat(z_s, z_r)
        
        return Z, X_prime, gamma
    
    def compute_params(self, Z, gamma):
        return self.dagmm.compute_params(Z, gamma)
    
    def estimate_sample_energy(self, Z, phi, mu, Sigma):
        return self.dagmm.estimate_sample_energy(Z, phi, mu, Sigma)

    def compute_loss(self, X, X_prime, energy, Sigma):
        rec_loss = ((X - X_prime) ** 2).mean()
        sample_energy = self.lamb_1 * energy
        penalty_term = self.lamb_2 * Sigma

        return rec_loss + sample_energy + penalty_term