from tqdm import trange
from model.SOMDAGMM import SOMDAGMM
import torch
import numpy as np


class SOMDAGMMTrainer:

    def __init__(self, model: SOMDAGMM, train_set, optimizer_factory):
        self.model = model
        self.optim = optimizer_factory(self.model)
        self.metric_hist = []
        self.train_set = train_set

    def train(self, n_epochs: int):
        mean_loss = np.inf
        with trange(n_epochs) as t:
            for i, X_i in enumerate(self.train_set):
                loss = self.train_iter(X_i)
                mean_loss = loss / (i + 1)
                t.set_postfix(loss='{:05.3f}'.format(mean_loss))
                t.update()
        return mean_loss

    def train_iter(self, X):
        self.optim.zero_grad()

        # SOM-generated low-dimensional representation
        Z, X_prime, gamma = self.model(X)

        phi, mu, Sigma = self.model.compute_params(Z, gamma)
        energy, penalty_term = self.model.estimate_sample_energy(Z, phi, mu, Sigma)

        loss = self.model.compute_loss(X, X_prime, energy, penalty_term)

        # Use autograd to compute the backward pass.
        loss.backward()

        # updates the weights using gradient descent
        self.optimizer.step()

        return loss.item()
    