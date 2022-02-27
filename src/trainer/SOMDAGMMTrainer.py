from torch.utils.data import DataLoader
import torch
import numpy as np


from trainer.BaseTrainer import BaseTrainer


class SOMDAGMMTrainer(BaseTrainer):

    def train_som(self, X):
        self.model.train_som(X)

    def train_iter(self, X):

        # SOM-generated low-dimensional representation
        code, X_prime, cosim, Z, gamma = self.model(X)

        phi, mu, Sigma = self.model.compute_params(Z, gamma)
        energy, penalty_term = self.model.estimate_sample_energy(Z, phi, mu, Sigma, device=self.device)

        loss = self.model.compute_loss(X, X_prime, energy, penalty_term)

        return loss.item()

    def test(self, dataset: DataLoader):
        """
        function that evaluate the model on the test set every iteration of the
        active learning process
        """
        self.model.eval()

        with torch.no_grad():

            scores, y_true = [], []

            for row in dataset:
                X, y = row
                X = X.to(self.device).float()

                sample_energy, _ = self.score(X)

                y_true.extend(y)
                scores.extend(sample_energy.cpu().numpy())

            return np.array(y_true), np.array(scores)

    def score(self, sample: torch.Tensor):
        code, x_prime, cosim, z, gamma = self.model(sample)
        phi, mu, cov_mat = self.model.compute_params(z, gamma)
        sample_energy, pen_cov_mat = self.model.estimate_sample_energy(
            z, phi, mu, cov_mat, average_energy=False
        )
        return sample_energy, pen_cov_mat
