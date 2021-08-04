from typing import List, Tuple

import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
from sklearn.cluster import KMeans
from tqdm import trange
from src.distributions import multivariate_normal_pdf, estimate_GMM_params
from src.metrics import accuracy_precision_recall_f1_scores
from src.model.MLAD import MLAD
import random


class MLADTrainer:
    def __init__(self, model: MLAD, train_set: Tensor, optim, device: str = 'cpu', **kwargs):
        self.device = torch.device(device)
        self.model = model
        self.optim = optim(self.model)
        self.train_set = train_set
        self.D = self.train_set.shape[1]
        self.K = kwargs.get('K', 4)
        self.L = kwargs.get('L', 1)
        self.verbose = kwargs.get('verbose', True)
        self.batch_size = kwargs.get('batch_size', 64)

    def fit_clusters(self, X: Tensor) -> np.ndarray:
        # TODO: Question: train common_net prior or just a single pass?
        Z = self.model.common_pass(X)
        clusters = KMeans(n_clusters=self.K, random_state=0).fit(Z.detach().numpy())
        return clusters.labels_

    def create_batches(self, X_1: Tensor, X_2: Tensor, Z_1: Tensor, Z_2: Tensor, metric_input) -> List[Tuple]:
        N = self.batch_size
        # Number of batches
        n_batch = np.int(len(X_1) // N)
        # Handle the case where len(X_1) / N yields a remainder
        overflow = (len(X_1) % N) - 1
        # Prepare the indices which will be used to split X_1 and X_2 in mini batches
        indices = [(i * N, (i + 1) * N) for i in range(0, n_batch)]
        # Last batch will contain remainder
        if overflow > 0:
            indices[-1] = (indices[-1][0], indices[-1][1] + overflow)
        assert indices[-1][1] == len(X_1) - 1
        return [(
            (X_1[start:end, :], X_2[start:end, :]),
            (Z_1[start:end, :], Z_2[start:end, :]),
            metric_input[start:end, :]
        ) for start, end in indices]

    def split_siamese(self, X_1, X_2, labels):
        # TODO: By shuffling again, are we not mixing the different clusters together?
        idx_1 = random.sample(range(0, len(X_1)), len(X_1))
        idx_2 = random.sample(range(0, len(X_2)), len(X_2))
        input_x1 = X_1[idx_1, :]
        input_x2 = X_2[idx_1, :]
        x_label = labels[idx_1, :]
        input_z1 = input_x1[idx_2, :]
        input_z2 = input_x2[idx_2, :]
        z_label = labels[idx_2, :]
        metric_label = (torch.abs(x_label - z_label) == 0).float()
        return input_x1, input_x2, input_z1, input_z2, metric_label

    def create_samples(self, clusters) -> Tuple[List[Tuple], Tensor]:
        assert self.D
        input_x1 = torch.zeros(0, self.D)
        input_x2 = torch.zeros(0, self.D)
        x_label = torch.zeros(0, 1)

        for i in range(0, self.K):
            [coding_index] = np.where(clusters == i)
            input_x1 = torch.vstack((input_x1, self.train_set[coding_index, :]))
            np.random.shuffle(coding_index)
            # TODO: we run the risk of training the model on the same data
            input_x2 = torch.vstack((input_x2, self.train_set[coding_index, :]))
            x_label = torch.vstack((x_label, torch.ones([len(coding_index), 1]) * i))

        input_x1, input_x2, input_z1, input_z2, metric_label = self.split_siamese(
            input_x1, input_x2, x_label
        )

        return self.create_batches(input_x1, input_x2, input_z1, input_z2, metric_label)

    def train(self, n_epochs) -> list:
        train_loss = 0.0
        loss_history = []

        for epoch in range(n_epochs):
            print("Epoch: {} of {}".format(epoch + 1, n_epochs))
            clusters = self.fit_clusters(self.train_set)
            batches = self.create_samples(clusters)
            with trange(len(batches)) as t:
                for i, meta_tup in enumerate(batches):
                    X_tup, Z_tup, metric_label = meta_tup[0], meta_tup[1], meta_tup[2]
                    loss = self.forward(*X_tup, *Z_tup, metric_label)
                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()
                    train_loss += loss.item()
                    loss_history.append(loss.item())
                    t.set_postfix(loss='{:05.3f}'.format(train_loss / (i + 1)))
                    t.update()
                print('Cumulative loss = {:10.4f}'.format(sum(loss_history)))
        return loss_history

    def compute_density(self, X: Tensor, phi: Tensor, mu: Tensor, Sigma: Tensor):
        density = 0.0
        # TODO: replace loops by matrix operations
        for k in range(0, self.K):
            density += multivariate_normal_pdf(X, phi[k], mu[k], Sigma[k, :, :])
        return density

    def compute_densities(self, test_set: Tensor, phi: Tensor, mu: Tensor, Sigma: Tensor) -> List[float]:
        test_z = self.model.common_pass(test_set)
        self.verbose and print(
            f'calculating GMM densities using \u03C6={phi.shape}, \u03BC={mu.shape}, \u03A3={Sigma.shape}')
        densities = []
        # TODO: replace loops by matrix operations
        for i in range(0, len(test_z)):
            densities[i] = self.compute_density(test_z[i], phi, mu, Sigma)
        return densities

    def evaluate(self, y, densities, p):
        anomaly_idx = np.where(densities < p)
        y_hat = np.zeros(len(densities))
        if len(anomaly_idx) > 0:
            y_hat[anomaly_idx] = 1
        return accuracy_precision_recall_f1_scores(y.squeeze(), y_hat)

    def find_optimal_threshold(self, y: Tensor, densities: List[float], scale_coef: float = 0.3125,
                               n_iter: int = 20_000):
        """
        Implements Jianhai's original method `Functions.search_Threshold_Metric`.

        Parameters
        ----------
        y
        densities
        scale_coef
        n_iter

        Returns
        -------

        """
        # `p` refers to the anomaly threshold
        p_hist = f1_hist = list()
        for i in range(n_iter):
            p = 10 ** (-i * scale_coef)
            p_hist.append(p)
            if self.verbose and (i + 1) % 10 == 0:
                print(f'iter {i}: threshold={p}')
            acc, precision, recall, f1 = self.evaluate(y.squeeze(), densities, p)
            f1_hist.append(f1)
        # The best p-threshold is the one that yield the best F1-Score
        idx_max = np.argmax(np.array(f1_hist))
        acc, precision, recall, f1 = self.evaluate(y.squeeze(), densities, p_hist[idx_max])
        return {'Accuracy': acc, 'Precision': precision, 'Recall': recall, 'F1-Score': f1}, p_hist[idx_max]

    def evaluate_on_test_set(self, test_set: Tensor, y):
        self.model.eval()
        with torch.no_grad():
            # 1- estimate GMM parameters
            gmm_z = self.model.gmm_net.encode(self.train_set)
            train_set_z = self.model.common_pass(self.train_set)
            phi, mu, Sigma = estimate_GMM_params(gmm_z, train_set_z)
            # 2- compute densities based on computed GMM parameters
            densities = self.compute_densities(test_set, phi, mu, Sigma)
            # 3- Find best p threshold
            return self.find_optimal_threshold(y, densities)

    def forward(self, X_1: Tensor, X_2: Tensor, Z_1: Tensor, Z_2: Tensor, metric_labels: Tensor):
        com_meta_tup, err_meta_tup, gmm_meta_tup, dot_met, ex_meta_tup, rec_meta_tup = self.model(X_1, X_2, Z_1, Z_2)
        return self.model.loss(
            com_meta_tup,
            gmm_meta_tup,
            dot_met,
            ex_meta_tup,
            rec_meta_tup,
            ((X_1, X_2), (Z_1, Z_2)),
            metric_labels
        )
