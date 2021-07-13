from typing import List, Tuple

import torch
from torch import Tensor
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

from src.distributions import multivariate_normal_pdf, estimate_GMM_params
from src.metrics import accuracy_precision_recall_f1_scores
from src.model.MLAD import MLAD


class MLADTrainManager:
    def __init__(self, model: MLAD, train_set: Tensor, optim, use_cuda: bool = False, **kwargs):
        device_name = 'cuda:0' if use_cuda else 'cpu'
        self.device = torch.device(device_name)
        self.model = model.to(self.device)
        self.optim = optim(self.model)
        self.use_cuda = use_cuda
        self.train_set = train_set
        self.K = kwargs.get('K', 4)
        self.L = kwargs.get('L', None)
        self.verbose = kwargs.get('verbose', True)
        self.batch_size = kwargs.get('batch_size', 64)

    def fit_clusters(self, X: Tensor) -> List:
        Z = self.model.common_net.forward(X)
        clusters = KMeans(n_clusters=self.K, random_state=0).fit(Z)
        return clusters.labels_

    def create_batches(self, X_1: Tensor, X_2: Tensor) -> List[Tuple[Tensor, Tensor]]:
        N = self.batch_size
        # Number of batches
        n_batch = np.int(len(X_1) // N)
        # Handle the case where len(X_1) / N yields a remainder
        overflow = len(X_1) % N
        # Prepare the indices which will be used to split X_1 and X_2 in mini batches
        indices = [(i * n_batch, (i + 1) * n_batch) for i in range(0, n_batch)]
        # Last batch will contain remainder
        if overflow > 0:
            indices[-1][1] += overflow
        assert indices[-1][1] == len(X_1) - 1
        return [(X_1[start:end, :], X_2[start:end, :]) for start, end in indices]

    def create_samples(self, clusters) -> List[Tuple[Tensor, Tensor]]:
        X_1 = X_2 = None
        for k in range(0, self.K):
            [coding_idx] = np.where(clusters == k)
            X_1 = torch.cat((X_1, self.train_set[coding_idx, :])) if X_1 else self.train_set[coding_idx, :]
            np.random.shuffle(coding_idx)
            # TODO: we run the risk of training the model on the same data
            X_2 = torch.cat((X_2, self.train_set[coding_idx, :])) if X_2 else self.train_set[coding_idx, :]
        return self.create_batches(X_1, X_2)

    def train(self, n_epochs) -> None:
        train_loss = 0.0
        loss_history = []

        for epoch in range(n_epochs):
            print("Epoch: {} of {}".format(epoch + 1, n_epochs))
            clusters = self.fit_clusters(self.train_set)
            batches = self.create_samples(clusters)
            with tqdm(range(len(batches))) as t:
                for i, X_1, X_2 in enumerate(batches):
                    self.optim.zero_grad()
                    loss = self.forward(X_1, X_2)
                    loss_history.append(loss)
                    loss.backward()
                    self.optim.step()
                    train_loss += loss.items()
                    t.set_postfix(loss='{:05.3f}'.format(train_loss / (i + 1)))
                    t.update()
        return loss_history

    def compute_density(self, X: Tensor, phi: Tensor, mu: Tensor, Sigma: Tensor):
        density = 0.0
        # TODO: replace loops by matrix operations
        for k in range(0, self.K):
            density += multivariate_normal_pdf(X, phi[k], mu[k], Sigma[k, :, :])
        return density

    def compute_densities(self, test_set: Tensor, phi: Tensor, mu: Tensor, Sigma: Tensor) -> List[float]:
        test_z = self.model.common_net.forward(test_set)
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
            train_set_z = self.model.common_net.forward(self.train_set)
            phi, mu, Sigma = estimate_GMM_params(gmm_z, train_set_z)
            # 2- compute densities based on computed GMM parameters
            densities = self.compute_densities(test_set, phi, mu, Sigma)
            # 3- Find best p threshold
            return self.find_optimal_threshold(y, densities)

    def forward(self, X_1: Tensor, X_2: Tensor):
        common_tup, gmm_tup, ex_tup, rec_tup = self.model.forward(X_1, X_2)
        return self.model.loss(common_tup, gmm_tup, ex_tup, rec_tup, X_1, X_2)
