import warnings

from sklearn.metrics import confusion_matrix
from tqdm import trange
from model import SOMDAGMM
import torch
import numpy as np
from datamanager.DataManager import DataManager
from typing import Callable

from sklearn import metrics

from utils import score_recall_precision

from utils import score_recall_precision_w_thresold


class SOMDAGMMTrainer:

    def __init__(self, model: SOMDAGMM, dm: DataManager,
                 optimizer_factory: Callable[[torch.nn.Module], torch.optim.Optimizer],
                 use_cuda=True,
                 ):

        self.metric_hist = []
        self.dm = dm

        device_name = 'cuda:0' if use_cuda else 'cpu'
        if use_cuda and not torch.cuda.is_available():
            warnings.warn("CUDA is not available. Suppress this warning by passing "
                          "use_cuda=False to {}()."
                          .format(self.__class__.__name__), RuntimeWarning)
            print('\n\n')
            device_name = 'cpu'

        self.device = torch.device(device_name)
        self.model = model.to(self.device)
        self.optim = optimizer_factory(self.model)

    def train_som(self, X):
        self.model.train_som(X)

    def train(self, n_epochs: int):
        print(f'Training with {self.__class__.__name__}')
        mean_loss = np.inf
        train_ldr = self.dm.get_train_set()

        for epoch in range(n_epochs):
            print(f"\nEpoch: {epoch + 1} of {n_epochs}")
            loss = 0
            with trange(len(train_ldr)) as t:
                for i, X_i in enumerate(train_ldr, 0):
                    train_inputs = X_i[0].to(self.device).float()
                    loss += self.train_iter(train_inputs)
                    mean_loss = loss / (i + 1)
                    t.set_postfix(loss='{:05.3f}'.format(mean_loss))
                    t.update()
        return mean_loss

    def train_iter(self, X):
        self.optim.zero_grad()

        # SOM-generated low-dimensional representation
        code, X_prime, cosim, Z, gamma = self.model(X)

        phi, mu, Sigma = self.model.compute_params(Z, gamma)
        energy, penalty_term = self.model.estimate_sample_energy(Z, phi, mu, Sigma, device=self.device)

        loss = self.model.compute_loss(X, X_prime, energy, penalty_term)

        # Use autograd to compute the backward pass.
        loss.backward()

        # updates the weights using gradient descent
        self.optim.step()

        return loss.item()

    def evaluate_on_test_set(self, pos_label=1, **kwargs):
        """
        function that evaluate the model on the test set every iteration of the
        active learning process
        """
        energy_threshold = kwargs.get('threshold', 80)
        test_loader = self.dm.get_test_set()
        N = gamma_sum = mu_sum = cov_mat_sum = 0

        # Change the model to evaluation mode
        self.model.eval()

        with torch.no_grad():
            # Create pytorch's train data_loader
            train_loader = self.dm.get_train_set()

            for i, data in enumerate(train_loader, 0):
                # transfer tensors to selected device
                train_inputs = data[0].float().to(self.device)

                # forward pass
                code, x_prime, cosim, z, gamma = self.model(train_inputs)
                phi, mu, cov_mat = self.model.compute_params(z, gamma)

                batch_gamma_sum = gamma.sum(axis=0)

                gamma_sum += batch_gamma_sum
                mu_sum += mu * batch_gamma_sum.unsqueeze(-1)  # keep sums of the numerator only
                cov_mat_sum += cov_mat * batch_gamma_sum.unsqueeze(-1).unsqueeze(-1)  # keep sums of the numerator only
                N += train_inputs.shape[0]

            train_phi = gamma_sum / N
            train_mu = mu_sum / gamma_sum.unsqueeze(-1)
            train_cov = cov_mat_sum / gamma_sum.unsqueeze(-1).unsqueeze(-1)

            print("Train N:", N)
            print("\u03C6 :\n", train_phi.shape)
            print("\u03BC :\n", train_mu.shape)
            print("\u03A3 :\n", train_cov.shape)

            # Calculate energy using estimated parameters

            train_energy = []
            train_labels = []
            train_z = []


            test_energy = []
            test_labels = []
            test_z = []

            for data in test_loader:
                test_inputs, label_inputs = data[0].float().to(self.device), data[1]

                # forward pass
                code, x_prime, cosim, z, gamma = self.model(test_inputs)
                sample_energy, pen_cov_mat = self.model.estimate_sample_energy(
                    z, train_phi, train_mu, train_cov, average_energy=False, device=self.device
                )
                test_energy.append(sample_energy.cpu().numpy())
                test_z.append(z.cpu().numpy())
                test_labels.append(label_inputs.numpy())

            test_energy = np.concatenate(test_energy, axis=0)
            test_z = np.concatenate(test_z, axis=0)
            test_labels = np.concatenate(test_labels, axis=0)

            combined_energy = np.concatenate([train_energy, test_energy], axis=0)

            comp_threshold = 100 * sum(test_labels == 0) / len(test_labels)
            res_max = score_recall_precision(combined_energy, test_energy, test_labels)
            res = score_recall_precision_w_thresold(combined_energy, test_energy, test_labels, pos_label=pos_label,
                                                    threshold=comp_threshold)

            res = dict(res, **res_max)

            # switch back to train mode
            self.model.train()

        return res, test_z, test_labels, combined_energy
