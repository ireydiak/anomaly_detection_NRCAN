import warnings
from copy import deepcopy

from sklearn.metrics import confusion_matrix
from torch import nn
from tqdm import trange

import torch
import numpy as np

from typing import Callable

from sklearn import metrics

from src.datamanager import DataManager
from src.model import DUAD
from sklearn.mixture import GaussianMixture

from src.viz.viz import plot_2D_latent


class DUADTrainer:

    def __init__(self, model: DUAD, dm: DataManager,
                 optimizer_factory: Callable[[torch.nn.Module], torch.optim.Optimizer],
                 use_cuda=True,
                 **kwargs
                 ):

        self.metric_hist = []
        self.dm = dm

        self.r = kwargs.get('r', 10)
        self.p = kwargs.get('p', 30)
        self.p0 = kwargs.get('p0', 35)
        self.num_cluster = kwargs.get('num_cluster', 20)

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

        self.criterion = nn.MSELoss()

    def re_evaluation(self, X, p, num_clusters=20):
        gmm = GaussianMixture(n_components=num_clusters)
        gmm.fit(X)
        pred_label = gmm.predict(X)
        X_means = torch.from_numpy(gmm.means_)

        clusters_vars = []
        for i in range(num_clusters):
            var_ci = torch.sum((X[pred_label == i] - X_means[i].unsqueeze(dim=0)) ** 2)
            var_ci /= (pred_label == i).sum()
            clusters_vars.append(var_ci)

        clusters_vars = torch.stack(clusters_vars)
        q = torch.quantile(clusters_vars, (100 - p) / 100)

        selected_clusters = (clusters_vars <= q).nonzero().squeeze()
        # pred_label_ = torch.from_numpy(pred_label).unsqueeze(dim=1)
        indices_selection = torch.from_numpy(
            np.array([pred in selected_clusters for pred in pred_label])).nonzero().squeeze()

        return indices_selection

    def train(self, n_epochs: int):
        mean_loss = np.inf
        self.dm.update_train_set(self.dm.get_selected_indices())
        train_ldr = self.dm.get_train_set()

        # run clustering, select instances from low variance clusters
        X = []
        y = []
        indices = []
        for i, X_i in enumerate(train_ldr, 0):
            X.append(X_i[0])
            indices.append(X_i[2])
            y.append(X_i[1])

        X = torch.cat(X, axis=0)
        y = torch.cat(y, axis=0)
        indices = torch.cat(indices, axis=0)
        print(np.unique(y.cpu().numpy()))

        selected_indices = indices[self.re_evaluation(X, self.p0, self.num_cluster)]
        self.dm.update_train_set(selected_indices)
        train_ldr = self.dm.get_train_set()

        L = selected_indices
        L_old = []
        # print(set(L).difference(set(L_old)))
        while set(L).difference(set(L_old)) != set():
            for epoch in range(n_epochs):
                print(f"\nEpoch: {epoch + 1} of {n_epochs}")
                if (epoch + 1) % self.r == 0:
                    self.model.eval()
                    L_old = deepcopy(L)
                    with torch.no_grad():
                        # TODO
                        # Re-evaluate normality every r epoch
                        print("\nRe-evaluation")
                        indices = []
                        Z = []
                        X = []
                        y = []
                        X_loader = self.dm.get_init_train_loader()
                        for i, X_i in enumerate(X_loader, 0):
                            indices.append(X_i[2])
                            train_inputs = X_i[0].to(self.device).float()
                            code, X_prime, Z_r = self.model(train_inputs)
                            Z_i = torch.cat([code, Z_r.unsqueeze(-1)], axis=1)
                            Z.append(Z_i)
                            X.append(X_i)
                            y.append(X_i[1])

                        # X = torch.cat(X, axis=0)
                        indices = torch.cat(indices, axis=0)
                        y = torch.cat(y, axis=0)
                        Z = torch.cat(Z, axis=0)
                        print(np.unique(y.cpu().numpy()))

                        plot_2D_latent(Z.cpu(), y.cpu())

                        selection_mask = self.re_evaluation(Z.cpu(), self.p, self.num_cluster)
                        selected_indices = indices[selection_mask]
                        self.dm.update_train_set(selected_indices)
                        train_ldr = self.dm.get_train_set()

                    # switch back to train mode
                    self.model.train()
                    L = selected_indices
                    print(
                        f"Back to training--size L_old:{len(L_old)}, L:{len(L)}, diff:{len(set(L_old).difference(set(L)))}\n")

                else:
                    # TODO
                    # Train with the current trainset
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

        code, X_prime, h = self.model(X)
        loss = self.criterion(X, X_prime)

        # Use autograd to compute the backward pass.
        self.optim.zero_grad()
        loss.backward()
        # updates the weights using gradient descent
        self.optim.step()

        return loss.item()

    def evaluate_on_test_set(self, pos_label=1, **kwargs):
        """
        function that evaluate the model on the test set every iteration of the
        active learning process
        """
        energy_threshold = kwargs.get('energy_threshold', 80)
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

            for i, data in enumerate(train_loader, 0):
                # transfer tensors to selected device
                train_inputs, train_inputs_labels = data[0].float().to(self.device), data[1]

                # forward pass
                code, x_prime, cosim, z, gamma = self.model(train_inputs)
                sample_energy, pen_cov_mat = self.model.estimate_sample_energy(
                    z, train_phi, train_mu, train_cov, average_energy=False, device=self.device
                )

                train_energy.append(sample_energy.cpu().numpy())
                train_z.append(z.cpu().numpy())
                train_labels.append(train_inputs_labels.numpy())

            train_energy = np.concatenate(train_energy, axis=0)

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

            thresh = np.percentile(combined_energy, energy_threshold)
            print("Threshold :", thresh)

            # Prediction using the threshold value
            y_pred = (test_energy > thresh).astype(int)
            y_true = test_labels.astype(int)

            accuracy = metrics.accuracy_score(y_true, y_pred)
            precision, recall, f_score, _ = metrics.precision_recall_fscore_support(y_true, y_pred, average='binary',
                                                                                    pos_label=pos_label)
            cm = confusion_matrix(y_true, y_pred)
            res = {"Accuracy": accuracy,
                   "Precision": precision,
                   "Recall": recall,
                   "F1-Score": f_score,
                   'Confusion': cm
                   }

            print(f"Accuracy:{accuracy}, "
                  f"Precision:{precision}, "
                  f"Recall:{recall}, "
                  f"F-score:{f_score}, "
                  f"\nconfusion-matrix: {cm}")

            # switch back to train mode
            self.model.train()

            return res, test_z, test_labels, combined_energy
