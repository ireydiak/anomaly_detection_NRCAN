import warnings
from copy import deepcopy

from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, average_precision_score, precision_recall_curve, \
    plot_precision_recall_curve
from torch import nn
from tqdm import trange

import torch
import numpy as np

from typing import Callable

from sklearn import metrics

from datamanager.DataManager import DataManager
from model.DUAD import DUAD
from sklearn.mixture import GaussianMixture

from utils.metrics import score_recall_precision, score_recall_precision_w_thresold
from viz.viz import plot_2D_latent, plot_energy_percentile


class DUADTrainer:

    def __init__(self, model: DUAD, dm: DataManager,
                 optimizer_factory: Callable[[torch.nn.Module], torch.optim.Optimizer],
                 use_cuda=True,
                 **kwargs
                 ):

        self.metric_hist = []
        self.dm = dm

        self.r = kwargs.get('r', 10)
        self.p = kwargs.get('p_s', 30)
        self.p0 = kwargs.get('p_0', 35)
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
        # uv = np.unique(X, axis=0)
        gmm = GaussianMixture(n_components=num_clusters, max_iter=400)
        gmm.fit(X)
        pred_label = gmm.predict(X)
        X_means = torch.from_numpy(gmm.means_)

        clusters_vars = []
        for i in range(num_clusters):
            var_ci = torch.sum((X[pred_label == i] - X_means[i].unsqueeze(dim=0)) ** 2)
            var_ci /= (pred_label == i).sum()
            clusters_vars.append(var_ci)

        clusters_vars = torch.stack(clusters_vars)
        qp = 100 - p
        # q_ = np.percentile(clusters_vars, qp)
        q = torch.quantile(clusters_vars, qp / 100)

        selected_clusters = (clusters_vars <= q).nonzero().squeeze()
        # pred_label_ = torch.from_numpy(pred_label).unsqueeze(dim=1)

        selection_mask = [pred in list(selected_clusters.cpu().numpy()) for pred in pred_label]
        indices_selection = torch.from_numpy(
            np.array(selection_mask)).nonzero().squeeze()

        return indices_selection

    def train(self, n_epochs: int):
        print(f'Training with {self.__class__.__name__}')
        mean_loss = np.inf
        self.dm.update_train_set(self.dm.get_selected_indices())
        train_ldr = self.dm.get_train_set()
        REEVAL_LIMIT = 20

        # run clustering, select instances from low variance clusters
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

        # selected_indices = indices[self.re_evaluation(X, self.p0, self.num_cluster)]

        sel_from_clustering = self.re_evaluation(X, self.p0, self.num_cluster)
        selected_indices = indices[sel_from_clustering]

        print(f"label 0 ratio:{(y == 0).sum() / len(y)}"
              f"\n")
        print(f"label 1 ratio:{(y == 1).sum() / len(y)}"
              f"\n")
        print(f"selected label 0 ratio:{(y[sel_from_clustering] == 0).sum() / len(y)}"
              f"\n")
        print(f"selected label 1 ratio:{(y[sel_from_clustering] == 1).sum() / len(y)}"
              f"\n")

        self.dm.update_train_set(selected_indices)
        train_ldr = self.dm.get_train_set()

        L = []
        L_old = [-1]
        # print(set(L).difference(set(L_old)))
        reev_count = 0
        while len(set(L_old).difference(set(L))) <= 10 or reev_count > REEVAL_LIMIT:
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
                        Z = torch.cat(Z, axis=0)
                        y = torch.cat(y, axis=0).cpu().numpy()

                        # plot_2D_latent(Z.cpu(), y)

                        selection_mask = self.re_evaluation(Z.cpu(), self.p, self.num_cluster)
                        selected_indices = indices[selection_mask]
                        y_s = y[selection_mask.cpu().numpy()]

                        print(f"selected label 0 ratio:{(y_s == 0).sum() / len(y)}"
                              f"")
                        print(f"selected label 1 ratio:{(y_s == 1).sum() / len(y)}"
                              f"")

                        self.dm.update_train_set(selected_indices)
                        train_ldr = self.dm.get_train_set()

                    # switch back to train mode
                    self.model.train()
                    L = selected_indices.cpu().numpy()
                    print(
                        f"Back to training--size L_old:{len(L_old)}, L:{len(L)}, "
                        f"diff:{len(set(L_old).difference(set(L)))}\n")

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
            # self.evaluate_on_test_set()
            # break
        return mean_loss

    def train__(self, n_epochs: int):
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

        indices = torch.cat(indices, axis=0)
        y = torch.cat(y, axis=0).cpu().numpy().astype(int)

        sel_from_clustering = self.re_evaluation(X, self.p0, self.num_cluster)
        selected_indices = indices[sel_from_clustering]
        print(f"label 0 ratio:{(y[sel_from_clustering] == 0).sum() / len(y)}"
              f"")
        # TODO
        # to uncomment
        # self.dm.update_train_set(selected_indices)

        train_ldr = self.dm.get_train_set()

        L = selected_indices.cpu().numpy()
        L_old = [-1]
        # print(set(L).difference(set(L_old)))
        while len(set(L_old).difference(set(L))) != 0:
            for epoch in range(n_epochs):
                print(f"\nEpoch: {epoch + 1} of {n_epochs}")
                if False and (epoch + 1) % self.r == 0:
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
                        Z = torch.cat(Z, axis=0)
                        y = torch.cat(y, axis=0).cpu().numpy()

                        # plot_2D_latent(Z.cpu(), y)

                        selection_mask = self.re_evaluation(Z.cpu(), self.p, self.num_cluster)
                        selected_indices = indices[selection_mask]
                        y_s = y[selection_mask.cpu().numpy()]

                        print(f"label 0 ratio:{(y_s == 0).sum() / len(y_s)}"
                              f"")
                        print(f"label 1 ratio:{(y_s == 1).sum() / len(y_s)}"
                              f"")

                        self.dm.update_train_set(selected_indices)
                        train_ldr = self.dm.get_train_set()

                    # switch back to train mode
                    self.model.train()
                    L = selected_indices.cpu().numpy()
                    print(
                        f"Back to training--size L_old:{len(L_old)}, L:{len(L)}, "
                        f"diff:{len(set(L_old).difference(set(L)))}\n")

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
            self.evaluate_on_test_set()
            break
        return mean_loss

    def train_iter(self, X):

        code, X_prime, Z_r = self.model(X)
        l2_z = (torch.cat([code, Z_r.unsqueeze(-1)], axis=1).norm(2, dim=1)).mean()
        reg = 0.5
        loss = ((X - X_prime) ** 2).sum(axis=-1).mean() + reg * l2_z  # self.criterion(X, X_prime)

        # Use autograd to compute the backward pass.
        self.optim.zero_grad()
        loss.backward()
        # updates the weights using gradient descent
        self.optim.step()

        return loss.item()

    def evaluate_on_test_set(self, pos_label=1, **kwargs):
        """
        function that evaluate the model on the test set
        """

        energy_threshold = kwargs.get('threshold', 80)
        test_loader = self.dm.get_test_set()
        # Change the model to evaluation mode
        self.model.eval()
        train_score = []

        with torch.no_grad():
            # Create pytorch's train data_loader
            train_loader = self.dm.get_init_train_loader()
            for i, data in enumerate(train_loader, 0):
                # transfer tensors to selected device
                train_inputs = data[0].float().to(self.device)

                # forward pass
                code, X_prime, h_x = self.model(train_inputs)

                # (X - X_prime)

                # train_score.append(h_x.cpu().numpy())
                train_score.append(((train_inputs - X_prime) ** 2).sum(axis=-1).squeeze().cpu().numpy())
            train_score = np.concatenate(train_score, axis=0)

            # Calculate score using estimated parameters
            test_score = []
            test_labels = []
            test_z = []

            for data in test_loader:
                test_inputs, label_inputs = data[0].float().to(self.device), data[1]

                # forward pass
                # forward pass
                code, X_prime, h_x = self.model(test_inputs)

                test_score.append(((test_inputs - X_prime) ** 2).sum(axis=-1).squeeze().cpu().numpy())
                # test_score.append(h_x.cpu().numpy())
                test_z.append(code.cpu().numpy())
                test_labels.append(label_inputs.numpy())

            test_score = np.concatenate(test_score, axis=0)
            test_z = np.concatenate(test_z, axis=0)
            test_labels = np.concatenate(test_labels, axis=0)

            combined_score = np.concatenate([train_score, test_score], axis=0)

            res = score_recall_precision_w_thresold(combined_score, test_score, test_labels, pos_label=pos_label,
                                                    threshold=energy_threshold)

            score_recall_precision(combined_score, test_score, test_labels)

            # switch back to train mode
            self.model.train()

            return res, test_z, test_labels, combined_score
