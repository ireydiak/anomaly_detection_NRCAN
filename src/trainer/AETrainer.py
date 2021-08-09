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

from src.datamanager import DataManager
from src.model import DUAD
from sklearn.mixture import GaussianMixture

from src.utils.metrics import score_recall_precision
from src.viz.viz import plot_2D_latent, plot_energy_percentile


class AETrainer:

    def __init__(self, model: DUAD, dm: DataManager,
                 optimizer_factory: Callable[[torch.nn.Module], torch.optim.Optimizer],
                 use_cuda=True,
                 **kwargs
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

        self.criterion = nn.MSELoss()

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
        code, X_prime = self.model(X)
        l2_z = code.norm(2, dim=1).mean()
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
                code, X_prime = self.model(train_inputs)

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
                code, X_prime = self.model(test_inputs)

                test_score.append(((test_inputs - X_prime) ** 2).sum(axis=-1).squeeze().cpu().numpy())
                # test_score.append(h_x.cpu().numpy())
                test_z.append(code.cpu().numpy())
                test_labels.append(label_inputs.numpy())

            test_score = np.concatenate(test_score, axis=0)
            test_z = np.concatenate(test_z, axis=0)
            test_labels = np.concatenate(test_labels, axis=0)

            combined_score = np.concatenate([train_score, test_score], axis=0)
            # plot_energy_percentile(combined_score)

            # apc = average_precision_score(test_labels, test_score)
            #
            # precision, recall, thresholds = precision_recall_curve(test_labels, test_score)
            #

            # print(f"Precision:{precision}"
            #       f"\nRecall:{recall}"
            #       f"\nthresholds:{thresholds}"
            #       f"\nAPR:{apc}")
            # print(apc)

            # print((np.unique(test_labels)))
            # Search for best threshold

            score_recall_precision(combined_score, test_score, test_labels)

            # switch back to train mode
            self.model.train()

            # result_search = np.array(result_search)
            # best_result = np.max(result_search, axis=0)
            # idx_best_result = np.argmax(result_search, axis=0)
            #
            # res = {"Accuracy": best_result[0],
            #        "Precision": best_result[1],
            #        "Recall": best_result[2],
            #        "F1-Score": [3],
            #        'Confusion': confusion_matrices[idx_best_result],
            #        'Best threshold': thresholds[idx_best_result]
            #
            #        }
            res = {}
            return res, test_z, test_labels, combined_score
