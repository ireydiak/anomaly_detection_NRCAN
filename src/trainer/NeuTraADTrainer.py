import warnings
from copy import deepcopy

from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, average_precision_score, precision_recall_curve, \
    plot_precision_recall_curve
from torch import nn, optim
from torch.optim.lr_scheduler import ExponentialLR, StepLR
from tqdm import trange

import torch
import numpy as np

from typing import Callable

from datamanager import DataManager

from utils.metrics import score_recall_precision, score_recall_precision_w_thresold


class NeuTraADTrainer:

    def __init__(self, model, dm: DataManager,
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
        mask_params = list()
        for mask in self.model.masks:
            mask_params += list(mask.parameters())
        self.optim =optim.Adam(list(self.model.enc.parameters()) + mask_params, lr=kwargs.get('learning_rate'),
                               weight_decay=kwargs.get('weight_decay'))
        # self.optim = optimizer_factory()
        self.scheduler = StepLR(self.optim, step_size=20, gamma=0.9)

        self.criterion = nn.MSELoss()

    def train(self, n_epochs: int):
        print(f'Training with {self.__class__.__name__}')
        mean_loss = np.inf
        train_ldr = self.dm.get_train_set()

        lrs = []
        losses = []
        val_losses = []

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
            losses.append(loss / len(train_ldr))
            lrs.append(self.optim.param_groups[0]["lr"])
            # val_losses.append(self.evaluate_on_validation_set())
            # if epoch % 10 == 0:
            #     self.evaluate_on_test_set()

            # self.scheduler.step()

        # fig, axs = plt.subplots(2)
        # axs[0].plot(range(len(lrs)), lrs)
        # # l1, _ = axs[1].plot(range(len(losses)), np.log(losses)) #, range(len(losses)), np.log(val_losses))
        # axs[1].plot(range(len(losses)), np.log(losses))
        # axs[1].legend(['train loss']) # , 'val loss'])
        # plt.show()

        return mean_loss

    def evaluate_on_validation_set(self):
        self.model.eval()
        validation_ldr = self.dm.get_validation_set()
        val_loss = []
        with torch.no_grad():
            for i, X_i in enumerate(validation_ldr, 0):
                val_inputs = X_i[0].to(self.device).float()
                scores = self.model(val_inputs)
                loss = scores.mean().item()
                val_loss.append(loss)

                # switch back to train mode
        self.model.train()
        return np.mean(val_loss)

    def train_iter(self, X):
        scores = self.model(X)
        loss = scores.mean()

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
        energy_threshold = kwargs.get('energy_threshold', 80)
        # Change the model to evaluation mode
        self.model.eval()
        # train_score = []

        with torch.no_grad():
            # Create pytorch's train data_loader

            # Calculate score using estimated parameters
            test_score = []
            test_labels = []
            test_z = []

            for data in test_loader:
                test_inputs, label_inputs = data[0].float().to(self.device), data[1]

                # forward pass
                # code, X_prime = self.model(test_inputs)
                # forward pass
                scores = self.model(test_inputs)
                test_score.append(scores.cpu().numpy())
                # test_z.append(code.cpu().numpy())
                test_labels.append(label_inputs.numpy())

            test_score = np.concatenate(test_score, axis=0)
            # test_z = np.concatenate(test_z, axis=0)
            test_labels = np.concatenate(test_labels, axis=0)

            combined_score = test_score # np.concatenate([train_score, test_score], axis=0)

            comp_threshold = 100 * sum(test_labels == 0)/len(test_labels)

            res_max = score_recall_precision(combined_score, test_score, test_labels)
            res = score_recall_precision_w_thresold(combined_score, test_score, test_labels, pos_label=pos_label,
                                                    threshold=comp_threshold)

            # switch back to train mode
            self.model.train()

            res = dict(res, **res_max)
            # print(res)
            return res, test_z, test_labels, combined_score
