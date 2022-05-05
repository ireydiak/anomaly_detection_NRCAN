import os
from collections import defaultdict

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import trange
from src.trainer.reconstruction import DAGMMTrainer, MemAETrainer
from src.trainer.one_class import DeepSVDDTrainer
import matplotlib.pyplot as plt

from src.utils import metrics


def plot_metrics(path, precision, recall, figname='fig1.png'):
    """
    Function that plots train and validation losses and accuracies after
    training phase
    """
    epochs = range(1, len(precision) + 1)

    f, ax1 = plt.subplots(figsize=(10, 5))

    ax1.plot(
        epochs, precision, '-o', label='Test precision', c="blue"
    )
    ax1.plot(
        epochs, recall, '-o', label='Test recall', c="orange"
    )
    ax1.set_title('Test recall and precision')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Metrics')
    ax1.legend()

    f.savefig(os.path.join(path, figname))
    plt.show()


class MemAEIDSTrainer(MemAETrainer):
    def __init__(
            self,
            train_ldr,
            test_ldr,
            ckpt_fname: str = None,
            run_test_validation=False,
            keep_ckpt=True,
            **kwargs
    ):
        super(MemAEIDSTrainer, self).__init__(**kwargs)
        self.train_ldr = train_ldr
        self.test_ldr = test_ldr
        self.metric_values = {"test_precision": [], "test_recall": []}
        self.ckpt_fname = ckpt_fname or self.model.name.lower()
        self.ckpt_file = None
        self.run_test_validation = run_test_validation
        self.keep_ckpt = keep_ckpt

    def train(self, dataset: DataLoader):
        self.model.train()

        print("Started training")
        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            with trange(len(self.train_ldr)) as t:
                for sample in self.train_ldr:
                    X, _, _ = sample
                    X = X.to(self.device).float()

                    # Reset gradient
                    self.optimizer.zero_grad()
                    loss = self.train_iter(X)

                    # Backpropagation
                    loss.backward()
                    self.optimizer.step()

                    epoch_loss += loss.item()
                    t.set_postfix(
                        loss='{:05.3f}'.format(epoch_loss / (epoch + 1)),
                        epoch=epoch + 1
                    )
                    t.update()
            if self.keep_ckpt and epoch % 5 == 0:
                self.save_ckpt(self.ckpt_fname + "_epoch={}.pt".format(epoch+1))

            if self.run_test_validation and (epoch % 5 == 0 or epoch == 0):
                y_true, scores, _ = self.test(self.test_ldr)
                test_res = metrics.estimate_optimal_threshold(scores, y_true)
                self.metric_values["test_precision"].append(test_res["Precision"])
                self.metric_values["test_recall"].append(test_res["Recall"])

    def inspect_gradient_wrt_input(self, all_labels):
        self.model.eval()
        y_true, scores, labels = [], [], []
        y_grad_wrt_X, label_grad_wrt_X, = {0: [], 1: []}, {label: [] for label in all_labels}
        losses = []
        for row in self.test_ldr:
            X, y, label = row
            # TODO: put in dataloader
            label = np.array(label)
            X = X.to(self.device).float()
            X.requires_grad = True
            self.optimizer.zero_grad()

            loss = self.train_iter(X)
            loss.backward()

            for y_c in [0, 1]:
                dsdx = X.grad[y == y_c].mean(dim=0).cpu().numpy()
                if len(X.grad[y == y_c]) > 0:
                    y_grad_wrt_X[y_c].append(dsdx)
            for y_c in all_labels:
                dsdx = X.grad[label == y_c].cpu().numpy()
                if len(X.grad[label == y_c]) > 0:
                    label_grad_wrt_X[y_c].append(dsdx)
            losses.append(loss.item())
            #score = self.score(X)

            y_true.extend(y.cpu().tolist())
            #scores.extend(score.cpu().tolist())
            labels.extend(list(label))
        self.model.train()
        y_grad_wrt_X[0], y_grad_wrt_X[1] = np.asarray(y_grad_wrt_X[0]), np.asarray(y_grad_wrt_X[1])
        for y_c in all_labels:
            label_grad_wrt_X[y_c] = np.concatenate(label_grad_wrt_X[y_c])
        return {
          "y_true": np.array(y_true),
          # "scores": np.array(scores),
          "labels":  np.array(labels),
          "y_grad_wrt_X": y_grad_wrt_X,
          "label_grad_wrt_X": label_grad_wrt_X,
          "losses": np.asarray(losses),
        }

    def test(self, dataset: DataLoader):
        self.model.eval()
        y_true, scores, labels = [], [], []
        with torch.no_grad():
            for row in dataset:
                X, y, label = row
                X = X.to(self.device).float()
                score = self.score(X)
                y_true.extend(y.cpu().tolist())
                scores.extend(score.cpu().tolist())
                labels.extend(list(label))
        self.model.train()
        return np.array(y_true), np.array(scores), np.array(labels)

    def save_ckpt(self, fname: str):
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metric_values": self.metric_values,
            "alpha": self.model.alpha,
            "mem_dim": self.model.mem_dim,
            "shrink_thres": self.model.shrink_thres
        }, fname)


class DeepSVDDIDSTrainer(DeepSVDDTrainer):

    def __init__(self, train_ldr, test_ldr, ckpt_fname: str = None, run_test_validation=False, keep_ckpt=True, **kwargs):
        super(DeepSVDDIDSTrainer, self).__init__(**kwargs)
        self.train_ldr = train_ldr
        self.test_ldr = test_ldr
        self.metric_values = {"test_precision": [], "test_recall": []}
        self.ckpt_fname = ckpt_fname or self.model.name.lower()
        self.ckpt_file = None
        self.run_test_validation = run_test_validation
        self.keep_ckpt = keep_ckpt

    def save_ckpt(self, fname: str):
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "c": self.c,
            "R": self.R,
            "metric_values": self.metric_values
        }, fname)

    def init_center_c(self, train_loader, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data.
           Code taken from https://github.com/lukasruff/Deep-SVDD-PyTorch/blob/master/src/optim/deepSVDD_trainer.py"""
        n_samples = 0
        c = torch.zeros(self.model.rep_dim, device=self.device)

        self.model.eval()
        with torch.no_grad():
            for sample in train_loader:
                # get the inputs of the batch
                X, _, _ = sample
                X = X.to(self.device).float()
                outputs = self.model(X)
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)

        if c.isnan().sum() > 0:
            raise Exception("NaN value encountered during init_center_c")

        if torch.allclose(c, torch.zeros_like(c)):
            raise Exception("Center c initialized at 0")

        c /= n_samples

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c

    def test(self, dataset: DataLoader):
        self.model.eval()
        y_true, scores, labels = [], [], []
        with torch.no_grad():
            for row in dataset:
                X, y, label = row
                X = X.to(self.device).float()
                score = self.score(X)
                y_true.extend(y.cpu().tolist())
                scores.extend(score.cpu().tolist())
                labels.extend(list(label))
        self.model.train()
        return np.array(y_true), np.array(scores), np.array(labels)

    def inspect_gradient_wrt_input(self, all_labels):
        self.model.eval()
        y_true, scores, labels = [], [], []
        y_grad_wrt_X, label_grad_wrt_X, = {0: [], 1: []}, {label: [] for label in all_labels}
        losses = []
        for row in self.test_ldr:
            X, y, label = row
            # TODO: put in dataloader
            label = np.array(label)
            X = X.to(self.device).float()
            X.requires_grad = True
            self.optimizer.zero_grad()

            loss = self.train_iter(X)
            loss.backward()

            for y_c in [0, 1]:
                dsdx = X.grad[y == y_c].mean(dim=0).cpu().numpy()
                if len(X.grad[y == y_c]) > 0:
                    y_grad_wrt_X[y_c].append(dsdx)
            for y_c in all_labels:
                dsdx = X.grad[label == y_c].cpu().numpy()
                if len(X.grad[label == y_c]) > 0:
                    label_grad_wrt_X[y_c].append(dsdx)
            losses.append(loss.item())
            #score = self.score(X)

            y_true.extend(y.cpu().tolist())
            #scores.extend(score.cpu().tolist())
            labels.extend(list(label))
        self.model.train()
        y_grad_wrt_X[0], y_grad_wrt_X[1] = np.asarray(y_grad_wrt_X[0]), np.asarray(y_grad_wrt_X[1])
        for y_c in all_labels:
            label_grad_wrt_X[y_c] = np.concatenate(label_grad_wrt_X[y_c])
        return {
          "y_true": np.array(y_true),
          # "scores": np.array(scores),
          "labels":  np.array(labels),
          "y_grad_wrt_X": y_grad_wrt_X,
          "label_grad_wrt_X": label_grad_wrt_X,
          "losses": np.asarray(losses),
        }

    def train(self, dataset: DataLoader):
        self.model.train()

        self.c = self.init_center_c(self.train_ldr)

        print("Started training")
        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            with trange(len(self.train_ldr)) as t:
                for sample in self.train_ldr:
                    X, _, _ = sample
                    X = X.to(self.device).float()

                    # Reset gradient
                    self.optimizer.zero_grad()
                    loss = self.train_iter(X)

                    # Backpropagation
                    loss.backward()
                    self.optimizer.step()

                    epoch_loss += loss.item()
                    t.set_postfix(
                        loss='{:05.3f}'.format(epoch_loss / (epoch + 1)),
                        epoch=epoch + 1
                    )
                    t.update()
            if self.keep_ckpt and epoch % 5 == 0:
                self.save_ckpt(self.ckpt_fname + "_epoch={}.pt".format(epoch+1))

            if self.run_test_validation and (epoch % 5 == 0 or epoch == 0):
                y_true, scores, _ = self.test(self.test_ldr)
                test_res = metrics.estimate_optimal_threshold(scores, y_true)
                self.metric_values["test_precision"].append(test_res["Precision"])
                self.metric_values["test_recall"].append(test_res["Recall"])


class DAGMMIDSTrainer(DAGMMTrainer):

    def __init__(self, train_ldr, test_ldr, ckpt_fname=None, run_test_validation=False, keep_ckpt=True, **kwargs):
        super(DAGMMIDSTrainer, self).__init__(**kwargs)
        self.train_ldr = train_ldr
        self.test_ldr = test_ldr
        self.metric_values = {"test_precision": [], "test_recall": []}
        self.ckpt_fname = ckpt_fname or self.model.name.lower()
        self.ckpt_file = None
        self.cur_epoch = None
        self.run_test_validation = run_test_validation
        self.keep_ckpt = keep_ckpt

    def test(self, dataset: DataLoader):
        """
        function that evaluate the model on the test set every iteration of the
        active learning process
        """
        self.model.eval()

        with torch.no_grad():
            scores, y_true, labels = [], [], []
            for row in dataset:
                X, y, label = row
                X = X.to(self.device).float()
                # forward pass
                code, x_prime, cosim, z, gamma = self.model(X)
                sample_energy, pen_cov_mat = self.estimate_sample_energy(
                    z, self.phi, self.mu, self.cov_mat, average_energy=False
                )
                y_true.extend(y)
                scores.extend(sample_energy.cpu().numpy())
                labels.extend(label)
        self.model.train()
        return np.array(y_true), np.array(scores), np.array(labels)

    def train(self, dataset: DataLoader):
        self.model.train()

        print("Started training")
        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            with trange(len(self.train_ldr)) as t:
                for sample in self.train_ldr:
                    X, _, _ = sample
                    X = X.to(self.device).float()

                    # Reset gradient
                    self.optimizer.zero_grad()

                    loss = self.train_iter(X)

                    # Backpropagation
                    loss.backward()
                    self.optimizer.step()

                    epoch_loss += loss.item()
                    t.set_postfix(
                        loss='{:05.3f}'.format(epoch_loss / (epoch + 1)),
                        epoch=epoch + 1
                    )
                    t.update()
            if self.keep_ckpt and epoch % 5 == 0:
                self.save_ckpt(self.ckpt_fname + "_epoch={}.pt".format(epoch + 1))

            if self.run_test_validation and (epoch % 5 == 0 or epoch == 0):
                y_true, scores, _ = self.test(self.test_ldr)
                test_res = metrics.estimate_optimal_threshold(scores, y_true)
                self.metric_values["test_precision"].append(test_res["Precision"])
                self.metric_values["test_recall"].append(test_res["Recall"])
