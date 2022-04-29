import os

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import trange
from src.trainer.reconstruction import DAGMMTrainer
from src.trainer.one_class import DeepSVDDTrainer
import matplotlib.pyplot as plt


class DeepSVDDIDSTrainer(DeepSVDDTrainer):

    def __init__(self, train_ldr, test_ldr, ckpt_fname: str = None, **kwargs):
        super(DeepSVDDIDSTrainer, self).__init__(**kwargs)
        self.train_ldr = train_ldr
        self.test_ldr = test_ldr
        self.metric_values = {"test_precision": [], "test_recall": []}
        self.ckpt_fname = ckpt_fname or self.model.name.lower()
        self.ckpt_file = None

    def save_ckpt(self, fname: str):
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "c": self.c,
            "R": self.R,
            "metric_values": self.metric_values
        }, fname)

    def plot_metrics(self, path, fname="fig1.png"):
        """
        Function that plots train and validation losses and accuracies after
        training phase
        """
        epochs = range(1, len(self.metric_values['test_precision']) + 1)

        f = plt.figure(figsize=(10, 5))
        ax1 = f.add_subplot(121)
        ax2 = f.add_subplot(122)

        # loss plot
        ax1.plot(
            epochs, self.metric_values['test_precision'], '-o', label='Test precision', c="blue"
        )
        ax1.plot(
            epochs, self.metric_values['test_recall'], '-o', label='Test recall', c="orange"
        )
        ax1.set_title('Test recall and precision')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Metrics')
        ax1.legend()

        # accuracy plot
        # ax2.plot(
        #     epochs, self.metric_values['train_acc'], '-o',
        #     label='Training accuracy')
        # ax2.plot(
        #     epochs, self.metric_values['val_acc'], '-o',
        #     label='Validation accuracy')
        # ax2.set_title('Training and validation accuracy')
        # ax2.set_xlabel('Epochs')
        # ax2.set_ylabel('accuracy')
        # ax2.legend()

        f.savefig(os.path.join(path, fname))
        plt.show()

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
            if epoch % 5 == 0: # or epoch == 0:
                # y_true, scores, _ = self.test(self.test_ldr)
                # test_res = metrics.estimate_optimal_threshold(scores, y_true)
                # self.metric_values["test_precision"].append(test_res["Precision"])
                # self.metric_values["test_recall"].append(test_res["Recall"])
                self.save_ckpt(self.ckpt_fname + "_epoch={}.pt".format(epoch+1))


class DAGMMIDSTrainer(DAGMMTrainer):

    def __init__(self, train_ldr, test_ldr, ckpt_fname=None, **kwargs):
        super(DAGMMIDSTrainer, self).__init__(**kwargs)
        self.train_ldr = train_ldr
        self.test_ldr = test_ldr
        self.metric_values = {"test_precision": [], "test_recall": []}
        self.ckpt_fname = ckpt_fname or self.model.name.lower()
        self.ckpt_file = None
        self.cur_epoch = None

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

    def plot_metrics(self, path):
        """
        Function that plots train and validation losses and accuracies after
        training phase
        """
        epochs = range(1, len(self.metric_values['test_precision']) + 1)

        f, ax1 = plt.subplots(figsize=(10, 5))
        #f = plt.figure(figsize=(10, 5))
        #ax1 = f.add_subplot(121)
        #ax2 = f.add_subplot(122)

        # loss plot
        ax1.plot(
            epochs, self.metric_values['test_precision'], '-o', label='Test precision', c="blue"
        )
        ax1.plot(
            epochs, self.metric_values['test_recall'], '-o', label='Test recall', c="orange"
        )
        ax1.set_title('Test recall and precision')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Metrics')
        ax1.legend()

        # accuracy plot
        # ax2.plot(
        #     epochs, self.metric_values['train_acc'], '-o',
        #     label='Training accuracy')
        # ax2.plot(
        #     epochs, self.metric_values['val_acc'], '-o',
        #     label='Validation accuracy')
        # ax2.set_title('Training and validation accuracy')
        # ax2.set_xlabel('Epochs')
        # ax2.set_ylabel('accuracy')
        # ax2.legend()

        f.savefig(os.path.join(path, 'fig1.png'))
        plt.show()

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
            if epoch % 5 == 0 or epoch == 0:
                # y_true, scores, _ = self.test(self.test_ldr)
                # test_res = metrics.estimate_optimal_threshold(scores, y_true)
                # self.metric_values["test_precision"].append(test_res["Precision"])
                # self.metric_values["test_recall"].append(test_res["Recall"])
                self.save_ckpt(self.ckpt_fname + "_epoch={}.pt".format(epoch + 1))