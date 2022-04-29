import torch
import numpy as np
from torch import optim
from torch.utils.data import DataLoader
from tqdm import trange
from src.trainer.reconstruction import DAGMMTrainer
from src.trainer.one_class import DeepSVDDTrainer


class DAGMMIDSTrainer(DAGMMTrainer):

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
        return np.array(y_true), np.array(scores), np.array(labels)

    def train(self, dataset: DataLoader):
        self.model.train()

        print("Started training")
        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            with trange(len(dataset)) as t:
                for sample in dataset:
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
                        loss='{:05.3f}'.format(epoch_loss/(epoch + 1)),
                        epoch=epoch + 1
                    )
                    t.update()


class DeepSVDDIDSTrainer(DeepSVDDTrainer):

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

        return np.array(y_true), np.array(scores), np.array(labels)

    def train(self,  dataset: DataLoader):
        self.model.train()

        self.c = self.init_center_c(dataset)

        print("Started training")
        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            with trange(len(dataset)) as t:
                for sample in dataset:
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
