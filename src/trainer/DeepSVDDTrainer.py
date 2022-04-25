from torch.utils.data.dataloader import DataLoader
import torch
import numpy as np
from tqdm import trange
from utils import precision_recall_f1_roc_pr


class DeepSVDDTrainer:

    def __init__(self, model, dm, optimizer_factory, device, R=None, c=None,
                 lr: float = 1e-4, n_epochs: int = 100, batch_size: int = 128, n_jobs_dataloader: int = 0):
        self.device = device
        self.model = model.to(device)
        self.batch_size = batch_size
        self.n_jobs_dataloader = n_jobs_dataloader
        self.n_epochs = n_epochs
        self.lr = lr
        self.c = c
        self.R = R
        self.dm = dm
        self.optim = optimizer_factory(self.model.parameters(), lr=self.lr)

    def train(self, n_epochs: int):
        self.model.train()
        train_ldr = self.dm.get_train_set()

        # Initialize hypersphere center c (if c not loaded)
        if self.c is None:
            print("Initializing center c...")
            self.c = self.init_center_c(train_ldr)
            print("Center c initialized.")

        print('Started training')
        epoch_loss = 0.0
        for epoch in range(n_epochs):
            print(f"\nEpoch: {epoch + 1} of {n_epochs}")
            with trange(len(train_ldr)) as t:
                for sample in train_ldr:
                    X, _ = sample
                    X = X.to(self.device).float()

                    # Reset gradient
                    self.optim.zero_grad()

                    outputs = self.model(X)
                    dist = torch.sum((outputs - self.c) ** 2, dim=1)
                    loss = torch.mean(dist)

                    # Backpropagation
                    loss.backward()
                    self.optim.step()

                    epoch_loss += loss
                t.set_postfix(
                    loss='{:05.3f}'.format(epoch_loss),
                )
                epoch_loss = 0.0
                t.update()
        print("Finished training")

    def evaluate_on_test_set(self, threshold: int):
        y_train_true, train_scores = self.test(self.dm.get_train_set)
        y_test_true, test_scores = self.test(self.dm.get_test_set)
        y_true = np.concatenate((y_train_true, y_test_true), axis=0)
        scores = np.concatenate((train_scores, test_scores), axis=0)
        return precision_recall_f1_roc_pr(y_true, scores, threshold=threshold)

    def test(self, ldr: DataLoader) -> (np.array, np.array):
        self.model.eval()
        y_true, scores = [], []
        with torch.no_grad():
            for row in ldr:
                X, y = row
                X = X.to(self.device).float()
                outputs = self.model(X)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                score = dist
                y_true.extend(y.cpu().tolist())
                scores.extend(score.cpu().tolist())

        return np.array(y_true), np.array(scores)

    def init_center_c(self, train_loader: DataLoader, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data.
           Code taken from https://github.com/lukasruff/Deep-SVDD-PyTorch/blob/master/src/optim/deepSVDD_trainer.py"""
        n_samples = 0
        c = torch.zeros(self.model.rep_dim, device=self.device)

        self.model.eval()
        with torch.no_grad():
            for sample in train_loader:
                # get the inputs of the batch
                X, _ = sample
                X = X.to(self.device).float()
                outputs = self.model(X)
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)

        c /= n_samples

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c

    def get_params(self) -> dict:
        return {'c': self.c, 'R': self.R, **self.model.get_params()}


def get_radius(dist: torch.Tensor, nu: float):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)
