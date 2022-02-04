import torch
import torch.nn as nn
import numpy as np
from tqdm import trange
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from typing import Union
from .BaseTrainer import BaseTrainer

torch.autograd.set_detect_anomaly(True)


class ALADTrainer(BaseTrainer):
    def __init__(self, **kwargs):
        super(ALADTrainer, self).__init__(**kwargs)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optim_ge, self.optim_d = None, None
        self.set_optimizer()

    def train_iter(self, sample: torch.Tensor):
        pass

    def score(self, sample: torch.Tensor):
        _, feature_real = self.model.D_xx(sample, sample)
        _, feature_gen = self.model.D_xx(sample, self.model.G(self.model.E(sample)))
        return torch.linalg.norm(feature_real - feature_gen, 2, keepdim=False, dim=1)

    def set_optimizer(self):
        self.optim_ge = optim.Adam(
            list(self.model.G.parameters()) + list(self.model.E.parameters()),
            lr=self.lr, betas=(0.5, 0.999)
        )
        self.optim_d = optim.Adam(
            list(self.model.D_xz.parameters()) + list(self.model.D_zz.parameters()) + list(
                self.model.D_xx.parameters()),
            lr=self.lr, betas=(0.5, 0.999)
        )

    def test(self, dataset: DataLoader) -> Union[np.array, np.array]:
        self.model.eval()
        y_true, scores = [], []
        with torch.no_grad():
            for row in dataset:
                X, y = row
                X = X.to(self.device).float()

                score = self.score(X)

                y_true.extend(y.cpu().tolist())
                scores.extend(score.cpu().tolist())

        return np.array(y_true), np.array(scores)

    def train_iter_dis(self, X):
        # Labels
        y_true = Variable(torch.zeros(X.size(0), 1)).to(self.device)
        y_fake = Variable(torch.ones(X.size(0), 1)).to(self.device)
        # Forward pass
        out_truexz, out_fakexz, out_truezz, out_fakezz, out_truexx, out_fakexx = self.model(X)
        # Compute loss
        # Discriminators Losses
        loss_dxz = self.criterion(out_truexz, y_true) + self.criterion(out_fakexz, y_fake)
        loss_dzz = self.criterion(out_truezz, y_true) + self.criterion(out_fakezz, y_fake)
        loss_dxx = self.criterion(out_truexx, y_true) + self.criterion(out_fakexx, y_fake)
        loss_d = loss_dxz + loss_dzz + loss_dxx

        return loss_d

    def train_iter_gen(self, X):
        # Labels
        y_true = Variable(torch.zeros(X.size(0), 1)).to(self.device)
        y_fake = Variable(torch.ones(X.size(0), 1)).to(self.device)
        # Forward pass
        out_truexz, out_fakexz, out_truezz, out_fakezz, out_truexx, out_fakexx = self.model(X)
        # Generator losses
        loss_gexz = self.criterion(out_fakexz, y_true) + self.criterion(out_truexz, y_fake)
        loss_gezz = self.criterion(out_fakezz, y_true) + self.criterion(out_truezz, y_fake)
        loss_gexx = self.criterion(out_fakexx, y_true) + self.criterion(out_truexx, y_fake)
        cycle_consistency = loss_gexx + loss_gezz
        loss_ge = loss_gexz + cycle_consistency

        return loss_ge

    def train(self, dataset: DataLoader, nep = None):
        self.model.train()

        for epoch in range(self.n_epochs):
            ge_losses, d_losses = 0, 0
            with trange(len(dataset)) as t:
                for sample in dataset:
                    X, _ = sample
                    X_dis, X_gen = X.to(self.device).float(), X.clone().to(self.device).float()
                    # Cleaning gradients
                    self.optim_ge.zero_grad()
                    self.optim_d.zero_grad()
                    # Forward pass
                    loss_d = self.train_iter_dis(X_dis)
                    loss_ge = self.train_iter_gen(X_gen)
                    # Backward pass
                    loss_d.backward()
                    loss_ge.backward()
                    self.optim_d.step()
                    self.optim_ge.step()
                    # Journaling
                    d_losses += loss_d.item()
                    ge_losses += loss_ge.item()
                    t.set_postfix(
                        loss_d='{:05.4f}'.format(loss_d),
                        loss_ge='{:05.4f}'.format(loss_ge),
                    )
                    t.update()
