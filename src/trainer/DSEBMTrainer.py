import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from torch import optim
from torch.nn import Parameter
from tqdm import trange

from utils import score_recall_precision_w_thresold, score_recall_precision


class DSEBMTrainer:
    def __init__(self, model: nn.Module, dm, device, batch, dim, learning_rate, optimizer_factory=None):
        assert optimizer_factory is None
        self.model = model
        self.device = device
        self.dm = dm
        self.criterion = nn.BCEWithLogitsLoss()
        self.lr = learning_rate
        self.batch = batch
        self.b_prime = Parameter(torch.Tensor(batch, dim).to(self.device))
        torch.nn.init.xavier_normal_(self.b_prime)
        self.optim = optim.Adam(
            list(self.model.parameters()) + [self.b_prime],
            lr=self.lr, betas=(0.5, 0.999)
        )

    def train(self, n_epochs):
        train_ldr = self.dm.get_train_set()
        for epoch in range(n_epochs):
            print(f"\nEpoch: {epoch + 1} of {n_epochs}")
            losses = 0
            energies = 0
            with trange(len(train_ldr)) as t:
                for i, data in enumerate(train_ldr, 0):

                    X = data[0].float().to(self.device)
                    if len(X) < self.batch:
                        break
                    noise = self.model.random_noise_like(X).to(self.device)

                    X_noise = X + noise
                    X.requires_grad_()
                    X_noise.requires_grad_()

                    out = self.model(X)
                    out_noise = self.model(X_noise)

                    energy = self.energy(X, out)
                    energy_noise = self.energy(X_noise, out_noise)

                    dEn_dX = torch.autograd.grad(energy_noise, X_noise, retain_graph=True, create_graph=True)
                    fx_noise = (X_noise - dEn_dX[0])
                    loss = self.loss(X, fx_noise)

                    self.optim.zero_grad()
                    # Use autograd to compute the backward pass.
                    loss.backward()

                    # updates the weights using gradient descent
                    self.optim.step()

                    losses += loss.item()
                    energies += energy.item()
                    losses /= (i + 1)
                    energies /= (i + 1)
                    t.set_postfix(
                        l='{:05.4f}'.format(loss),
                        e='{:05.4f}'.format(energy),

                    )
                    t.update()

    def evaluate_on_test_set(self, pos_label=1, **kwargs):
        labels, scores_r, scores_e = [], [], []
        test_ldr = self.dm.get_test_set()
        energy_threshold = kwargs.get('energy_threshold', 80)

        self.model.eval()

        scores_e_train = []
        scores_r_train = []

        # Create pytorch's train data_loader
        train_loader = self.dm.get_init_train_loader()
        for X_i, _, _ in train_loader:
            # transfer tensors to selected device
            X = X_i.float().to(self.device)
            if len(X) < self.batch:
                break

            # Evaluation of the score based on the energy
            with torch.no_grad():
                flat = X - self.b_prime
                out = self.model(X)
                energies = 0.5 * torch.sum(torch.square(flat), dim=1) - torch.sum(out, dim=1)
            scores_e_train.append(energies.cpu().numpy())

            # Evaluation of the score based on the reconstruction error
            X.requires_grad_()
            out = self.model(X)
            energy = self.energy(X, out)
            dEn_dX = torch.autograd.grad(energy, X)[0]
            # fx = X - dEn_dX
            # delta = X - fx
            rec_errs = torch.linalg.norm(dEn_dX, 2, keepdim=False, dim=1)
            scores_r_train.append(rec_errs.cpu().numpy())

        scores_e_train = np.concatenate(scores_e_train, axis=0)
        scores_r_train = np.concatenate(scores_r_train, axis=0)

        # Calculate score using estimated parameters on test set
        for X_i, label in test_ldr:
            # transfer tensors to selected device
            X = X_i.float().to(self.device)
            if len(X) < self.batch:
                break

            # Evaluation of the score based on the energy
            with torch.no_grad():
                flat = X - self.b_prime
                out = self.model(X)
                energies = 0.5 * torch.sum(torch.square(flat), dim=1) - torch.sum(out, dim=1)
            scores_e.append(energies.cpu().numpy())

            # Evaluation of the score based on the reconstruction error
            X.requires_grad_()
            out = self.model(X)
            energy = self.energy(X, out)
            dEn_dX = torch.autograd.grad(energy, X)[0]
            # fx = X - dEn_dX

            rec_errs = torch.linalg.norm(dEn_dX, 2, keepdim=False, dim=1)
            scores_r.append(rec_errs.cpu().numpy())
            labels.append(label.numpy())

        scores_r = np.concatenate(scores_r, axis=0)
        scores_e = np.concatenate(scores_e, axis=0)
        labels = np.concatenate(labels, axis=0)

        combined_scores_e = np.concatenate([scores_e_train, scores_e], axis=0)
        combined_scores_r = np.concatenate([scores_r_train, scores_r], axis=0)

        # Result based on energy
        res1 = score_recall_precision_w_thresold(combined_scores_e, scores_e, labels, pos_label=pos_label,
                                                 threshold=energy_threshold)
        res1 = self.ren_dict_keys(res1, 'en')
        print('Evaluation with energy\n')
        score_recall_precision(combined_scores_e, scores_e, labels)

        # Result based on reconstruction
        res2 = score_recall_precision_w_thresold(combined_scores_r, scores_r, labels, pos_label=pos_label,
                                                 threshold=energy_threshold)
        res2 = self.ren_dict_keys(res2, 'rec')
        print('\n\nEvaluation with reconstruction')
        score_recall_precision(combined_scores_r, scores_r, labels)

        res = dict(res1, **res2)
        # switch back to train mode
        self.model.train()

        return res, _, _, _

    def ren_dict_keys(self, d: dict, prefix=''):
        d_ = {}
        for k in d.keys():
            d_[f"{prefix}_{k}"] = d[k]

        return d_

    def loss(self, X, fx_noise):
        out = torch.square(X - fx_noise)
        out = torch.sum(out, dim=-1)
        out = torch.mean(out)
        return out

    def energy(self, X, X_hat):
        return 0.5 * torch.sum(torch.square(X - self.b_prime)) - torch.sum(X_hat)
