import numpy as np
import torch
import torch.nn as nn

from collections import defaultdict
from typing import Union
from torch import optim
from torch.nn import Parameter
from torch.utils.data import DataLoader
from sklearn import metrics
from .base import BaseTrainer


class DSEBMTrainer(BaseTrainer):
    def __init__(self, score_metric="reconstruction", **kwargs):
        assert score_metric == "reconstruction" or score_metric == "energy"
        super(DSEBMTrainer, self).__init__(**kwargs)
        self.score_metric = score_metric
        self.criterion = nn.BCEWithLogitsLoss()
        self.b_prime = Parameter(torch.Tensor(self.batch_size, self.model.in_features).to(self.device))
        torch.nn.init.xavier_normal_(self.b_prime)
        self.optim = optim.Adam(
            list(self.model.parameters()) + [self.b_prime],
            lr=self.lr, betas=(0.5, 0.999)
        )

    def train_iter(self, X):
        noise = self.model.random_noise_like(X).to(self.device)
        X_noise = X + noise
        X.requires_grad_()
        X_noise.requires_grad_()
        out_noise = self.model(X_noise)
        energy_noise = self.energy(X_noise, out_noise)
        dEn_dX = torch.autograd.grad(energy_noise, X_noise, retain_graph=True, create_graph=True)
        fx_noise = (X_noise - dEn_dX[0])
        return self.loss(X, fx_noise)

    def score(self, sample: torch.Tensor):
        # Evaluation of the score based on the energy
        with torch.no_grad():
            flat = sample - self.b_prime
            out = self.model(sample)
            energies = 0.5 * torch.sum(torch.square(flat), dim=1) - torch.sum(out, dim=1)

        # Evaluation of the score based on the reconstruction error
        sample.requires_grad_()
        out = self.model(sample)
        energy = self.energy(sample, out)
        dEn_dX = torch.autograd.grad(energy, sample)[0]
        rec_errs = torch.linalg.norm(dEn_dX, 2, keepdim=False, dim=1)
        return energies.cpu().numpy(), rec_errs.cpu().numpy()

    def test(self, dataset: DataLoader) -> Union[np.array, np.array]:
        self.model.eval()
        y_true, scores = [], []
        scores_e, scores_r = [], []
        for row in dataset:
            X, y = row[0], row[1]
            X = X.to(self.device).float()
            score_e, score_r = self.score(X)

            y_true.extend(y.cpu().tolist())
            scores_e.extend(score_e)
            scores_r.extend(score_r)

        scores = scores_r if self.score_metric == "reconstruction" else scores_e
        return np.array(y_true), np.array(scores)

    def evaluate(self, y_true: np.array, scores: np.array, threshold, pos_label: int = 1) -> dict:
        res = defaultdict()
        for score, name in zip(scores, ["score_e", "score_r"]):
            res[name] = {"Precision": -1, "Recall": -1, "F1-Score": -1, "AUROC": -1, "AUPR": -1}
            thresh = np.percentile(scores, threshold)
            y_pred = self.predict(scores, thresh)
            precision, recall, f1, _ = metrics.precision_recall_fscore_support(
                y_true, y_pred, average='binary', pos_label=pos_label
            )
            res[name]["Precision"], res[name]["Recall"], res[name]["F1-Score"] = precision, recall, f1
            res[name]["AUROC"] = metrics.roc_auc_score(y_true, scores)
            res[name]["AUPR"] = metrics.average_precision_score(y_true, scores)
        return res

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
