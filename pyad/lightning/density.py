from typing import List
from pytorch_lightning.utilities.cli import MODEL_REGISTRY

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from pyad.loss.TripletCenterLoss import TripletCenterLoss
from pyad.lightning.base import BaseLightningModel
from ray import tune

score_metrics_opts = {"reconstruction", "energy"}


@MODEL_REGISTRY
class LitDSEBM(BaseLightningModel):

    def __init__(
            self,
            fc_1_out: int,
            fc_2_out: int,
            batch_size: int,
            score_metric="reconstruction",
            b_prime: torch.Tensor = None,
            **kwargs
    ):
        super(LitDSEBM, self).__init__(**kwargs)
        self.save_hyperparameters(
            ignore=["in_features", "n_instances", "threshold", "batch_size", "b_prime"]
        )
        # energy or reconstruction-based anomaly score function
        assert score_metric in score_metrics_opts, "unknown `score_metric` %s, please select %s" % (score_metric, score_metrics_opts)
        self.score_metric = score_metric
        # loss function
        self.criterion = nn.BCEWithLogitsLoss()
        self.batch_size = batch_size
        self.b_prime = b_prime or torch.nn.Parameter(
            torch.Tensor(self.batch_size, self.in_features).to(self.device)
        )
        torch.nn.init.xavier_normal_(self.b_prime)

        self._build_network()

    def _build_network(self):
        # TODO: Make model more flexible. Users should be able to set the number of layers
        self.fc_1 = nn.Linear(self.in_features, self.hparams.fc_1_out)
        self.fc_2 = nn.Linear(self.hparams.fc_1_out, self.hparams.fc_2_out)
        self.softp = nn.Softplus()
        self.bias_inv_1 = torch.nn.Parameter(torch.Tensor(self.hparams.fc_1_out))
        self.bias_inv_2 = torch.nn.Parameter(torch.Tensor(self.in_features))
        torch.nn.init.xavier_normal_(self.fc_2.weight)
        torch.nn.init.xavier_normal_(self.fc_1.weight)
        self.fc_1.bias.data.zero_()
        self.fc_2.bias.data.zero_()
        self.bias_inv_1.data.zero_()
        self.bias_inv_2.data.zero_()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            betas=(0.5, 0.999)
        )
        return optimizer

    def random_noise_like(self, X: torch.Tensor):
        return torch.normal(mean=0., std=1., size=X.shape).float()

    def forward(self, X: torch.Tensor):
        output = self.softp(self.fc_1(X))
        output = self.softp(self.fc_2(output))

        # inverse layer
        output = self.softp((output @ self.fc_2.weight) + self.bias_inv_1)
        output = self.softp((output @ self.fc_1.weight) + self.bias_inv_2)

        return output

    def energy(self, X, X_hat):
        diff = self.b_prime.shape[0] - X.shape[0]
        if diff > 0:
            energy = 0.5 * torch.sum(torch.square(X - self.b_prime[:X.shape[0]])) - torch.sum(X_hat)
        else:
            energy = 0.5 * torch.sum(torch.square(X - self.b_prime)) - torch.sum(X_hat)

        return energy

    def compute_loss(self, X, fx_noise):
        out = torch.square(X - fx_noise)
        out = torch.sum(out, dim=-1)
        out = torch.mean(out)
        return out

    def score(self, X: torch.Tensor, y: torch.Tensor = None):
        # TODO: add support for multi-score
        if self.score_metric == "energy":
            # Evaluation of the score based on the energy
            with torch.no_grad():
                diff = self.b_prime.shape[0] - X.shape[0]
                if diff > 0:
                    flat = X - self.b_prime[:X.shape[0]]
                else:
                    flat = X - self.b_prime
                out = self.forward(X)
                energies = 0.5 * torch.sum(torch.square(flat), dim=1) - torch.sum(out, dim=1)
                scores = energies
        else:
            # Evaluation of the score based on the reconstruction error
            X.requires_grad_()
            out = self.forward(X)
            energy = self.energy(X, out)
            dEn_dX = torch.autograd.grad(energy, X)[0]
            rec_errs = torch.linalg.norm(dEn_dX, 2, keepdim=False, dim=1)
            scores = rec_errs
        return scores #energies.cpu().numpy(), rec_errs.cpu().numpy()

    def training_step(self, batch, batch_idx):
        X, _, _ = batch
        X = X.float()
        # add noise to input data
        noise = self.random_noise_like(X).to(self.device)
        X_noise = X + noise
        X.requires_grad_()
        X_noise.requires_grad_()
        # forward pass on noisy input
        out_noise = self.forward(X_noise)
        energy_noise = self.energy(X_noise, out_noise)
        # compute gradient
        dEn_dX = torch.autograd.grad(energy_noise, X_noise, retain_graph=True, create_graph=True)
        fx_noise = (X_noise - dEn_dX[0])

        return self.compute_loss(X, fx_noise)

    def on_test_model_eval(self) -> None:
        torch.set_grad_enabled(True)
