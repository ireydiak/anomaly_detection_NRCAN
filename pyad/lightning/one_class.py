from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning.utilities.cli import MODEL_REGISTRY
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from pyad.lightning.base import BaseLightningModel, layer_options_helper
from pyad.lightning.base import create_net_layers
from torch import nn
from typing import List
from ray import tune as ray_tune


@MODEL_REGISTRY
class LitDeepSVDD(BaseLightningModel):

    def __init__(
            self,
            feature_dim: int,
            hidden_dims: List[int],
            activation="relu",
            radius=None,
            center=None,
            eps: float = 0.1,
            **kwargs):
        super(LitDeepSVDD, self).__init__(**kwargs)
        self.save_hyperparameters(ignore=["radius", "center", "in_features", "n_instances", "threshold"])
        self.center = center
        self.radius = radius
        self._build_network()

    def _build_network(self):
        self.net = nn.Sequential(
            *create_net_layers(
                in_dim=self.in_features,
                out_dim=self.hparams.feature_dim,
                hidden_dims=self.hparams.hidden_dims,
                activation=self.hparams.activation
            )
        )

    def init_center_c(self, train_loader: DataLoader):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data.
           Code taken from https://github.com/lukasruff/Deep-SVDD-PyTorch/blob/master/src/optim/deepSVDD_trainer.py"""
        n_samples = 0
        c = torch.zeros(self.hparams.feature_dim, device=self.device)

        self.net.eval()
        self.eval()
        with torch.no_grad():
            for sample in train_loader:
                # get the inputs of the batch
                X, _, _ = sample
                X = X.float()
                outputs = self.net(X)
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)
        self.train(mode=True)
        self.net.train(mode=True)

        if c.isnan().sum() > 0:
            raise Exception("NaN value encountered during init_center_c")

        if torch.allclose(c, torch.zeros_like(c)):
            raise Exception("Center c initialized at 0")

        c /= n_samples

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < self.hparams.eps) & (c < 0)] = -self.hparams.eps
        c[(abs(c) < self.hparams.eps) & (c > 0)] = self.hparams.eps

        return c

    def before_train(self, dataloader: DataLoader):
        print("Initializing center ...")
        self.center = self.init_center_c(dataloader).to(self.device)

    def score(self, X: torch.Tensor, y: torch.Tensor = None):
        assert torch.allclose(self.center, torch.zeros_like(self.center)) is False, "center not initialized"
        outputs = self.net(X)
        if self.center.device != outputs.device:
            self.center = self.center.to(outputs.device)
        return torch.sum((outputs - self.center) ** 2, dim=1)

    def compute_loss(self, X: torch.Tensor):
        loss = self.score(X).mean()
        return loss

    def training_step(self, batch, batch_idx):
        X, _, _ = batch
        X = X.float()
        return self.compute_loss(X)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr
        )
        scheduler = StepLR(optimizer, step_size=20, gamma=0.9)
        return [optimizer], [scheduler]

    @staticmethod
    def get_ray_config(in_features: int, n_instances: int) -> dict:
        # read parent config
        parent_cfg = BaseLightningModel.get_ray_config(in_features, n_instances)

        hidden_dims_opts, _ = layer_options_helper(in_features)
        child_cfg = {
            "feature_dim": ray_tune.choice([32, 64, 256, 512]),
            "hidden_dims": ray_tune.choice(hidden_dims_opts),
            "activation": "relu",
        }
        return dict(
            **parent_cfg,
            **child_cfg
        )


@MODEL_REGISTRY
class LitDROCC(BaseLightningModel):
    def __init__(
            self,
            lamb: float = 1.,
            radius: float = None,
            gamma: float = 2.,
            n_classes: int = 1,
            n_hidden_nodes: int = 20,
            only_ce_epochs: int = 50,
            ascent_step_size: float = 0.01,
            ascent_num_steps: int = 50,
            **kwargs
    ):
        """
        Implements architecture presented in `DROCC: Deep Robust One-Class Classification` by Goyal et al. published
        in 2020 (https://arxiv.org/pdf/2002.12718.pdf). Most of the implementation is adapted directly from the original
        GitHub repository: https://github.com/microsoft/EdgeML

        Parameters
        ----------
        lamb: float
            weight for the adversarial loss
        radius: float
            radius of the hypersphere
        gamma: float
            used to fit the maximum volume of the hypersphere (gamma * radius)
        n_classes: int
            number of different classes (should always be one)
        n_hidden_nodes: int
        only_ce_epochs: int
            number of training epochs where only the binary cross-entropy loss is considered
        ascent_step_size: float
            step size during gradient ascent
        ascent_num_steps: int
            number of gradient ascent steps
        kwargs
        """
        super(LitDROCC, self).__init__(**kwargs)
        self.hparams.radius = radius or np.sqrt(self.in_features) / 2
        self._build_network()

    def _build_network(self):
        self.feature_extractor = nn.Sequential(
            OrderedDict([
                ('fc', nn.Linear(self.in_features, self.hparams.n_hidden_nodes)),
                ('relu1', torch.nn.ReLU(inplace=True))])
        )
        self.classifier = nn.Sequential(
            OrderedDict([
                ('fc1', nn.Linear(self.hparams.n_hidden_nodes, self.hparams.n_classes))
            ])
        )

    def forward(self, X: torch.Tensor):
        features = self.feature_extractor(X)
        logits = self.classifier(features.view(-1, self.hparams.n_hidden_nodes))
        return logits

    def score(self, X: torch.Tensor, y: torch.Tensor = None):
        logits = self.forward(X)
        logits = logits.squeeze(dim=1)
        return logits

    def on_epoch_start(self) -> None:
        # Placeholders for the two losses
        self.epoch_adv_loss = torch.tensor([0]).float().to(self.device)  # AdvLoss
        self.epoch_ce_loss = 0  # Cross entropy Loss

    def compute_loss(self, X: torch.Tensor, y: torch.Tensor):
        # Extract the logits for cross entropy loss
        logits = self.score(X)
        ce_loss = F.binary_cross_entropy_with_logits(logits, y)
        self.epoch_ce_loss += ce_loss

        if self.current_epoch >= self.hparams.only_ce_epochs:
            data = X[y == 0]
            # AdvLoss
            adv_loss = self.one_class_adv_loss(data).float()
            self.epoch_adv_loss += adv_loss
            loss = ce_loss + adv_loss * self.hparams.lamb
        else:
            # If only CE based training has to be done
            loss = ce_loss
        return loss

    def one_class_adv_loss(self, X: torch.Tensor):
        """Computes the adversarial loss:
        1) Sample points initially at random around the positive training
            data points
        2) Gradient ascent to find the most optimal point in set N_i(r)
            classified as +ve (label=0). This is done by maximizing
            the CE loss wrt label 0
        3) Project the points between spheres of radius R and gamma * R
            (set N_i(r))
        4) Pass the calculated adversarial points through the model,
            and calculate the CE loss wrt target class 0
        Parameters
        ----------
        X: torch.Tensor
            Batch of data to compute loss on.
        """
        batch_size = len(X)

        # Randomly sample points around the training data
        # We will perform SGD on these to find the adversarial points
        x_adv = torch.randn(X.shape).to(self.device).detach().requires_grad_()
        x_adv_sampled = x_adv + X
        for step in range(self.hparams.ascent_num_steps):
            with torch.enable_grad():
                new_targets = torch.zeros(batch_size, 1).to(self.device)
                if new_targets.squeeze().ndim > 0:
                    new_targets = torch.squeeze(new_targets)
                else:
                    new_targets = torch.zeros(batch_size).to(self.device)
                # new_targets = torch.zeros(batch_size, 1).to(self.device)
                # new_targets = torch.squeeze(new_targets)
                new_targets = new_targets.to(torch.float)

                logits = self.forward(x_adv_sampled)
                logits = torch.squeeze(logits, dim=1)

                new_loss = F.binary_cross_entropy_with_logits(logits, new_targets)
                grad = torch.autograd.grad(new_loss, [x_adv_sampled])[0]
                grad_norm = torch.norm(grad, p=2, dim=tuple(range(1, grad.dim())))
                grad_norm = grad_norm.view(-1, *[1] * (grad.dim() - 1))
                grad_norm[grad_norm == 0.0] = 10e-10
                grad_normalized = grad / grad_norm

            with torch.no_grad():
                x_adv_sampled.add_(self.hparams.ascent_step_size * grad_normalized)

            if (step + 1) % 10 == 0:
                # Project the normal points to the set N_i(r)
                h = x_adv_sampled - X
                norm_h = torch.sqrt(
                    torch.sum(h ** 2, dim=tuple(range(1, h.dim())))
                )
                alpha = torch.clamp(
                    norm_h, self.hparams.radius, self.hparams.gamma * self.hparams.radius
                ).to(self.device)
                # Make use of broadcast to project h
                proj = (alpha / norm_h).view(-1, *[1] * (h.dim() - 1))
                h = proj * h
                x_adv_sampled = X + h  # These adv_points are now on the surface of hyper-sphere

        adv_pred = self.forward(x_adv_sampled)
        adv_pred = torch.squeeze(adv_pred, dim=1)
        adv_loss = F.binary_cross_entropy_with_logits(adv_pred, (new_targets + 1))

        return adv_loss

    def training_step(self, batch, batch_idx):
        X, y, _ = batch
        X = X.float()
        return self.compute_loss(X, y)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr
        )
        scheduler = StepLR(optimizer, step_size=20, gamma=0.9)
        return [optimizer], [scheduler]

    @staticmethod
    def get_ray_config(in_features: int, n_instances: int) -> dict:
        # read parent config
        parent_cfg = BaseLightningModel.get_ray_config(in_features, n_instances)

        radius_opts = [
            min(1., np.sqrt(in_features) / 2 - 1),
            np.sqrt(in_features) / 2,
            np.sqrt(in_features) / 2 + 1
        ]
        child_cfg = {
            "lamb": 1.,
            "radius": ray_tune.choice(radius_opts),
            "gamma": 2.,
            "n_hidden_nodes": ray_tune.choice([64, 128, 512]),
            "only_ce_epochs": ray_tune.choice([10, 50, 100]),
            "ascent_step_size": ray_tune.loguniform(0.1, 0.01),
            "ascent_num_steps": ray_tune.choice([50, 100])
        }
        return dict(
            **parent_cfg,
            **child_cfg
        )
