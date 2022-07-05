import torch
from pytorch_lightning.utilities.cli import MODEL_REGISTRY
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from pyad.lightning.base import BaseLightningModel
from pyad.lightning.base import create_net_layers
from torch import nn
from typing import List


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
