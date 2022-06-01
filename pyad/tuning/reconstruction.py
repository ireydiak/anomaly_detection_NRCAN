import torch
import numpy as np
from torch import optim
from torch import nn
from pyad.loss.EntropyLoss import EntropyLoss
from pyad.model.reconstruction import AutoEncoder, MemAutoEncoder
from ray import tune as ray_tune
from pyad.tuning.data import get_data_loaders
from pyad.tuning.base import BaseTuner


class AutoEncoderTuner(BaseTuner):
    def setup(self, config: dict):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoEncoder(
            in_features=config["in_features"],
            n_instances=config["n_instances"],
            n_layers=config["n_layers"],
            compression_factor=config["compression_factor"],
            latent_dim=config["latent_dim"],
            act_fn=config["act_fn"],
            reg=config["reg"]
        )
        self.model = self.model.to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config["lr"]
        )
        self.dataset = config["dataset"]
        self.train_ldr, self.val_ldr, self.test_ldr = get_data_loaders(dataset=self.dataset, batch_size=config["batch_size"])

    def train_iter(self, X: torch.Tensor):
        code, X_prime = self.model(X)
        l2_z = code.norm(2, dim=1).mean()
        reg = 0.5
        loss = ((X - X_prime) ** 2).sum(axis=-1).mean() + reg * l2_z

        return loss

    def score(self, X: torch.Tensor):
        _, X_prime = self.model(X)
        return ((X - X_prime) ** 2).sum(axis=1)

    @staticmethod
    def get_tunable_params(n_instances: int, in_features: int):
        if in_features < 20:
            n_layers = [1, 2]
            latent_dims = [1, in_features // 2]
        else:
            n_layers = [2, 3, 4]
            latent_dims = np.linspace(1, in_features // n_layers[-1], 5, dtype=int)
        return {
            "n_layers": ray_tune.choice(n_layers),
            "compression_factor": 2,
            "latent_dim": ray_tune.choice(latent_dims),
            "act_fn": "relu",
            "reg": ray_tune.choice([0.1, 0.5, 0.9])
        }


class MemAETuner(BaseTuner):
    def setup(self, config: dict):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = MemAutoEncoder(
            in_features=config["in_features"],
            n_instances=config["n_instances"],
            mem_dim=config["mem_dim"],
            latent_dim=config["latent_dim"],
            shrink_thres=config["shrink_thres"],
            n_layers=config["n_layers"],
            compression_factor=config["compression_factor"],
            alpha=config["alpha"],
            act_fn=config["act_fn"]
        )
        self.alpha = config["alpha"]
        self.model = self.model.to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config["lr"]
        )
        self.recon_loss_fn = nn.MSELoss().to(self.device)
        self.entropy_loss_fn = EntropyLoss().to(self.device)
        self.dataset = config["dataset"]
        self.train_ldr, self.val_ldr, self.test_ldr = get_data_loaders(
            dataset=self.dataset,
            batch_size=config["batch_size"]
        )

    def train_iter(self, sample: torch.Tensor):
        x_hat, w_hat = self.model(sample)
        R = self.recon_loss_fn(sample, x_hat)
        E = self.entropy_loss_fn(w_hat)
        return R + (self.alpha * E)

    def score(self, X: torch.Tensor):
        x_hat, _ = self.model(X)
        return torch.sum((X - x_hat) ** 2, dim=1)

    @staticmethod
    def get_tunable_params(n_instances: int, in_features: int):
        if in_features < 20:
            n_layers = [1, 2]
            latent_dims = [1, in_features // 2]
        else:
            n_layers = [2, 3, 4]
            latent_dims = np.linspace(1, in_features // n_layers[-1], 5, dtype=int)
        return {
            "mem_dim": ray_tune.choice([50, 150, 300]),
            "latent_dim": ray_tune.choice(latent_dims),
            "shrink_thres": ray_tune.loguniform(1 / n_instances, 3 / n_instances),
            "n_layers": ray_tune.choice(n_layers),
            "compression_factor": 2,
            "alpha": ray_tune.loguniform(2e-4, 1e-1),
            "act_fn": "relu"
        }
