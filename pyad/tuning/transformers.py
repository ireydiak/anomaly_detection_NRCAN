import os
from typing import Union
import numpy as np
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import trange
from pyad.loss.TripletCenterLoss import TripletCenterLoss
from pyad.model.transformers import NeuTraLAD
from pyad.trainer.base import BaseTrainer
from pyad.tuning.base import BaseTuner
from ray import tune as ray_tune



class NeuTraLADTrainer(BaseTuner):
    def setup(self, config: dict):
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.model = MemAutoEncoder(
        #     in_features=config["in_features"],
        #     n_instances=config["n_instances"],
        #     mem_dim=config["mem_dim"],
        #     latent_dim=config["latent_dim"],
        #     shrink_thres=config["shrink_thres"],
        #     n_layers=config["n_layers"],
        #     compression_factor=config["compression_factor"],
        #     alpha=config["alpha"],
        #     act_fn=config["act_fn"]
        # )
        # self.alpha = config["alpha"]
        # self.model = self.model.to(self.device)
        # self.optimizer = optim.Adam(
        #     self.model.parameters(),
        #     lr=config["lr"]
        # )
        # self.recon_loss_fn = nn.MSELoss().to(self.device)
        # self.entropy_loss_fn = EntropyLoss().to(self.device)
        # self.dataset = config["dataset"]
        # self.train_ldr, self.val_ldr, self.test_ldr = get_data_loaders(
        #     dataset=self.dataset,
        #     batch_size=config["batch_size"]
        # )

    def train_iter(self, sample: torch.Tensor):
        # x_hat, w_hat = self.model(sample)
        # R = self.recon_loss_fn(sample, x_hat)
        # E = self.entropy_loss_fn(w_hat)
        return R + (self.alpha * E)

    def score(self, X: torch.Tensor):
        # x_hat, _ = self.model(X)
        return torch.sum((X - x_hat) ** 2, dim=1)

    @staticmethod
    def get_tunable_params(n_instances: int, in_features: int):
        if in_features > 20:
            layer_opts = [3, 4]
        else:
            layer_opts = [1, 2]
            compression_opts = []
        return {
            "encoder_hidden_dims": ray_tune.choice([]),
            "n_transforms": ray_tune.choice([11]),
            "n_layers": ray_tune.choice(layer_opts),
            "trans_type": ray_tune.choice(["mul", "res"]),
            "temperature": ray_tune.loguniform(1, 0.001),
            "trans_hidden_dims": 0 ,
        }
        # if in_features < 20:
        #     n_layers = [1, 2]
        #     latent_dims = [1, in_features // 2]
        # else:
        #     n_layers = [2, 3, 4]
        #     latent_dims = np.linspace(1, in_features // n_layers[-1], 5, dtype=int)
        # return {
        #     "mem_dim": ray_tune.choice([50, 150, 300]),
        #     "latent_dim": ray_tune.choice(latent_dims),
        #     "shrink_thres": ray_tune.loguniform(1 / n_instances, 3 / n_instances),
        #     "n_layers": ray_tune.choice(n_layers),
        #     "compression_factor": 2,
        #     "alpha": ray_tune.loguniform(2e-4, 1e-1),
        #     "act_fn": "relu"
        # }


class GOADTrainer(BaseTuner):
    pass
