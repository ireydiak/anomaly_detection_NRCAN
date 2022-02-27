import torch
from .BaseTrainer import BaseTrainer
from torch import nn
from loss import EntropyLoss


class MemAETrainer(BaseTrainer):

    def __init__(self, alpha: float, **kwargs) -> None:
        super(MemAETrainer, self).__init__(**kwargs)
        self.alpha = alpha
        self.recon_loss_fn = nn.MSELoss().to(self.device)
        self.entropy_loss_fn = EntropyLoss().to(self.device)

    def train_iter(self, sample: torch.Tensor):
        x_hat, w_hat = self.model(sample)
        R = self.recon_loss_fn(sample, x_hat)
        E = self.entropy_loss_fn(w_hat)
        return R + (self.alpha * E)

    def score(self, sample: torch.Tensor):
        x_hat, _ = self.model(sample)
        return torch.sum((sample - x_hat) ** 2, axis=1)
