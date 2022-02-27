import torch
from trainer.BaseTrainer import BaseTrainer


class AETrainer(BaseTrainer):

    def score(self, sample: torch.Tensor):
        _, X_prime = self.model(sample)
        return ((sample - X_prime) ** 2).sum()

    def train_iter(self, X):
        code, X_prime = self.model(X)
        l2_z = code.norm(2, dim=1).mean()
        reg = 0.5
        loss = ((X - X_prime) ** 2).sum(axis=-1).mean() + reg * l2_z

        return loss.item()
