import torch

from torch.utils.data import DataLoader
from src.trainer.base import BaseShallowTrainer


class RecForestTrainer(BaseShallowTrainer):

    def score(self, sample: torch.Tensor):
        return self.model.predict(sample.numpy())

    def train_iter(self, sample: torch.Tensor):
        pass

    def train(self, dataset: DataLoader):
        self.model.fit(dataset.dataset.dataset.dataset.X)

    def get_params(self) -> dict:
        return {
            **self.model.get_params()
        }


class OCSVMTrainer(BaseShallowTrainer):

    def score(self, sample: torch.Tensor):
        return -self.model.predict(sample.numpy())

    def train_iter(self, sample: torch.Tensor):
        pass

    def train(self, dataset: DataLoader):
        self.model.fit(dataset.dataset.dataset.dataset.X)

    def get_params(self) -> dict:
        return {
            **self.model.get_params()
        }
