import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from src.trainer.base import BaseTrainer


class NeuTraLADTrainer(BaseTrainer):

    def __init__(self, **kwargs):
        super(NeuTraLADTrainer, self).__init__(**kwargs)
        self.metric_hist = []
        self.scheduler = StepLR(self.optimizer, step_size=20, gamma=0.9)
        self.criterion = nn.MSELoss()

    def score(self, sample: torch.Tensor):
        return self.model(sample)

    def train_iter(self, X):
        scores = self.model(X)
        return scores.mean()
