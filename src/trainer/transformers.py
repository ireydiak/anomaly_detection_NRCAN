import torch
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from .base import BaseTrainer


class NeuTraLADTrainer(BaseTrainer):

    def __init__(self, **kwargs):
        super(NeuTraLADTrainer, self).__init__(**kwargs)
        self.metric_hist = []

        mask_params = list()
        for mask in self.model.masks:
            mask_params += list(mask.parameters())
        self.optimizer = optim.Adam(list(self.model.enc.parameters()) + mask_params, lr=self.lr,
                                    weight_decay=self.weight_decay)

        self.scheduler = StepLR(self.optimizer, step_size=20, gamma=0.9)
        self.criterion = nn.MSELoss()

    def score(self, sample: torch.Tensor):
        return self.model(sample)

    def train_iter(self, X):
        scores = self.model(X)
        return scores.mean()
