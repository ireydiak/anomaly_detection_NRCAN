import torch
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR

from src.model.transformers import NeuTraLAD
from src.trainer.base import BaseTrainer


class NeuTraLADTrainer(BaseTrainer):

    def __init__(self, **kwargs):
        super(NeuTraLADTrainer, self).__init__(**kwargs)
        self.metric_hist = []

        mask_params = list()
        for mask in self.model.masks:
            mask_params += list(mask.parameters())
        self.optimizer = optim.Adam(
            list(self.model.enc.parameters()) + mask_params,
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        self.scheduler = StepLR(self.optimizer, step_size=20, gamma=0.9)
        self.criterion = nn.MSELoss()

    @staticmethod
    def load_from_file(fname: str, device: str = None):
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(fname, map_location=device)
        metric_values = ckpt["metric_values"]
        model = NeuTraLAD.load_from_ckpt(ckpt)
        trainer = NeuTraLADTrainer(model=model, batch_size=1, device=device)
        trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        trainer.metric_values = metric_values

        return trainer, model

    def score(self, sample: torch.Tensor):
        return self.model(sample)

    def train_iter(self, X):
        scores = self.model(X)
        return scores.mean()
