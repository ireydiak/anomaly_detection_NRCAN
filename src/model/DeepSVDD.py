import torch
import torch.nn as nn
from torch import Tensor
from .BaseModel import BaseModel


class DeepSVDD:
    """
    Follows SKLearn's API
    (https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html#sklearn.svm.OneClassSVM.decision_function)
    """

    def __init__(self, D: int, **kwargs):
        super(DeepSVDD, self).__init__()
        self.D = D
        self.device = kwargs.get('device', 'cpu')
        self.net = self._build_network()
        self.rep_dim = D // 4

    def _build_network(self):
        return nn.Sequential(
            nn.Linear(self.D, self.D // 2),
            nn.ReLU(),
            nn.Linear(self.D // 2, self.D // 4)
        ).to(self.device)

    def forward(self, X: Tensor):
        return self.net(X)

    def get_params(self) -> dict:
        return {'D': self.D, 'rep_dim': self.rep_dim}
