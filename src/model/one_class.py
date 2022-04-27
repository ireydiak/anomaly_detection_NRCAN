from collections import OrderedDict

import torch
import torch.nn as nn
from torch import Tensor
from .base import BaseModel


class DeepSVDD(BaseModel):
    """
    Follows SKLearn's API
    (https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html#sklearn.svm.OneClassSVM.decision_function)
    """

    def resolve_params(self, dataset_name: str):
        pass

    def __init__(self, **kwargs):
        super(DeepSVDD, self).__init__(**kwargs)
        self.rep_dim = self.in_features // 4
        self.name = "DeepSVDD"
        self.net = self._build_network()

    def _build_network(self):
        return nn.Sequential(
            nn.Linear(self.in_features, self.in_features // 2),
            nn.ReLU(),
            nn.Linear(self.in_features // 2, self.rep_dim)
        ).to(self.device)

    def forward(self, X: Tensor):
        return self.net(X)

    def get_params(self) -> dict:
        return {
            "in_features": self.in_features,
            "rep_dim": self.rep_dim
        }


class DROCC(BaseModel):

    def __init__(self,
                 num_classes=1,
                 num_hidden_nodes=20,
                 **kwargs):
        super(DROCC, self).__init__(**kwargs)
        self.name = "DROCC"
        self.num_classes = num_classes
        self.num_hidden_nodes = num_hidden_nodes
        activ = nn.ReLU(True)
        self.feature_extractor = nn.Sequential(
            OrderedDict([
                ('fc', nn.Linear(self.in_features, self.num_hidden_nodes)),
                ('relu1', activ)])
        )
        self.size_final = self.num_hidden_nodes
        self.classifier = nn.Sequential(
            OrderedDict([
                ('fc1', nn.Linear(self.size_final, self.num_classes))
            ])
        )

    def get_params(self) -> dict:
        return {
            "num_classes": self.num_classes,
            "num_hidden_nodes": self.num_hidden_nodes
        }

    def forward(self, X: torch.Tensor):
        features = self.feature_extractor(X)
        logits = self.classifier(features.view(-1, self.size_final))
        return logits

    def resolve_params(self, dataset_name: str):
        pass
