import torch
import torch.nn as nn


class TwoLayerMLP(nn.Module):
    def __init__(self, in_features: int, out_features: int, negative_slope=1e-2, p=0.5):
        super(TwoLayerMLP, self).__init__()
        self.input_space = in_features
        self.out_space = out_features
        self.fc_1 = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LeakyReLU(negative_slope),
            nn.Dropout(p),
        )
        self.fc_2 = nn.Linear(out_features, 1)

    def forward(self, X):
        mid_layer = self.fc_1(X)
        logits = self.fc_2(mid_layer)
        return logits, mid_layer