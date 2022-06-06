import torch
import torch.nn as nn


class TripletLoss(nn.Module):
    def __init__(self, margin: float = 1.):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, X: torch.Tensor):
        device = X.device
        means = X.mean(0).unsqueeze(0)
        res = ((X.unsqueeze(2) - means.unsqueeze(1)) ** 2).sum(-1)
        pos = torch.diagonal(res, dim1=1, dim2=2)
        offset = torch.diagflat(torch.ones(X.size(1))).unsqueeze(0).to(device) * 1e6
        neg = (res + offset).min(-1)[0]
        loss = torch.clamp(pos + self.margin - neg, min=0).mean()
        return loss
