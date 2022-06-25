import torch
import torch.nn as nn


class TripletCenterLoss(nn.Module):
    def __init__(self, margin: float = 1.):
        super(TripletCenterLoss, self).__init__()
        self.margin = margin

    def forward(self, X: torch.Tensor):
        device = X.device
        means = X.mean(0).unsqueeze(0)
        # Distance between each transformed samples
        # e.g. dist_{ij} represents the distance of f(x_i) with centers y_i and y_j
        # where f(.) denotes the neural network
        dist = ((X.unsqueeze(2) - means.unsqueeze(1)) ** 2).sum(-1)
        # Inner-class distance
        pos = torch.diagonal(dist, dim1=1, dim2=2)
        # Diagonal matrix of size (1, num_hidden_nodes, num_hidden_nodes) with large values (1e6)
        offset = torch.diagflat(torch.ones(X.size(1))).unsqueeze(0).to(device) * 1e6
        # Min inter-class distance
        neg = (dist + offset).min(-1)[0]
        # Clamp acts as the `min` function
        loss = torch.clamp(pos + self.margin - neg, min=0)
        return loss.mean()
