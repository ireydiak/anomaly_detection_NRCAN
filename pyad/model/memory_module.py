import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import math


# relu based hard shrinkage function, only works for positive values
def hard_shrink_relu(X, lamb=0, epsilon=1e-12) -> float:
    return (F.relu(X - lamb) * input) / (torch.abs(X - lamb) + epsilon)


class MemoryUnit(nn.Module):
    def __init__(self, mem_dim: int, in_features: int, shrink_thres=0.0025, device='cpu'):
        super(MemoryUnit, self).__init__()
        self.device = device
        self.mem_dim = mem_dim
        self.in_features = in_features
        # M x C
        self.weight = Parameter(torch.Tensor(self.mem_dim, self.in_features)).to(device)
        self.shrink_thres = shrink_thres
        self.bias = None
        self.reset_params()

    def reset_params(self) -> None:
        stdv = 1. / math.sqrt(self.weight.size()[1])
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x) -> (float, float):
        # Fea x Mem^T, (TxC) x (CxM) = TxM
        att_weight = F.linear(x, self.weight)
        # TxM
        att_weight = F.softmax(att_weight, dim=1)
        if self.shrink_thres > 0:
            # TODO: test with hard_shrink_relu
            # att_weight = hard_shrink_relu(att_weight, lamb=self.shrink_thres)
            att_weight = F.relu(att_weight)
            att_weight = F.normalize(att_weight, p=1, dim=1)
        # Mem^T, MxC
        # AttWeight x Mem^T^T = AW x Mem, (TxM) x (MxC) = TxC
        output = F.linear(att_weight, self.weight.T)
        return output, att_weight

    def get_params(self) -> dict:
        return {
            'mem_dim': self.mem_dim,
            'in_features': self.in_features
        }
