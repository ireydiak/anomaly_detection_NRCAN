import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import math


# relu based hard shrinkage function, only works for positive values
def hard_shrink_relu(input, lamb=0, epsilon=1e-12) -> float:
    return (F.relu(input - lamb) * input) / (torch.abs(input - lamb) + epsilon)


class MemoryUnit(nn.Module):
    def __init__(self, mem_dim: int, D: int, shrink_thres=0.0025):
        super(MemoryUnit, self).__init__()
        self.mem_dim = mem_dim
        self.D = D
        # M x C
        self.weight = Parameter(torch.Tensor(self.mem_dim, self.D))
        self.shrink_thres = shrink_thres
        self.bias = None
        self.reset_params()

    def reset_params(self) -> None:
        # TODO: diff stdv = 1. / math.sqrt(self.weight.size(1))
        stdv = 1. / math.sqrt(3)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x) -> (float, float):
        # Fea x Mem^T, (TxC) x (CxM) = TxM
        att_weight = F.linear(x, self.weight)
        # TxM
        att_weight = F.softmax(att_weight, dim=1)
        if self.shrink_thres > 0:
            # att_weight = hard_shrink_relu(att_weight, lamb=self.shrink_thres)
            att_weight = F.relu(att_weight)
            # att_weight = F.softshrink(att_weight, lambd=self.shrink_thres)
            # normalize???
            att_weight = F.normalize(att_weight, p=1, dim=1)
            # att_weight = F.softmax(att_weight, dim=1)
            # att_weight = self.hard_sparse_shrink_opt(att_weight)
        # Mem^T, MxC
        # mem_trans = self.weight.permute(1, 0)
        # AttWeight x Mem^T^T = AW x Mem, (TxM) x (MxC) = TxC
        output = F.linear(att_weight, self.weight.T)
        return output, att_weight

    def get_params(self) -> dict:
        return {
            'Mem Dim': self.mem_dim,
            'D': self.D
        }


class MemoryModule(nn.Module):
    def __init__(self, mem_dim: int, D: int, shrink_thres=0.0025, device='cpu'):
        super(MemoryModule, self).__init__()
        self.mem_dim = mem_dim
        self.D = D
        self.shrink_thres = shrink_thres
        self.memory = MemoryUnit(mem_dim, D, shrink_thres)
        self.device = device

    def forward(self, x):
        return self.memory(x)
