import torch.nn as nn
from .memory_module import MemoryModule


class MemAutoEncoder(nn.Module):

    def __init__(self, D, L, mem_dim, shrink_thres=0.0025, device='cpu'):
        """
        Implements model xyx as described in paper xyx, on page 8 (or 1712 if reading from journal)
        https://github.com/donggong1/memae-anomaly-detection

        Parameters
        ----------
        chnum_in: TODO find its definition
        mem_dim: TODO find its definition
        shrink_thres: TODO find its definition
        """
        super(MemAutoEncoder, self).__init__()
        self.D = D
        self.L = L
        # for kdd10, use D = 120; L = 3
        self.encoder = nn.Sequential(
            nn.Linear(D, 60),
            nn.Tanh(),
            nn.Linear(60, 30),
            nn.Tanh(),
            nn.Linear(30, 10),
            nn.Tanh(),
            nn.Linear(10, L)
        ).to(device)
        # TODO: replace L with D to see diff
        self.mem_rep = MemoryModule(mem_dim, L, shrink_thres, device=device).to(device)
        self.decoder = nn.Sequential(
            nn.Linear(L, 10),
            nn.Tanh(),
            nn.Linear(10, 30),
            nn.Tanh(),
            nn.Linear(30, 60),
            nn.Tanh(),
            nn.Linear(60, D)
        ).to(device)

    def forward(self, x):
        f_e = self.encoder(x)
        f_mem, att = self.mem_rep(f_e)
        f_d = self.decoder(f_mem)
        return f_d, att

    def get_params(self):
        return {
            'L': self.L,
            'D': self.D
        }
