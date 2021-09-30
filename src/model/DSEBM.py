import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter


class DSEBM(nn.Module):
    def __init__(self, D: int):
        super(DSEBM, self).__init__()
        self.D = D
        self.noise = None
        self._build_network()

    def _build_network(self):
        self.fc_1 = nn.Linear(self.D, 128)
        self.fc_2 = nn.Linear(128, 512)
        self.softp = nn.Softplus()

        self.bias_inv_1 = Parameter(torch.Tensor(128))
        self.bias_inv_2 = Parameter(torch.Tensor(self.D))
        torch.nn.init.xavier_normal_(self.fc_2.weight)
        torch.nn.init.xavier_normal_(self.fc_1.weight)
        # torch.nn.init.xavier_normal_(self.bias_inv_1)
        # torch.nn.init.xavier_normal_(self.bias_inv_2)


    def random_noise_like(self, X: torch.Tensor):
        return torch.normal(mean=0., std=1., size=X.shape).float()

    def forward(self, X: torch.Tensor):

        output = self.softp(self.fc_1(X))
        output = self.softp(self.fc_2(output))

        # inverse layer
        output = self.softp((output @ self.fc_2.weight) + self.bias_inv_1)
        output = self.softp((output @ self.fc_1.weight) + self.bias_inv_2)

        return output

