import torch
from torch import nn
from torch.nn import Parameter

from BaseModel import BaseModel


class DSEBM(BaseModel):
    def __init__(self, **kwargs):
        self.noise = None
        self.fc_1 = None
        self.fc_2 = None
        self.softp = None
        self.bias_inv_1 = None
        self.bias_inv_2 = None
        super(DSEBM, self).__init__(**kwargs)

    def resolve_params(self, dataset_name: str):
        if dataset_name == 'Arrhythmia' or dataset_name == 'Thyroid':
            self.fc_1 = nn.Linear(self.in_features, 10)
            self.fc_2 = nn.Linear(10, 2)
            self.softp = nn.Softplus()

            self.bias_inv_1 = Parameter(torch.Tensor(10))
            self.bias_inv_2 = Parameter(torch.Tensor(self.in_features))
        else:
            self.fc_1 = nn.Linear(self.in_features, 128)
            self.fc_2 = nn.Linear(128, 512)
            self.softp = nn.Softplus()

            self.bias_inv_1 = Parameter(torch.Tensor(128))
            self.bias_inv_2 = Parameter(torch.Tensor(self.in_features))

        torch.nn.init.xavier_normal_(self.fc_2.weight)
        torch.nn.init.xavier_normal_(self.fc_1.weight)
        self.fc_1.bias.data.zero_()
        self.fc_2.bias.data.zero_()
        self.bias_inv_1.data.zero_()
        self.bias_inv_2.data.zero_()

    def random_noise_like(self, X: torch.Tensor):
        return torch.normal(mean=0., std=1., size=X.shape).float()

    def forward(self, X: torch.Tensor):
        output = self.softp(self.fc_1(X))
        output = self.softp(self.fc_2(output))

        # inverse layer
        output = self.softp((output @ self.fc_2.weight) + self.bias_inv_1)
        output = self.softp((output @ self.fc_1.weight) + self.bias_inv_2)

        return output

    def get_params(self):
        return {
            'in_features': self.in_features
        }
