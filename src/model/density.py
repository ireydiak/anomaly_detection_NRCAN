import torch
from torch import nn
from torch.nn import Parameter
from src.model.base import BaseModel


class DSEBM(BaseModel):
    name = "DSEBM"

    def __init__(self, fc_1_out: int, fc_2_out: int, **kwargs):
        super(DSEBM, self).__init__(**kwargs)
        self.fc_1_out = fc_1_out
        self.fc_2_out = fc_2_out
        self._build_network()

    @staticmethod
    def load_from_ckpt(ckpt):
        model = DSEBM(
            in_features=ckpt["in_features"],
            n_instances=ckpt["n_instances"],
            fc_1_out=ckpt["fc_1_out"],
            fc_2_out=ckpt["fc_2_out"]
        )
        return model

    @staticmethod
    def get_args_desc():
        return [
            ("fc_1_out", int, 128, "Output dimension of the first layer"),
            ("fc_2_out", int, 512, "Output dimension of the last layer")
        ]

    def _build_network(self):
        # TODO: Make model more flexible. Users should be able to set the number of layers
        self.fc_1 = nn.Linear(self.in_features, self.fc_1_out)
        self.fc_2 = nn.Linear(self.fc_1_out, self.fc_2_out)
        self.softp = nn.Softplus()
        self.bias_inv_1 = Parameter(torch.Tensor(self.fc_1_out))
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

    def get_params(self) -> dict:
        params = {
            "fc_1_out": self.fc_1_out,
            "fc_2_out": self.fc_2_out
        }

        return dict(
            **super().get_params(),
            **params
        )
