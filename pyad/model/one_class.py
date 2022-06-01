from collections import OrderedDict
import torch
import torch.nn as nn
from torch import Tensor
from pyad.model.base import BaseModel
from pyad.model.utils import activation_mapper, create_network


class DeepSVDD(BaseModel):
    name = "DeepSVDD"

    def __init__(self, n_layers: int, compression_factor: int, act_fn: str, **kwargs):
        super(DeepSVDD, self).__init__(**kwargs)
        self.n_layers = n_layers
        self.compression_factor = compression_factor
        self.rep_dim = None
        self.act_fn = activation_mapper[act_fn]
        self._build_network()

    @staticmethod
    def get_args_desc():
        return [
            ("n_layers", int, 2, "Number of layers"),
            ("compression_factor", int, 2, "Compression factor of the network"),
            ("act_fn", str, "relu", "Activation function of the network")
        ]

    def _build_network(self):
        in_features = self.in_features
        compression_factor = self.compression_factor
        out_features = in_features // compression_factor
        layers = []
        for _ in range(self.n_layers - 1):
            layers.append([in_features, out_features, self.act_fn])
            in_features = out_features
            out_features = in_features // compression_factor
            assert out_features > 0, "out_features {} <= 0".format(out_features)
        layers.append(
            [in_features, out_features, None]
        )
        self.rep_dim = out_features
        self.net = create_network(layers).to(self.device)

    @staticmethod
    def load_from_ckpt(ckpt):
        model = DeepSVDD(
            in_features=ckpt["in_features"],
            n_instances=ckpt["n_instances"],
            n_layers=ckpt["n_layers"],
            compression_factor=ckpt["compression_factor"],
            act_fn=str(ckpt["act_fn"]).lower().replace("()", "")
        )
        return model

    def forward(self, X: Tensor):
        return self.net(X)

    def get_params(self) -> dict:
        params = {
            "n_layers": self.n_layers,
            "compression_factor": self.compression_factor,
            "act_fn": str(self.act_fn).lower().replace("()", "")
        }
        return dict(
            **super().get_params(),
            **params
        )


class DROCC(BaseModel):
    name = "DROCC"

    def __init__(
            self,
            lamb=1.,
            radius=3.,
            gamma=2.,
            num_classes=1,
            num_hidden_nodes=20,
            **kwargs
    ):
        super(DROCC, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.num_hidden_nodes = num_hidden_nodes
        self.lamb = lamb
        self.radius = radius
        self.gamma = gamma
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

    @staticmethod
    def get_args_desc():
        return [
            ("lamb", float, 1., "Weight given to the adversarial loss"),
            ("radius", float, 3., "Radius of hypersphere to sample points from"),
            ("gamma", float, 2., "Parameter to vary projection"),
            ("num_classes", int, 1, ""),
            ("num_hidden_nodes", int, 20, "")
        ]

    @staticmethod
    def load_from_ckpt(ckpt):
        model = DROCC(
            in_features=ckpt["in_features"],
            n_instances=ckpt["n_instances"],
            lamb=ckpt["lamb"],
            radius=ckpt["radius"],
            gamma=ckpt["gamma"],
            num_classes=ckpt["num_classes"],
            num_hidden_nodes=ckpt["num_hidden_nodes"]
        )
        return model

    def get_params(self) -> dict:

        params = {
            "lamb": self.lamb,
            "radius": self.radius,
            "gamma": self.gamma,
            "num_classes": self.num_classes,
            "num_hidden_nodes": self.num_hidden_nodes
        }
        return dict(
            **super(DROCC, self).get_params(),
            **params
        )

    def forward(self, X: torch.Tensor):
        features = self.feature_extractor(X)
        logits = self.classifier(features.view(-1, self.size_final))
        return logits
