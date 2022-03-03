import copy
import gzip
import pickle
from abc import abstractmethod

import torch

from torch import nn


class BaseModel(nn.Module):

    def __init__(self, dataset_name: str, in_features: int, n_instances: int, device: str):
        super().__init__()
        self.dataset_name = dataset_name
        self.device = device
        self.in_features = in_features
        self.n_instances = n_instances
        self.resolve_params(dataset_name)

    def reset(self):
        self.apply(self.weight_reset)

    def weight_reset(self, m):
        reset_parameters = getattr(m, "reset_parameters", None)
        if callable(reset_parameters):
            m.reset_parameters()

    @staticmethod
    def load(filename):
        # Load model from file (.pklz)
        with gzip.open(filename, 'rb') as f:
            model = pickle.load(f)
        assert isinstance(model, BaseModel)
        return model

    @abstractmethod
    def resolve_params(self, dataset_name: str):
        pass

    def save(self, filename):
        # Save model to file (.pklz)
        # model = copy.deepcopy(self.detach())
        # model.to('cpu')
        # with gzip.open(filename, 'wb') as f:
        #     pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
        torch.save(self.state_dict(), filename + "pt")
        # return torch.load(weights_path)
