import copy
import gzip
import pickle
import torch
from abc import abstractmethod, ABC
from torch import nn


class BaseModel(nn.Module):

    def __init__(self, dataset_name: str, in_features: int, n_instances: int, device: str, **kwargs):
        super(BaseModel, self).__init__()
        self.dataset_name = dataset_name
        self.device = device
        self.in_features = in_features
        self.n_instances = n_instances
        self.resolve_params(dataset_name)

    def get_params(self) -> dict:
        return {
            "in_features": self.in_features,
            "n_instances": self.n_instances
        }

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
        torch.save(self.state_dict(), filename)
        # return torch.load(weights_path)


class BaseShallowModel(ABC):

    def __init__(self, dataset_name: str, in_features: int, n_instances: int, device: str):
        self.dataset_name = dataset_name
        self.device = device
        self.in_features = in_features
        self.n_instances = n_instances
        self.resolve_params(dataset_name)

    def resolve_params(self, dataset_name: str):
        pass

    def reset(self):
        """
        This function does nothing.
        It exists only for consistency with deep models
        """
        pass

    def save(self, filename):
        """
        This function does nothing.
        It exists only for consistency with deep models
        """
        pass
