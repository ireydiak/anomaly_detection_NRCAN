import copy
import gzip
import pickle
import torch

from torch import nn

class BaseModel(nn.Module):

    def __init__(self, in_features: int, device: str):
        super().__init__()
        self.device = device
        self.D = in_features

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

    def save(self, filename):
        # Save model to file (.pklz)
        # model = copy.deepcopy(self.detach())
        # model.to('cpu')
        # with gzip.open(filename, 'wb') as f:
        #     pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
        torch.save(self.state_dict(), filename + "pt")
        #return torch.load(weights_path)
