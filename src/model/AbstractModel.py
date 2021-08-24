import copy
import gzip
import pickle

from torch import nn


class AbstractModel(nn.Module):

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
        assert isinstance(model, AbstractModel)
        return model

    def save(self, filename):
        # Save model to file (.pklz)
        model = copy.deepcopy(self)
        model.reset()
        with gzip.open(filename, 'wb') as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
