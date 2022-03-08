import torch
from sklearn.svm import OneClassSVM
from recforest import RecForest as PyPiRecForest
from torch.utils.data import DataLoader

from src.model.base import BaseShallowModel


class RecForest(BaseShallowModel):

    def __init__(self, n_jobs=-1, random_state=-1, **kwargs):
        super(RecForest, self).__init__(**kwargs)
        self.clf = PyPiRecForest(n_jobs=n_jobs, random_state=random_state)
        
    def get_params(self) -> dict:
        return {}

    def score(self, sample: torch.Tensor):
        return self.clf.predict(sample.numpy())

    def resolve_params(self, dataset_name: str):
        pass


class OCSVM(BaseShallowModel):
    def __init__(self, kernel="rbf", gamma="scale", shrinking=False, verbose=True, nu=0.5, **kwargs):
        super(OCSVM, self).__init__(**kwargs)
        self.clf = OneClassSVM(
            kernel=kernel,
            gamma=gamma,
            shrinking=shrinking,
            verbose=verbose,
            nu=nu
        )
        self.name = "OC-SVM"

    def score(self, sample: torch.Tensor):
        return -self.clf.predict(sample.numpy())

    def get_params(self) -> dict:
        return {
            "kernel": self.clf.kernel,
            "gamma": self.clf.gamma,
            "shrinking": self.clf.shrinking,
            "nu": self.clf.nu
        }

    def resolve_params(self, dataset_name: str):
        pass

