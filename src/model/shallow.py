from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from recforest import RecForest as PyPiRecForest
from .base import BaseShallowModel


class RecForest(BaseShallowModel):

    def __init__(self, n_jobs=-1, **kwargs):
        super(RecForest, self).__init__(**kwargs)
        self.clf = PyPiRecForest(n_jobs=n_jobs)
        self.name = "RecForest"

    def get_params(self) -> dict:
        return {}


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

    def get_params(self) -> dict:
        return {
            "kernel": self.clf.kernel,
            "gamma": self.clf.gamma,
            "shrinking": self.clf.shrinking,
            "nu": self.clf.nu
        }


class LOF(BaseShallowModel):
    def __init__(self, n_neighbors=20, verbose=True, **kwargs):
        super(LOF, self).__init__(**kwargs)
        self.clf = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            novelty=True,
            n_jobs=-1
        )
        self.name = "LOF"

    def get_params(self) -> dict:
        return {
            "n_neighbors": self.clf.n_neighbors
        }


