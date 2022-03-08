from sklearn.svm import OneClassSVM
from recforest import RecForest as PyPiRecForest
from src.model.base import BaseShallowModel


class RecForest(BaseShallowModel):
    def resolve_params(self, dataset_name: str):
        pass

    def __init__(self, n_jobs=-1, random_state=-1, **kwargs):
        super(RecForest, self).__init__(**kwargs)
        self.clf = PyPiRecForest(n_jobs=n_jobs, random_state=random_state)
        
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

    def resolve_params(self, dataset_name: str):
        pass

    def get_params(self) -> dict:
        return {
            "kernel": self.clf.kernel,
            "gamma": self.clf.gamma,
            "shrinking": self.clf.shrinking,
            "nu": self.clf.nu
        }

