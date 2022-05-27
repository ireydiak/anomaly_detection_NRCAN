from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from recforest import RecForest as PyPiRecForest
from src.model.base import BaseShallowModel
from sklearn.decomposition import PCA as skPCA


class RecForest(BaseShallowModel):
    name = "RecForest"

    def __init__(self, n_jobs=-1, **kwargs):
        super(RecForest, self).__init__(**kwargs)
        self.clf = PyPiRecForest(n_jobs=n_jobs)

    @staticmethod
    def get_args_desc():
        return []

    def get_params(self) -> dict:
        return {}


class OCSVM(BaseShallowModel):
    name = "OCSVM"

    def __init__(
            self,
            kernel="rbf",
            nu=0.5,
            gamma="scale",
            shrinking=False,
            verbose=True,
            **kwargs
    ):
        super(OCSVM, self).__init__(**kwargs)
        gamma = gamma if not any(char.isdigit() for char in gamma) else float(gamma)
        self.clf = OneClassSVM(
            kernel=kernel,
            gamma=gamma,
            shrinking=shrinking,
            verbose=verbose,
            nu=nu
        )

    @staticmethod
    def get_args_desc():
        return [
            ("kernel", str, "rbf", "sklearn: specifies the kernel type to be used in the algorithm"),
            ("gamma", str, "auto", "sklearn: kernel coefficient for 'rbf', 'poly' and 'sigmoid'"),
            ("nu", float, 0.5, "sklearn: an upper bound on the fraction of training errors and a lower bound of the fraction of support vectors (should be in the interval (0, 1])")
        ]

    def get_params(self) -> dict:
        return {
            "kernel": self.clf.kernel,
            "gamma": self.clf.gamma,
            "shrinking": self.clf.shrinking,
            "nu": self.clf.nu
        }


class LOF(BaseShallowModel):
    name = "LOF"

    def __init__(self, n_neighbors: int, **kwargs):
        super(LOF, self).__init__(**kwargs)
        self.clf = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            novelty=True,
            n_jobs=-1
        )

    @staticmethod
    def get_args_desc():
        return [
            ("n_neighbors", int, 20, "sklearn: the actual number of neighbors used for :meth:`kneighbors` queries")
        ]

    def get_params(self) -> dict:
        return {
            "n_neighbors": self.clf.n_neighbors
        }


class PCA(BaseShallowModel):
    name = "PCA"

    def __init__(self, n_components: int, **kwargs):
        super(PCA, self).__init__(**kwargs)
        self.n_components = n_components
        self.clf = skPCA(n_components=n_components)

    @staticmethod
    def get_args_desc():
        return [
            ("n_components", int, 1, "sklearn: Number of components to keep")
        ]

    def get_params(self) -> dict:
        return {
            "n_components": self.n_components
        }
