from pytorch_lightning.utilities.cli import MODEL_REGISTRY
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA as skPCA
import pytorch_lightning as pl
import numpy as np

from pyad.utils import metrics
from pyad.utils.utils import ids_misclf_per_label


class BaseLightningShallowModel(pl.LightningModule):
    is_nn = False

    def __init__(
            self,
            in_features: int = -1,
            n_instances: int = -1,
            threshold: float = None,
            **kwargs
    ):
        super(BaseLightningShallowModel, self).__init__()
        self.threshold = threshold
        self.in_features = in_features
        self.n_instances = n_instances
        self.threshold = threshold
        # call this to save hyper-parameters to the checkpoint
        # will save children parameters as well
        self.save_hyperparameters(
            ignore=["in_features", "n_instances", "threshold"]
        )
        # Performance metrics placeholder
        self.results = None
        self.per_class_accuracy = None

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> None:
        self.clf.fit(X)

    def score(self, X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        return -self.clf.score_samples(X)

    def test(self, data: np.ndarray, y_true: np.ndarray, labels: np.ndarray) -> dict:
        # compute anomaly scores
        scores = self.score(data)
        assert np.isnan(scores).any().item() is False, "found NaN values in the final scores, aborting evaluation"

        # compute binary classification scores
        #results, y_pred = metrics.score_recall_precision_w_threshold(scores, y_true, self.threshold)
        results, y_pred = metrics.estimate_optimal_threshold(scores, y_true)

        # evaluate multi-class if labels contain over two distinct values
        if len(np.unique(labels)) > 2:
            misclf_df = ids_misclf_per_label(y_pred, y_true, labels)
            misclf_df = misclf_df.sort_values("Misclassified ratio", ascending=False)
            self.per_class_accuracy = misclf_df
            for i, row in misclf_df.iterrows():
                results[i] = row["Accuracy"]

        return results


@MODEL_REGISTRY
class OCSVM(BaseLightningShallowModel):

    def __init__(
            self,
            kernel="rbf",
            nu=0.5,
            gamma="scale",
            shrinking=False,
            verbose=True,
            **kwargs
    ):
        """
            kernel: str
                from sklearn: specifies the kernel type to be used in the algorithm
            gamma: str
                from sklearn: kernel coefficient for 'rbf', 'poly' and 'sigmoid'
            nu: float
             from sklearn: an upper bound on the fraction of training errors and a lower bound of the fraction of
             support vectors (should be in the interval (0, 1])
        ]
        """
        super(OCSVM, self).__init__(**kwargs)
        self.save_hyperparameters(
            ignore=["verbose", "shrinking"]
        )
        gamma = gamma if not any(char.isdigit() for char in gamma) else float(gamma)
        self.clf = OneClassSVM(
            kernel=kernel,
            gamma=gamma,
            shrinking=shrinking,
            verbose=verbose,
            nu=nu
        )

    def score(self, X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        return -self.clf.score_samples(X)


@MODEL_REGISTRY
class LOF(BaseLightningShallowModel):
    name = "LOF"

    def __init__(
            self,
            n_neighbors: int,
            **kwargs
    ):
        """
        n_neighbors: int
            from sklearn: the actual number of neighbors used for :meth:`kneighbors` queries
        """
        super(LOF, self).__init__(**kwargs)
        self.clf = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            novelty=True,
            n_jobs=-1
        )


@MODEL_REGISTRY
class PCA(BaseLightningShallowModel):
    name = "PCA"

    def __init__(self, n_components: int, **kwargs):
        """
        n_components: int
            sklearn: Number of components to keep
        """
        super(PCA, self).__init__(**kwargs)
        self.n_components = n_components
        self.clf = skPCA(n_components=n_components)
