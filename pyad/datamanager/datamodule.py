import numpy
import pandas as pd
import pytorch_lightning as pl
from typing import Optional, Tuple
from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY
import scipy.io as scio
from torch.utils.data import DataLoader
import numpy as np
from pyad.datamanager.dataset import SimpleDataset, scaler_map


def train_test_split_normal_data(
        X: np.ndarray,
        y: np.ndarray,
        labels: np.ndarray,
        normal_size: float = 1.,
        normal_str_repr: str = "0",
        seed=None
):
    """
    Split matrix into random train and test subsets.

    X: np.ndarray
        Data matrix that will be split

    y: np.ndarray
        Binary labels (0, 1)

    labels: np.ndarray
        Text representation of the labels

    normal_size: float
        Optional argument to further subsample samples with where y == 0 (normal data)

    normal_str_repr: float
        String representation of the label for normal samples (e.g. "Benign" or "normal")
    """
    assert 0. < normal_size <= 1., "`normal_size` parameter must be inclusively in the range (0, 1], got {:2.4f}".format(
        normal_size)
    if seed:
        np.random.seed(seed)
    # separate normal and abnormal data
    normal_data = X[y == 0]
    abnormal_data = X[y == 1]
    # train, test split
    n_normal = int((len(normal_data) * normal_size) // 2)
    # shuffle normal data
    np.random.shuffle(normal_data)
    # train (normal only)
    X_train = normal_data[:n_normal]
    # test (normal + attacks)
    X_test_normal = normal_data[n_normal:]
    X_test = np.concatenate(
        (X_test_normal, abnormal_data)
    )
    y_test = np.concatenate((
        np.zeros(len(X_test_normal), dtype=np.int8),
        np.ones(len(abnormal_data), dtype=np.int8)
    ))
    test_labels = np.concatenate((
        np.array([normal_str_repr] * len(X_test_normal)),
        labels[y == 1]
    ))
    # sanity check: no attack labels associated with 0s and no normal labels associated with 1s
    for bin_y, label in zip(y_test, test_labels):
        assert bin_y == 0 and label == normal_str_repr or bin_y == 1 and label != normal_str_repr
    return X_train, X_test, y_test, test_labels


class BaseDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_dir: str,
            batch_size: int,
            scaler: str = None,
            label_col: str = None,
            normal_size: float = 1.,
            anomaly_label: int = 1,
            normal_str_label: str = "0",
            seed: int = 42):
        super(BaseDataModule, self).__init__()
        assert 0. < normal_size <= 1., "`normal_size` must be in the range (0, 1], got {:2.4f}".format(normal_size)
        # load data
        self.X, self.y, self.labels = self._load_data(data_dir, label_col)
        # sanity checks (NaNs, INF, etc.)
        self.sanity_check(self.X, self.y, self.labels)
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_data = None
        self.test_data = None
        self.in_features = self.X.shape[1]
        self.n_instances = self.X.shape[0]
        self.scaler = scaler_map[scaler]() if scaler is not None else None
        self.anomaly_label = anomaly_label
        self.anomaly_ratio = (self.y == anomaly_label).sum() / self.n_instances
        self.seed = seed
        # used to subsample the normal data
        self.normal_size = normal_size
        # used to the fetch the string labels
        self.label_col = label_col
        # used to compute per-class accuracy on the normal class
        self.normal_str_label = normal_str_label

    def _load_data(self, path: str, label_col=None) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
        if path.endswith(".npy") or path.endswith(".npz"):
            data = np.load(path, allow_pickle=True)
            X, y = data[:, :-1], data[:, -1]
            labels = data[:, label_col] if label_col else data[:, -1]
        elif path.endswith(".mat"):
            data = scio.loadmat(path)
            data = np.concatenate((data['X'], data['y']), axis=1)
            X, y = data[:, :-1], data[:, -1]
            labels = data[:, label_col] if label_col else data[:, -1]
        elif path.endswith(".csv"):
            df = pd.read_csv(path)
            y = df.loc[:, "Label"].to_numpy().astype(np.int8)
            if label_col:
                labels = df.loc[:, label_col].to_numpy()
                df = df.drop(label_col, axis=1)
            else:
                labels = y
            X = df.drop("Label", axis=1).astype(np.float32).to_numpy()
        else:
            raise RuntimeError(f"Could not open {path}. Dataset can only read .npy, .csv and .mat files.")
        return X, y, labels

    def sanity_check(self, X: np.ndarray, y: np.ndarray, labels: np.ndarray):
        assert np.isnan(X).sum() == 0, "found NaN values in the data"

    def setup(self, stage: Optional[str] = None):
        # Load and split data
        X_train, X_test, y_test, test_labels = train_test_split_normal_data(
            self.X, self.y, self.labels,
            normal_size=self.normal_size,
            normal_str_repr=self.normal_str_label,
            seed=self.seed
        )
        # normalize data
        if self.scaler:
            self.scaler.fit(X_train)
            X_train = self.scaler.transform(X_train)
            X_test = self.scaler.transform(X_test)
        # convert numpy arrays to PyTorch Datasets
        self.train_data = SimpleDataset(X=X_train, y=np.zeros(len(X_train)), labels=np.zeros(len(X_train)))
        self.test_data = SimpleDataset(X=X_test, y=y_test, labels=test_labels)

    def train_dataloader(self):
        assert self.train_data, "`train_data` undefined, did you forget to call `setup` first?"
        return DataLoader(self.train_data, batch_size=self.batch_size)

    def test_dataloader(self):
        assert self.test_data, "`test_data` undefined, did you forget to call `setup` first?"
        return DataLoader(self.test_data, batch_size=self.batch_size)


@DATAMODULE_REGISTRY
class ThyroidDataModule(BaseDataModule):
    pass


@DATAMODULE_REGISTRY
class ArrhythmiaDataModule(BaseDataModule):
    pass


@DATAMODULE_REGISTRY
class KDD10DataModule(BaseDataModule):
    def sanity_check(self, X: np.ndarray, y: np.ndarray, labels: np.ndarray):
        super(KDD10DataModule, self).sanity_check(X, y, labels)
        assert X[X < 0].sum() == 0, "detected negative values"


@DATAMODULE_REGISTRY
class NSLKDDDataModule(BaseDataModule):
    def sanity_check(self, X: np.ndarray, y: np.ndarray, labels: np.ndarray):
        super(NSLKDDDataModule, self).sanity_check(X, y, labels)
        assert X[X < 0].sum() == 0, "detected negative values"


@DATAMODULE_REGISTRY
class IDS2017DataModule(BaseDataModule):
    def sanity_check(self, X: np.ndarray, y: np.ndarray, labels: np.ndarray):
        super(IDS2017DataModule, self).sanity_check(X, y, labels)
        assert X[X < 0].sum() == 0, "detected negative values"


@DATAMODULE_REGISTRY
class IDS2018DataModule(BaseDataModule):
    def sanity_check(self, X: np.ndarray, y: np.ndarray, labels: np.ndarray):
        super(IDS2018DataModule, self).sanity_check(X, y, labels)
        assert X[X < 0].sum() == 0, "detected negative values"
