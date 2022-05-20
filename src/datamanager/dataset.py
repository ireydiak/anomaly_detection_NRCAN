import numpy as np
import pandas as pd
import scipy.io
import torch
from torch.utils.data import Dataset, Subset, DataLoader
from torch.utils.data.dataset import T_co
from typing import Tuple


class AbstractDataset(Dataset):
    def __init__(self, path: str, normal_size: float = 1.0, seed=None, **kwargs):
        self.name = self.__class__.__name__
        self.labels = np.array([])
        X = self._load_data(path)
        anomaly_label = kwargs.get('anomaly_label', 1)
        normal_label = kwargs.get('normal_label', 0)

        self.X = X[:, :-1]
        self.y = X[:, -1]
        if self.labels.size == 0:
            self.labels = self.y

        if normal_size < 1.0:
            # Keeps `normal_size` percent of normal labels
            normal_idx = np.where(X[:, -1] == normal_label)[0]
            anomaly_idx = np.where(X[:, -1] == anomaly_label)[0]

            if seed:
                np.random.seed(seed)
            np.random.shuffle(normal_idx)

            subsampled_normal_idx = normal_idx[int(len(normal_idx) * normal_size):]
            idx_to_keep = list(subsampled_normal_idx) + list(anomaly_idx)
            self.X = X[idx_to_keep]
            self.y = self.y[idx_to_keep]
            self.labels = self.labels[idx_to_keep]

        self.shape = self.X.shape
        self.anomaly_ratio = (self.y == anomaly_label).sum() / len(self.X)
        self.n_instances = self.X.shape[0]
        self.in_features = self.X.shape[1]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index) -> T_co:
        return self.X[index], self.y[index], self.labels[index]

    def _load_data(self, path: str):
        if path.endswith(".npz"):
            data = np.load(path)[self.npz_key()]
        elif path.endswith(".npy"):
            data = np.load(path)
        elif path.endswith(".mat"):
            data = scipy.io.loadmat(path)
            data = np.concatenate((data['X'], data['y']), axis=1)
        else:
            raise RuntimeError(f"Could not open {path}. Dataset can only read .npz and .mat files.")
        return data

    def get_data_index_by_label(self, label):
        return np.where(self.y == label)[0]

    def loaders(self,
                test_pct: float = 0.5,
                label: int = 0,
                batch_size: int = 128,
                num_workers: int = 0,
                seed: int = None) -> (DataLoader, DataLoader):
        train_set, test_set = self.split_train_test(test_pct, label, seed)
        train_ldr = DataLoader(dataset=train_set, batch_size=batch_size, num_workers=num_workers)
        test_ldr = DataLoader(dataset=test_set, batch_size=batch_size, num_workers=num_workers)
        return train_ldr, test_ldr

    def split_train_test(self, test_pct: float = .5, label: int = 0, seed=None) -> Tuple[Subset, Subset]:
        assert (label == 0 or label == 1)

        if seed:
            torch.manual_seed(seed)

        # Fetch and shuffle indices of the majority class
        maj_data_idx = np.where(self.y == label)[0]
        shuffled_idx = torch.randperm(len(maj_data_idx)).long()

        # Generate training set
        num_test_sample = int(len(maj_data_idx) * test_pct)
        num_train_sample = int(len(maj_data_idx) * (1. - test_pct))
        train_set = Subset(self, maj_data_idx[shuffled_idx[num_train_sample:]])

        # Generate test set based on the remaining data and the previously filtered out labels
        test_idx = np.concatenate([
            maj_data_idx[shuffled_idx[:num_test_sample]],
            np.where(self.y == int(not label))[0]
        ])
        test_set = Subset(self, test_idx)

        return train_set, test_set


class ArrhythmiaDataset(AbstractDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Arrhythmia"

    def npz_key(self):
        return "arrhythmia"


class IDS2017Dataset(AbstractDataset):
    def __init__(self, features: list = None, **kwargs):
        self.selected_features = features
        super(IDS2017Dataset, self).__init__(**kwargs)

    def _load_data(self, path: str):
        df = pd.read_csv(path)
        if self.selected_features:
            df = df[self.selected_features + ["Label", "Category"]]
        self.columns = list(df.columns)
        labels = df["Category"].to_numpy()
        y = df["Label"].astype(np.int8).to_numpy()
        X = df.drop(["Label", "Category"], axis=1).astype(np.float32).to_numpy()
        self.labels = labels
        assert np.isnan(X).sum() == 0, "detected nan values"
        assert X[X < 0].sum() == 0, "detected negative values"
        return np.concatenate(
            (X, np.expand_dims(y, 1)), axis=1
        )


class IDS2018Dataset(AbstractDataset):

    def __init__(self, features: list = None, **kwargs):
        self.selected_features = features
        super().__init__(**kwargs)
        self.name = "IDS2018"

    def _load_data(self, path: str):
        if path.endswith(".npz"):
            return np.load(path)[self.npz_key()]
        else:
            df = pd.read_csv(path)
            if self.selected_features:
                df = df[self.selected_features + ["Label", "Category"]]
            self.columns = list(df.columns)
            labels = df["Category"].to_numpy()
            y = df["Label"].astype(np.int8).to_numpy()
            X = df.drop(["Label", "Category"], axis=1).astype(np.float64).to_numpy()
            self.labels = labels
            assert np.isnan(X).sum() == 0, "detected nan values"
            assert X[X < 0].sum() == 0, "detected negative values"
            return np.concatenate(
                (X, np.expand_dims(y, 1)), axis=1
            )

    def npz_key(self):
        return "ids2018"


class KDD10Dataset(AbstractDataset):
    """
    This class is used to load KDD Cup 10% dataset as a pytorch Dataset
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "KDD10"

    def npz_key(self):
        return "kdd"


class NSLKDDDataset(AbstractDataset):
    """
    This class is used to load NSL-KDD Cup dataset as a pytorch Dataset
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "NSLKDD"

    def npz_key(self):
        return "kdd"


class ThyroidDataset(AbstractDataset):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Thyroid"

    def npz_key(self):
        return "thyroid"


class USBIDSDataset(AbstractDataset):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "USBIDS"

    def npz_key(self):
        return "usbids"


class MalMem2022Dataset(AbstractDataset):

    def __init__(self, **kwargs):
        super(MalMem2022Dataset, self).__init__(**kwargs)
        self.name = "MalMem2022"

    def npz_key(self):
        return "malmem2022"
