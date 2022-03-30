import numpy as np
import scipy.io
import torch
from abc import abstractmethod
from torch.utils.data import Dataset, Subset, DataLoader
from torch.utils.data.dataset import T_co
from typing import Tuple
import scipy.io as scio


class AbstractDataset(Dataset):
    def __init__(self, path: str, pct: float = 1.0, **kwargs):
        self.name = None
        X = self._load_data(path)
        anomaly_label = kwargs.get('anomaly_label', 1)
        normal_label = kwargs.get('normal_label', 0)

        if pct < 1.0:
            # Keeps `pct` percent of the original data while preserving
            # the normal/anomaly ratio
            anomaly_idx = np.where(X[:, -1] == anomaly_label)[0]
            normal_idx = np.where(X[:, -1] == normal_label)[0]
            np.random.shuffle(anomaly_idx)
            np.random.shuffle(normal_idx)

            X = np.concatenate(
                (X[anomaly_idx[:int(len(anomaly_idx) * pct)]],
                 X[normal_idx[:int(len(normal_idx) * pct)]])
            )
            self.X = X[:, :-1]
            self.y = X[:, -1]
        else:
            self.X = X[:, :-1]
            self.y = X[:, -1]

        self.anomaly_ratio = (X[:, -1] == anomaly_label).sum() / len(X)
        self.n_instances = self.X.shape[0]
        self.in_features = self.X.shape[1]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index) -> T_co:
        return self.X[index], self.y[index]

    def _load_data(self, path: str):
        if path.endswith(".npz"):
            return np.load(path)[self.npz_key()]
        elif path.endswith(".mat"):
            data = scipy.io.loadmat(path)
            X = np.concatenate((data['X'], data['y']), axis=1)
            return X
        else:
            raise RuntimeError(f"Could not open {path}. Dataset can only read .npz and .mat files.")

    def D(self):
        return self.X.shape[1]

    def shape(self):
        return self.X.shape

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

        # Fetch and shuffle indices of a single class
        label_data_idx = np.where(self.y == label)[0]
        shuffled_idx = torch.randperm(len(label_data_idx)).long()

        # Generate training set
        num_test_sample = int(len(label_data_idx) * test_pct)
        num_train_sample = int(len(label_data_idx) * (1. - test_pct))
        train_set = Subset(self, label_data_idx[shuffled_idx[num_train_sample:]])

        # Generate test set based on the remaining data and the previously filtered out labels
        remaining_idx = np.concatenate([
            label_data_idx[shuffled_idx[:num_test_sample]],
            np.where(self.y == int(not label))[0]
        ])
        test_set = Subset(self, remaining_idx)

        return train_set, test_set
    #
    # def __init__(self, path: str, pct: float = 1.0, **kwargs):
    #     self.name = None
    #     X = self._load_data(path)
    #     anomaly_label = kwargs.get('anomaly_label', 1)
    #     normal_label = kwargs.get('normal_label', 0)
    #
    #     if pct < 1.0:
    #         # Keeps `pct` percent of the original data while preserving
    #         # the normal/anomaly ratio
    #         anomaly_idx = np.where(X[:, -1] == anomaly_label)[0]
    #         normal_idx = np.where(X[:, -1] == normal_label)[0]
    #         np.random.shuffle(anomaly_idx)
    #         np.random.shuffle(normal_idx)
    #
    #         X = np.concatenate(
    #             (X[anomaly_idx[:int(len(anomaly_idx) * pct)]],
    #              X[normal_idx[:int(len(normal_idx) * pct)]])
    #         )
    #         self.X = X[:, :-1]
    #         self.y = X[:, -1]
    #     else:
    #         self.X = X[:, :-1]
    #         self.y = X[:, -1]
    #
    #     self.anomaly_ratio = (X[:, -1] == anomaly_label).sum() / len(X)
    #     self.n_instances = self.X.shape[0]
    #     self.in_features = self.X.shape[1]
    #
    # def __len__(self):
    #     return len(self.X)
    #
    # def __getitem__(self, index) -> T_co:
    #     return self.X[index], self.y[index]
    #
    # def _load_data(self, path: str):
    #     if path.endswith(".npz"):
    #         return np.load(path)[self.npz_key()]
    #     elif path.endswith(".mat"):
    #         data = scipy.io.loadmat(path)
    #         X = np.concatenate((data['X'], data['y']), axis=1)
    #         return X
    #     else:
    #         raise RuntimeError(f"Could not open {path}. Dataset can only read .npz and .mat files.")
    #
    # def get_shape(self):
    #     return self.X.shape
    #
    # def get_data_index_by_label(self, label):
    #     return np.where(self.y == label)[0]
    #
    # def split_train_test(self, test_perc=.2, seed=None) -> Tuple[Subset, Subset]:
    #     if seed:
    #         torch.manual_seed(seed)
    #     num_test_sample = int(self.n_instances * test_perc)
    #     shuffled_idx = torch.randperm(self.n_instances).long()
    #     train_set = Subset(self, shuffled_idx[num_test_sample:])
    #     test_set = Subset(self, shuffled_idx[:num_test_sample])
    #
    #     return train_set, test_set
    #
    # def one_class_split_train_test(self, test_perc=.2, label=0, seed=None) -> Tuple[Subset, Subset]:
    #     if seed:
    #         torch.manual_seed(seed)
    #
    #     label_data_index = self.get_data_index_by_label(label=label)
    #     num_test_sample = int(len(label_data_index) * test_perc)
    #     shuffled_idx = torch.randperm(len(label_data_index)).long()
    #     train_set = Subset(self, label_data_index[shuffled_idx[num_test_sample:]])
    #
    #     remaining_index = np.concatenate(
    #         [label_data_index[shuffled_idx[:num_test_sample]],
    #          self.get_data_index_by_label(label=0 if label == 1 else 1)]
    #     )
    #
    #     print(f'Size of data with label ={label} :', 100 * len(label_data_index) / self.n_instances)
    #
    #     test_set = Subset(self, remaining_index)
    #
    #     return train_set, test_set
    #
    # def train_test_split(self, test_ratio=.3, label=0, corruption_ratio=0.0, seed=42):
    #     """
    #     This function splits the dataset into training and test datasets. The training set contains only normal data
    #     with a ratio of :corruption_ratio anomalous data.
    #
    #     return: training set and test set
    #     """
    #     if seed:
    #         torch.manual_seed(seed)
    #
    #     # Randomly sample normal data with the corresponding proportion
    #     all_indices = torch.ones(self.n_instances)
    #     label_data_index = self.get_data_index_by_label(label=label)
    #     num_test_sample = int(len(label_data_index) * test_ratio)
    #     shuffled_idx = torch.randperm(len(label_data_index)).long()
    #     label_selected_indices = label_data_index[shuffled_idx[num_test_sample:]]
    #
    #     # Randomly sample anomalous data  with the corresponding proportion
    #     label_inv_data_index = self.get_data_index_by_label(label=1 - label)
    #     num_data_to_inject = int(len(label_data_index) * (1 - test_ratio) * corruption_ratio / (1 - corruption_ratio))
    #
    #     assert num_data_to_inject < len(label_inv_data_index)
    #
    #     shuffled_idx = torch.randperm(len(label_inv_data_index)).long()
    #     label_inv_selected_indices = label_inv_data_index[shuffled_idx[:num_data_to_inject]]
    #
    #     # Merge sampled data as training set
    #     train_indices = np.concatenate([label_selected_indices, label_inv_selected_indices])
    #     train_set = Subset(self, train_indices)
    #
    #     # Consider the remaining data as the test set
    #     all_indices[train_indices] = 0
    #     remaining_index = all_indices.nonzero().squeeze()
    #     print(f"Size of data with label ={label} :", len(label_data_index) / self.n_instances)
    #
    #     test_set = Subset(self, remaining_index)
    #
    #     return train_set, test_set
    #
    # @abstractmethod
    # def npz_key(self):
    #     raise NotImplementedError


class ArrhythmiaDataset(AbstractDataset):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Arrhythmia"

    def npz_key(self):
        return "arrhythmia"


class IDS2018Dataset(AbstractDataset):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "IDS2018"

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
