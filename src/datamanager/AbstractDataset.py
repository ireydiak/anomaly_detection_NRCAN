import numpy as np
from abc import ABC, abstractmethod
import torch
from torch.utils.data import Dataset, Subset
from torch.utils.data.dataset import T_co
from typing import Tuple

ANOMALY_LABEL = 1
NORMAL_LABEL = 0


class AbstractDataset(Dataset):

    def __init__(self, path: str, pct: float = 1.0):
        X = self._load_data(path)

        if pct < 1.0:
            # Keeps `pct` percent of the original data while preserving
            # the normal/anomaly ratio
            anomaly_idx = np.where(X[:, -1] == ANOMALY_LABEL)
            normal_idx = np.where(X[:, -1] == NORMAL_LABEL)
            np.random.shuffle(anomaly_idx)
            np.random.shuffle(normal_idx)
            X = np.concatenate(
                X[anomaly_idx[:int(len(anomaly_idx) * pct), ]],
                X[normal_idx[:int(len(normal_idx) * pct), ]]
            )
            self.X = X[:, :-1]
            self.y = X[:, -1]
        else:
            self.X = X[:, :-1]
            self.y = X[:, -1]

        self.N = len(self.X)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index) -> T_co:
        return self.X[index], self.y[index]

    def _load_data(self, path: str):
        if path.endswith(".npz"):
            return np.load(path)[self.npz_key()]
        else:
            raise RuntimeError(f"Could not open {path}. IDS20189Dataset can only read .npz files.")

    def get_shape(self):
        return self.X.shape

    def get_data_index_by_label(self, label):
        return np.where(self.y == label)[0]

    def split_train_test(self, test_perc=.2, seed=None) -> Tuple[Subset, Subset]:
        if seed:
            torch.manual_seed(seed)
        num_test_sample = int(self.N * test_perc)
        shuffled_idx = torch.randperm(self.N).long()
        train_set = Subset(self, shuffled_idx[num_test_sample:])
        test_set = Subset(self, shuffled_idx[:num_test_sample])

        return train_set, test_set

    def one_class_split_train_test(self, test_perc=.2, label=0, seed=None) -> Tuple[Subset, Subset]:
        if seed:
            torch.manual_seed(seed)

        label_data_index = self.get_data_index_by_label(label=label)
        num_test_sample = int(len(label_data_index) * test_perc)
        shuffled_idx = torch.randperm(len(label_data_index)).long()
        train_set = Subset(self, label_data_index[shuffled_idx[num_test_sample:]])

        remaining_index = np.concatenate(
            [label_data_index[shuffled_idx[:num_test_sample]],
             self.get_data_index_by_label(label=0 if label == 1 else 1)]
        )

        print(f'Size of data with label ={label} :', 100 * len(label_data_index) / self.N)

        test_set = Subset(self, remaining_index)

        return train_set, test_set

    @abstractmethod
    def npz_key(self):
        raise NotImplementedError
