import numpy as np
import scipy.io
import torch
from torch.utils.data import Dataset, Subset, DataLoader
from torch.utils.data.dataset import T_co
from typing import Tuple

from utils.utils import random_split_to_two


class AbstractDataset(Dataset):
    def __init__(self, path: str, pct: float = 1.0, **kwargs):
        self.name = self.__class__.__name__
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
                holdout: float = 0.0,
                contamination_rate: float = 0.0,
                validation_ratio: float = .2,
                batch_size: int = 128,
                num_workers: int = 0,
                seed: int = None,
                drop_last_batch: bool = False) -> (DataLoader, DataLoader, DataLoader):

        train_set, test_set, val_set = self.split_train_test(test_pct=test_pct,
                                                             label=label,
                                                             holdout=holdout,
                                                             contamination_rate=contamination_rate,
                                                             validation_ratio=validation_ratio,
                                                             seed=seed)

        train_ldr = DataLoader(dataset=train_set, batch_size=batch_size, num_workers=num_workers,
                               drop_last=drop_last_batch)

        val_ldr = DataLoader(dataset=val_set, batch_size=batch_size, num_workers=num_workers,
                             drop_last=drop_last_batch)

        test_ldr = DataLoader(dataset=test_set, batch_size=batch_size, num_workers=num_workers,
                              drop_last=drop_last_batch)
        return train_ldr, test_ldr, val_ldr

    def split_train_test(self, test_pct: float = .5,
                         label: int = 0,
                         holdout=0.05,
                         contamination_rate=0.01,
                         validation_ratio: float = .2,
                         seed=None, debug=True) -> Tuple[Subset, Subset, Subset]:
        assert (label == 0 or label == 1)
        assert 1 > holdout  # >=
        assert 0 <= contamination_rate <= 1

        # if seed:
        #     torch.manual_seed(seed)

        # Fetch and shuffle indices of a single class
        normal_data_idx = np.where(self.y == label)[0]
        shuffled_norm_idx = torch.randperm(len(normal_data_idx)).long()

        # Generate training set indices
        num_norm_test_sample = int(len(normal_data_idx) * test_pct)
        num_norm_train_sample = int(len(normal_data_idx) * (1. - test_pct))
        normal_train_idx = normal_data_idx[shuffled_norm_idx[num_norm_train_sample:]]

        #
        abnormal_data_idx = np.where(self.y == int(not label))[0]
        abnorm_test_idx = abnormal_data_idx

        if debug:
            print(f"Dataset size\nPositive class :{len(abnormal_data_idx)}"
                  f"\nNegative class :{len(normal_data_idx)}\n")

        if holdout > 0:
            # Generate test set by holding out a percentage [holdout] of abnormal
            # sample for a possible contamination
            shuffled_abnorm_idx = torch.randperm(len(abnormal_data_idx)).long()
            num_abnorm_test_sample = int(len(abnormal_data_idx) * (1 - holdout))
            abnorm_test_idx = abnormal_data_idx[shuffled_abnorm_idx[:num_abnorm_test_sample]]

            if contamination_rate > 0:
                num_abnorm_to_inject = int(len(shuffled_abnorm_idx[
                                               num_abnorm_test_sample:]) * contamination_rate)

                normal_train_idx = np.concatenate([
                    abnormal_data_idx[shuffled_abnorm_idx[
                                      num_abnorm_test_sample:
                                      num_abnorm_test_sample + num_abnorm_to_inject]],
                    normal_train_idx
                ])

        # Generate training set with contamination when applicable
        # Split the training set to train and validation
        normal_train_idx, normal_val_idx = random_split_to_two(normal_train_idx, ratio=validation_ratio)

        train_set = Subset(self, normal_train_idx)
        val_set = Subset(self, normal_val_idx)
        if debug:
            print(
                f'Training set\n'
                f'Contamination rate: '
                f'{len(np.where(self.y[normal_train_idx] == int(not label))[0]) / len(normal_train_idx)}\n')

        # Generate test set based on the remaining data and the previously filtered out labels
        remaining_idx = np.concatenate([
            normal_data_idx[shuffled_norm_idx[:num_norm_test_sample]],
            abnorm_test_idx
        ])
        test_set = Subset(self, remaining_idx)

        print(len(self.y) , (len(test_set) + len(train_set) + len(val_set)))

        return train_set, test_set, val_set

    def split_train_test_(self, test_pct: float = .5, label: int = 0, seed=None) -> Tuple[Subset, Subset]:
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


class MalMem2022Dataset(AbstractDataset):

    def __init__(self, **kwargs):
        super(MalMem2022Dataset, self).__init__(**kwargs)
        self.name = "MalMem2022"

    def npz_key(self):
        return "malmem2022"
