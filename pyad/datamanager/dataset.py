import numpy as np
import pandas as pd
import scipy.io
import torch
from ray import tune as ray_tune
from torch.utils.data import Dataset, Subset, DataLoader
from torch.utils.data.dataset import T_co
from typing import Tuple

from pyad.utils.utils import random_split_to_two


class AbstractDataset(Dataset):
    def __init__(self, path: str, normal_size: float = 1.0, seed=None, **kwargs):
        self.name = self.__class__.__name__
        self.labels = np.array([])
        self.seed = seed

        data = self._load_data(path)
        if normal_size < 1.:
            self.X, self.y = self.select_data_subset(normal_size, data, **kwargs)
        else:
            self.X = data[:, :-1]
            self.y = data[:, -1]
        if self.labels.size == 0:
            self.labels = self.y

        self.shape = self.X.shape
        self.anomaly_ratio = (self.y == 1).sum() / len(self.X)
        self.n_instances = self.X.shape[0]
        self.in_features = self.X.shape[1]

    def select_data_subset(self, normal_size, data, **kwargs):
        anomaly_label = kwargs.get('anomaly_label', 1)
        normal_label = kwargs.get('normal_label', 0)

        # Keeps `normal_size` percent of normal labels
        normal_idx = np.where(data[:, -1] == normal_label)[0]
        anomaly_idx = np.where(data[:, -1] == anomaly_label)[0]

        if self.seed:
            np.random.seed(self.seed)
        np.random.shuffle(normal_idx)

        subsampled_normal_idx = normal_idx[int(len(normal_idx) * normal_size):]
        idx_to_keep = list(subsampled_normal_idx) + list(anomaly_idx)
        X = data[idx_to_keep]

        return X[:, :-1], X[:, -1]

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
                holdout: float = 0.0,
                contamination_rate: float = 0.0,
                validation_ratio: float = 0.,
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

    def split_train_test(self,
                         test_pct: float = .5,
                         label: int = 0,
                         holdout=0.05,
                         contamination_rate=0.,
                         validation_ratio: float = 0.,
                         seed=None,
                         debug=True) -> Tuple[Subset, Subset, Subset]:
        assert (label == 0 or label == 1)
        assert 1 > holdout
        assert 0 <= contamination_rate <= 1

        if seed:
            torch.manual_seed(seed)

        # Fetch and shuffle indices of the majority class
        maj_data_idx = np.where(self.y == label)[0]
        shuffled_idx = torch.randperm(len(maj_data_idx)).long()

        # Generate training set
        num_test_sample = int(len(maj_data_idx) * test_pct)

        # Generate test set based on the remaining data and the previously filtered out labels
        test_idx = np.concatenate([
            maj_data_idx[shuffled_idx[:num_test_sample]],
            np.where(self.y == int(not label))[0]
        ])
        # Fetch and shuffle indices of a single class
        normal_data_idx = np.where(self.y == label)[0]
        shuffled_norm_idx = torch.randperm(len(normal_data_idx)).long()

        # Generate training set indices
        num_norm_train_sample = int(len(normal_data_idx) * (1. - test_pct))
        normal_train_idx = normal_data_idx[shuffled_norm_idx[num_norm_train_sample:]]
        abnormal_data_idx = np.where(self.y == int(not label))[0]

        if debug:
            print(f"Dataset size\nPositive class: {len(abnormal_data_idx)}"
                  f"\nNegative class: {len(normal_data_idx)}\n")

        if holdout > 0:
            # Generate test set by holding out a percentage [holdout] of abnormal data
            # sample for a possible contamination
            shuffled_abnorm_idx = torch.randperm(len(abnormal_data_idx)).long()
            num_abnorm_test_sample = int(len(abnormal_data_idx) * (1 - holdout))

            if contamination_rate > 0:
                num_abnorm_to_inject = int(normal_train_idx.shape[0] * contamination_rate / (1 - contamination_rate))

                assert num_abnorm_to_inject <= len(shuffled_abnorm_idx[num_abnorm_test_sample:])

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

        test_set = Subset(self, test_idx)

        return train_set, test_set, val_set

    @staticmethod
    def get_tunable_params():
        raise NotImplementedError


class ArrhythmiaDataset(AbstractDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Arrhythmia"

    def npz_key(self):
        return "arrhythmia"

    @staticmethod
    def get_tunable_params():
        return {
            "batch_size": ray_tune.choice([4, 16, 32, 64])
        }


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

    @staticmethod
    def get_tunable_params():
        return {
            "batch_size": ray_tune.choice([64, 128, 1024])
        }


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
            # Select 50% of normal samples
            normal_df = df[df.Label == 0].sample(frac=.5, random_state=self.seed)
            df = pd.concat((
                normal_df, df[df.Label == 1]
            ))
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

    def split_train_test(self,
                         test_pct: float = .5,
                         label: int = 0,
                         holdout: float = 0.,
                         contamination_rate: float = 0.,
                         validation_ratio: float = 0.,
                         seed=None,
                         debug=True,
                         corruption_label=None) -> Tuple[Subset, Subset, Subset]:
        assert (label == 0 or label == 1)
        assert 0 <= holdout <= 1, "`holdout` should be inclusively between 0 and 1"
        assert 0 <= contamination_rate <= 1

        if seed:
            torch.manual_seed(seed)

        # Fetch and shuffle indices of a single class
        normal_data_idx = np.where(self.y == label)[0]
        shuffled_norm_idx = torch.randperm(len(normal_data_idx)).long()
        val_set = Subset(self, [])

        # Generate training set indices
        num_norm_test_sample = int(len(normal_data_idx) * test_pct)
        num_norm_train_sample = int(len(normal_data_idx) * (1. - test_pct))
        normal_train_idx = normal_data_idx[shuffled_norm_idx[num_norm_train_sample:]]

        abnormal_data_idx = np.where(self.y == int(not label))[0]

        if debug:
            print(f"Dataset size\nPositive class :{len(abnormal_data_idx)}"
                  f"\nNegative class :{len(normal_data_idx)}\n")

        if holdout > 0:
            # Generate test set by holding out a percentage [holdout] of abnormal data
            # sample for a possible contamination
            shuffled_abnorm_idx = torch.randperm(len(abnormal_data_idx)).long()
            num_abnorm_test_sample = int(len(abnormal_data_idx) * (1 - holdout))
            abnorm_test_idx = abnormal_data_idx[shuffled_abnorm_idx[:num_abnorm_test_sample]]

            if contamination_rate > 0:
                holdout_ano_idx = abnormal_data_idx[shuffled_abnorm_idx[num_abnorm_test_sample:]]

                # Injection of only specified type of attacks
                if corruption_label:
                    all_labels = np.char.lower(self.labels[holdout_ano_idx].astype('str'))
                    corruption_label = corruption_label.lower()
                    corruption_by_lbl_idx = np.char.startswith(all_labels,
                                                               corruption_label)
                    holdout_ano_idx = holdout_ano_idx[corruption_by_lbl_idx]

                # Calculate the number of abnormal samples to inject
                # according to the contamination rate
                num_abnorm_to_inject = int(normal_train_idx.shape[0] * contamination_rate / (1 - contamination_rate))

                assert num_abnorm_to_inject <= len(holdout_ano_idx)

                normal_train_idx = np.concatenate([
                    holdout_ano_idx[:num_abnorm_to_inject],
                    normal_train_idx
                ])

        if validation_ratio > 0.:
            # Generate training set with contamination when applicable
            # Split the training set to train and validation
            normal_train_idx, normal_val_idx = random_split_to_two(normal_train_idx, ratio=validation_ratio)
            val_set = Subset(self, normal_val_idx)

        train_set = Subset(self, normal_train_idx)
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

        return train_set, test_set, val_set

    def __getitem__(self, index) -> T_co:
        return self.X[index], self.y[index], self.labels[index]

    @staticmethod
    def get_tunable_params():
        return {
            "batch_size": ray_tune.choice([64, 128, 1024])
        }


class KDD10Dataset(AbstractDataset):
    """
    This class is used to load KDD Cup 10% dataset as a pytorch Dataset
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "KDD10"

    def npz_key(self):
        return "kdd"

    @staticmethod
    def get_tunable_params():
        return {
            "batch_size": ray_tune.choice([16, 32, 64, 128, 1024])
        }


class NSLKDDDataset(AbstractDataset):
    """
    This class is used to load NSL-KDD Cup dataset as a pytorch Dataset
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "NSLKDD"

    def npz_key(self):
        return "kdd"

    @staticmethod
    def get_tunable_params():
        return {
            "batch_size": ray_tune.choice([16, 32, 64, 128])
        }


class ThyroidDataset(AbstractDataset):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Thyroid"

    def npz_key(self):
        return "thyroid"

    @staticmethod
    def get_tunable_params():
        return {
            "batch_size": ray_tune.choice([16, 32, 64, 128])
        }


class USBIDSDataset(AbstractDataset):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "USBIDS"

    def npz_key(self):
        return "usbids"

    @staticmethod
    def get_tunable_params():
        return {
            "batch_size": ray_tune.choice([64, 128, 1024])
        }


class MalMem2022Dataset(AbstractDataset):

    def __init__(self, **kwargs):
        super(MalMem2022Dataset, self).__init__(**kwargs)
        self.name = "MalMem2022"

    def npz_key(self):
        return "malmem2022"

    @staticmethod
    def get_tunable_params():
        return {
            "batch_size": ray_tune.choice([64, 128, 1024])
        }
