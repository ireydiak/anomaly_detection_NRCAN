import numpy as np
import pandas as pd
import scipy.io
import torch
from ray import tune as ray_tune
from torch.utils.data import Dataset, Subset, DataLoader
from torch.utils.data.dataset import T_co
from typing import Tuple
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pyad.utils.utils import random_split_to_two
from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY

scaler_map = {
    "minmax": MinMaxScaler,
    "minmaxscaler": MinMaxScaler,
    "standard": StandardScaler,
    "standardscaler": StandardScaler
}


class SimpleDataset(Dataset):
    def __init__(self, X, y, labels=None):
        self.X = X
        self.y = y
        self.labels = labels
        if self.labels is None:
            self.labels = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index) -> T_co:
        return self.X[index], self.y[index], self.labels[index]


class AbstractDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        normal_size: float = 1.0,
        seed=None,
        scaler=None,
        normal_label: int = 0,
        anomaly_label: int = 1,
    ):
        super(AbstractDataset, self).__init__()
        self.normal_label = normal_label
        self.anomaly_label = anomaly_label
        self.batch_size = batch_size
        self.name = self.__class__.__name__
        self.labels = np.array([])
        self.seed = seed
        if scaler is not None:
            assert scaler.lower() in set(scaler_map.keys()), "unknown scaler %s, please use %s" % (scaler, scaler_map.keys())
            self.scaler = scaler_map[scaler.lower()]()
        else:
            self.scaler = None
        data = self._load_data(data_dir)
        if normal_size < 1.:
            self.X, self.y = self.select_data_subset(normal_size, data)
        else:
            self.X = data[:, :-1]
            self.y = data[:, -1]
        if self.labels.size == 0:
            self.labels = self.y

        self.shape = self.X.shape
        self.anomaly_ratio = (self.y == 1).sum() / len(self.X)
        self.n_instances = self.X.shape[0]
        self.in_features = self.X.shape[1]

    def select_data_subset(self, normal_size, data):
        # Keeps `normal_size` percent of normal labels
        normal_idx = np.where(data[:, -1] == self.normal_label)[0]
        anomaly_idx = np.where(data[:, -1] == self.anomaly_label)[0]

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

    @staticmethod
    def get_default_cfg():
        return {
            "batch_size": 4,
            "normal_size": 1,
            "holdout": 0,
            "val_ratio": 0,
            "contamination_rate": 0,
            "path": "",
            "n_runs": 1
        }

    def _load_data(self, path: str):
        if path.endswith(".npz"):
            data = np.load(path, allow_pickle=True)
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

    def split_train_test_legacy(self,
                                batch_size,
                                test_pct: float = 0.5,
                                label: int = 0,
                                holdout: float = 0.0,
                                contamination_rate: float = 0.0,
                                validation_ratio: float = 0.,
                                num_workers: int = 0,
                                seed: int = None,
                                drop_last_batch: bool = False) -> (DataLoader, DataLoader, DataLoader):

        train_set, test_set, val_set = self.split_train_test_legacy(test_pct=test_pct,
                                                                    label=label,
                                                                    holdout=holdout,
                                                                    contamination_rate=contamination_rate,
                                                                    validation_ratio=validation_ratio,
                                                                    seed=seed)
        train_ldr = DataLoader(
            dataset=train_set,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True
        )

        val_ldr = DataLoader(dataset=val_set, batch_size=batch_size, num_workers=num_workers,
                             drop_last=drop_last_batch)

        test_ldr = DataLoader(
            dataset=test_set,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=drop_last_batch
        )
        return train_ldr, test_ldr, val_ldr

    def loaders(self,
                batch_size,
                test_size: float = 0.5,
                normal_label: int = 0,
                num_workers: int = 0,
                drop_last: bool = False
                ) -> (DataLoader, DataLoader):
        train_set, test_set = self.train_test_split(test_size=test_size, normal_label=normal_label)
        if self.scaler:
            train_set, test_set, _ = self.normalize(train_set, test_set)

        train_ldr = DataLoader(
            dataset=train_set,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=drop_last
        )

        test_ldr = DataLoader(
            dataset=test_set,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True
        )
        return train_ldr, test_ldr, None

    def normalize(self, train_set, test_set, val_set=None):
        # if train_set is a Subset
        if hasattr(train_set, "indices"):
            train_data = train_set.dataset.X[train_set.indices]
            train_y, train_labels = train_set.dataset.y[train_set.indices], train_set.dataset.labels[train_set.indices]
            test_data = test_set.dataset.X[test_set.indices]
            test_y, test_labels = test_set.dataset.y[test_set.indices], test_set.dataset.labels[test_set.indices]
        else:
            train_data = train_set.X
            train_y, train_labels = train_set.y, train_set.labels
            test_data = test_set.X
            test_y, test_labels = test_set.y, test_set.labels
        # with with data leak to evaluate difference
        #self.scaler.fit(np.concatenate((train_data, train_data)))
        # fit scaler on train set only (to avoid data leaks)
        self.scaler.fit(train_data)
        # transform the data
        train_set = SimpleDataset(X=self.scaler.transform(train_data), y=train_y, labels=train_labels)
        test_set = SimpleDataset(X=self.scaler.transform(test_data), y=test_y, labels=test_labels)
        # optionally transform the validation set
        if val_set is not None and len(val_set) > 0:
            # TODO: handle the case where validation set is not a Subset
            val_data = val_set.dataset.X[val_set.indices]
            val_y, val_labels = val_set.dataset.y[val_set.indices], val_set.dataset.labels[val_set.indices]
            val_set = SimpleDataset(X=self.scaler.transform(val_data), y=val_y, labels=val_labels)
        return train_set, test_set, val_set

    def train_test_split(self, test_size: float = 0.5, normal_label: int = 0) -> Tuple[Subset, Subset]:
        anomaly_label = int(1 - normal_label)

        # split between normal and abnormal samples
        normal_idx = np.where(self.y == normal_label)[0]
        abnormal_idx = np.where(self.y == anomaly_label)[0]

        # shuffle normal indexes for more randomness
        np.random.shuffle(normal_idx)

        # select training data
        n_train_normal = int(len(normal_idx) * (1 - test_size))
        train_X = self.X[normal_idx[:n_train_normal]]
        train_y = self.y[normal_idx[:n_train_normal]]
        train_dataset = SimpleDataset(X=train_X, y=train_y)  # Subset(self, normal_idx[:n_train_normal])

        # select test data
        test_normal_idx = normal_idx[n_train_normal:]
        test_idx = np.concatenate((
            test_normal_idx,
            abnormal_idx
        ))
        test_dataset = SimpleDataset(X=self.X[test_idx], y=self.y[test_idx])  # Subset(self, test_idx)

        assert (train_dataset.y == anomaly_label).sum() == 0, "found anomalies in the training set, aborting"
        assert (test_dataset.y == anomaly_label).sum() > 0, "no anomalies found in test set, aborting"
        return train_dataset, test_dataset

    def split_train_test_legacy(self,
                                test_pct: float = .5,
                                label: int = 0,
                                holdout=0.00,
                                contamination_rate=0.,
                                validation_ratio: float = 0.,
                                seed=None,
                                debug=False) -> Tuple[Subset, Subset, Subset]:
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


@DATAMODULE_REGISTRY
class ArrhythmiaDataset(AbstractDataset):
    name = "Arrhythmia"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Arrhythmia"

    @staticmethod
    def get_tunable_params():
        return {
            "batch_size": ray_tune.choice([4, 16, 32, 64])
        }


class IDS2017Dataset(AbstractDataset):
    name = "IDS2017"

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
    name = "IDS2018"

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


@DATAMODULE_REGISTRY
class KDD10Dataset(AbstractDataset):
    """
    This class is used to load KDD Cup 10% dataset as a pytorch Dataset
    """
    name = "KDD10"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "KDD10"

    @staticmethod
    def get_tunable_params():
        return {
            "batch_size": ray_tune.choice([16, 32, 64, 128, 1024])
        }


@DATAMODULE_REGISTRY
class NSLKDDDataset(AbstractDataset):
    """
    This class is used to load NSL-KDD Cup dataset as a pytorch Dataset
    """
    name = "NSLKDD"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "NSLKDD"

    @staticmethod
    def get_tunable_params():
        return {
            "batch_size": ray_tune.choice([16, 32, 64, 128])
        }


@DATAMODULE_REGISTRY
class ThyroidDataset(AbstractDataset):
    name = "Thyroid"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Thyroid"

    @staticmethod
    def get_tunable_params():
        return {
            "batch_size": ray_tune.choice([16, 32, 64, 128])
        }


class USBIDSDataset(AbstractDataset):
    name = "USBIDS"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "USBIDS"

    @staticmethod
    def get_tunable_params():
        return {
            "batch_size": ray_tune.choice([64, 128, 1024])
        }


class MalMem2022Dataset(AbstractDataset):
    name = "MalMem2022"

    def __init__(self, **kwargs):
        super(MalMem2022Dataset, self).__init__(**kwargs)
        self.name = "MalMem2022"

    @staticmethod
    def get_tunable_params():
        return {
            "batch_size": ray_tune.choice([64, 128, 1024])
        }
