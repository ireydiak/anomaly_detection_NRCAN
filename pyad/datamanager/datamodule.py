import numpy
import pandas as pd
import pytorch_lightning as pl
from typing import Optional, Tuple
from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY
import scipy.io as scio
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
from torch.utils.data.dataloader import T_co

from pyad.datamanager.dataset import SimpleDataset, scaler_map


class BaseDataset(Dataset):
    def __init__(self, root: str, label_col: str = None):
        super(BaseDataset, self).__init__()
        self.root = root
        self.labels = None
        data = self._load_data(self.root, label_col)
        self.X = data[:, :-1]
        self.y = data[:, -1]
        assert np.isnan(self.X).sum() == 0, "detected nan values"
        if self.labels is None or self.labels.size == 0:
            self.labels = self.y
        self.n_instances = self.X.shape[0]
        self.in_features = self.X.shape[1]

    def __getitem__(self, index) -> T_co:
        return self.X[index], self.y[index], self.labels[index]

    def __len__(self):
        return len(self.X)

    def train_test_split(self, test_size: float = 0.5, normal_label: int = 0) -> Tuple[Subset, Subset]:
        anomaly_label = int(1 - normal_label)

        # split between normal and abnormal samples
        normal_idx = np.where(self.y == normal_label)[0]
        abnormal_idx = np.where(self.y == anomaly_label)[0]

        # shuffle normal indexes for more randomness
        np.random.shuffle(normal_idx)

        # select training data
        n_train_normal = int(len(normal_idx) * (1 - test_size))
        train_idx = Subset(self, normal_idx[:n_train_normal])

        # select test data
        test_normal_idx = normal_idx[n_train_normal:]
        test_idx = np.concatenate((
            test_normal_idx,
            abnormal_idx
        ))
        test_idx = Subset(self, test_idx)

        return train_idx, test_idx

    def _load_data(self, path: str, label_col: str = None) -> numpy.ndarray:
        if path.endswith(".npz"):
            data = np.load(path)
        elif path.endswith(".npy"):
            data = np.load(path)
        elif path.endswith(".mat"):
            data = scio.loadmat(path)
            data = np.concatenate((data['X'], data['y']), axis=1)
        elif path.endswith(".csv"):
            df = pd.read_csv(path)
            if label_col:
                self.labels = df[label_col].to_numpy()
                data = df.drop(["Category"], axis=1).astype(np.float32).to_numpy()
            else:
                self.labels = None
                data = df.astype(np.float32).to_numpy()
        else:
            raise RuntimeError(f"Could not open {path}. Dataset can only read .npz and .mat files.")
        return data


class BaseDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_dir: str,
            batch_size: int,
            scaler: str = None,
            num_workers: int = 0,
            label_col: str = None
    ):
        super(BaseDataModule, self).__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = BaseDataset(self.data_dir, label_col=label_col)
        self.sanity_check()
        self.in_features = self.dataset.in_features
        self.n_instances = self.dataset.n_instances
        self.trainset = None
        self.testset = None
        self.scaler = scaler_map[scaler]() if scaler else None

    def sanity_check(self) -> None:
        """
        Optional hook to run sanity checks (e.g. look for NaNs, INF and other invalid values)
        prior to train any model
        """
        pass

    def normalize(self, train_set, test_set):
        # extract training/testing data and labels
        train_data = train_set.dataset.X[train_set.indices]
        train_y, train_labels = train_set.dataset.y[train_set.indices], train_set.dataset.labels[train_set.indices]
        test_data = test_set.dataset.X[test_set.indices]
        test_y, test_labels = test_set.dataset.y[test_set.indices], test_set.dataset.labels[test_set.indices]
        # fit scaler on train set only (to avoid data leaks)
        self.scaler.fit(train_data)
        # transform the data
        train_set = SimpleDataset(X=self.scaler.transform(train_data), y=train_y, labels=train_labels)
        test_set = SimpleDataset(X=self.scaler.transform(test_data), y=test_y, labels=test_labels)
        return train_set, test_set

    def setup(self, stage: Optional[str] = None):
        trainset, testset = self.dataset.train_test_split()
        if self.scaler:
            trainset, testset = self.normalize(trainset, testset)
        self.trainset = DataLoader(
            trainset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True
        )
        self.testset = DataLoader(
            testset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True
        )

    def train_dataloader(self):
        if not self.trainset:
            self.setup()
        return self.trainset

    def test_dataloader(self):
        if not self.testset:
            self.setup()
        return self.testset


@DATAMODULE_REGISTRY
class ThyroidDataModule(BaseDataModule):
    pass


@DATAMODULE_REGISTRY
class ArrhythmiaDataModule(BaseDataModule):
    pass


@DATAMODULE_REGISTRY
class KDD10DataModule(BaseDataModule):
    def sanity_check(self):
        X = self.dataset.X
        assert X[X < 0].sum() == 0, "detected negative values"


@DATAMODULE_REGISTRY
class NSLKDDDataModule(BaseDataModule):
    def sanity_check(self):
        X = self.dataset.X
        assert X[X < 0].sum() == 0, "detected negative values"


@DATAMODULE_REGISTRY
class IDS2017DataModule(BaseDataModule):
    def sanity_check(self):
        X = self.dataset.X
        assert X[X < 0].sum() == 0, "detected negative values"


@DATAMODULE_REGISTRY
class IDS2018DataModule(BaseDataModule):
    def sanity_check(self):
        X = self.dataset.X
        assert X[X < 0].sum() == 0, "detected negative values"
