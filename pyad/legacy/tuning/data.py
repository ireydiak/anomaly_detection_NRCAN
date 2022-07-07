import torch
import numpy as np
from typing import Tuple
from torch.utils.data import DataLoader, Subset

from pyad.datamanager.dataset import AbstractDataset


def get_data_loaders(
        dataset: AbstractDataset,
        batch_size,
        test_size=0.25,
        val_size=0.25
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    X, y, anomaly_ratio = dataset.X, dataset.y, dataset.anomaly_ratio
    # Compute train_size from test_size and val_size
    train_size = 1. - test_size - val_size
    assert test_size + val_size <= 1., "`test_size` and `val_size` must not exceed or equal 1."
    # Convert percentages to numbers
    train_size = int(len(dataset) * train_size)
    test_size = int(len(dataset) * test_size)
    val_size = int(len(dataset) * val_size)
    # Leave remaining data to train_set
    underflow = len(dataset) - (train_size + test_size + val_size)
    if underflow > 0:
        train_size += underflow
    # Split normal and abnormal indexes
    normal_idx = np.where(y == 0)[0]
    abnormal_idx = np.where(y == 1)[0]
    anomaly_idx = int(len(abnormal_idx) * .5)
    # Shuffle indexes
    normal_idx = torch.randperm(len(normal_idx)).long()
    abnormal_idx = torch.randperm(len(abnormal_idx)).long()
    # Create train set
    train_set = DataLoader(
        Subset(dataset, normal_idx[0:train_size]),
        batch_size=batch_size
    )
    # Create test set
    test_idx = np.concatenate((
        normal_idx[train_size:train_size + test_size], abnormal_idx[anomaly_idx:]
    ))
    test_set = DataLoader(
        Subset(dataset, test_idx),
        batch_size=batch_size
    )
    # Create val set
    val_idx = np.concatenate((
        normal_idx[train_size + test_size:], abnormal_idx[0:anomaly_idx],
    ))
    val_set = DataLoader(
        Subset(dataset, val_idx),
        batch_size=batch_size
    )
    return train_set, val_set, test_set
