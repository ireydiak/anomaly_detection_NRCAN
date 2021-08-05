import os

import torch
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, Subset, Sampler
from torch.utils.data.dataset import T_co
import pandas as pd
import numpy as np
from sklearn import preprocessing


class NSLKDDDataset(Dataset):
    """
    This class is used to load NSL-KDD Cup dataset as a pytorch Dataset
    """
    name = 'NSLKDD'

    def __init__(self, path='../data/kddcup_data', pct: float=1.0):
        self.path = path

        data = self._load_data(path)

        # Keep `pct` percent of the original data
        # Extract labels and features in two separate arrays
        self.X, self.y = data[:, :-1], data[:, -1]

        if pct < 1.0:
            np.random.shuffle(self.X)
            self.X = self.X[0: int(len(self.X) * pct), :-1]
            self.y = self.X[0: int(len(self.X) * pct), -1]

        self.N = len(data)

    def __len__(self):
        return self.N

    def __getitem__(self, index) -> T_co:
        return self.X[index], self.y[index]

    def get_data_index_by_label(self, label):
        indices = np.array([i for i in range(self.N)])
        return indices[self.y == label]

    def _load_data(self, path):
        return np.load(path, allow_pickle=True)["kdd"]

    def split_train_test(self, test_perc=.2, seed=0):
        # torch.manual_seed(seed)
        num_test_sample = int(self.N * test_perc)
        shuffled_idx = torch.randperm(self.N).long()
        train_set = Subset(self, shuffled_idx[num_test_sample:])
        test_set = Subset(self, shuffled_idx[:num_test_sample])

        stat = {'l1': (self.y[shuffled_idx[num_test_sample:]] == 1).sum() / (self.N - num_test_sample),
                'l0': (self.y[shuffled_idx[num_test_sample:]] == 0).sum() / (self.N - num_test_sample)}

        print(f'Percentage of label 0:{stat["l0"]}'
              f'\nPercentage of label 1:{stat["l1"]}')

        # stat = {'l1': (self.y == 1).sum() / (self.N),
        #         'l0': (self.y == 0).sum() / (self.N)}
        #
        # print(f'Percentage of label 0:{stat["l0"]}'
        #       f'\nPercentage of label 1:{stat["l1"]}')
        return train_set, test_set

    def one_class_split_train_test(self, test_perc=.2, label=0, seed=0):
        label_data_index = self.get_data_index_by_label(label=label)
        num_test_sample = int(len(label_data_index) * test_perc)
        shuffled_idx = torch.randperm(len(label_data_index)).long()
        train_set = Subset(self, label_data_index[shuffled_idx[num_test_sample:]])

        remaining_index = np.concatenate([label_data_index[shuffled_idx[:num_test_sample]],
                                          self.get_data_index_by_label(label=0 if label == 1 else 1)])

        print(f'Size of data with label ={label} :', 100 * len(label_data_index) / self.N)

        test_set = Subset(self, remaining_index)
        return train_set, test_set

    def get_shape(self):
        return self.X.shape
