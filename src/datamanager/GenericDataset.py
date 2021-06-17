import os

import torch
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, Subset, Sampler
from torch.utils.data.dataset import T_co
import pandas as pd
import numpy as np
from sklearn import preprocessing


class GenericDataset(Dataset):
    """
    This class is used to load KDD Cup dataset as a pytorch Dataset
    """

    def __init__(self, path='../data/kddcup_data'):
        self.path = path

        # load data
        # if os.path.exists('../data/kdd_cup.npz'):
        #     data = np.load('../data/kdd_cup.npz')['kdd']
        # else:
        data = self._load_data(path)

        self.X, self.y = data[:, :-1], data[:, -1]

        self.n = len(data)

    def __len__(self):
        return self.n

    def __getitem__(self, index) -> T_co:
        return self.X[index], self.y[index]

    def get_data_index_by_label(self, label):
        indices = np.array([i for i in range(self.n)])
        return indices[self.y == label]

    def _load_data(self, path):

        print(path)

        column_names = names = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
                                'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
                                'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
                                'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
                                'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
                                'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
                                'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
                                'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                                'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'type']

        df = pd.read_csv(path, header=None,)  # names=column_names)

        # regroup all the abnormal data to attack
        df['type'] = df['type'].map(lambda x: 1 if x == 'normal.' else 0)

        # transform categorical variables to one hot

        one_hot_protocol = pd.get_dummies(df["protocol_type"])
        one_hot_service = pd.get_dummies(df["service"])
        one_hot_flag = pd.get_dummies(df["flag"])

        df = df.drop("protocol_type", axis=1)
        df = df.drop("service", axis=1)
        df = df.drop("flag", axis=1)

        df = pd.concat([one_hot_protocol, one_hot_service, one_hot_flag, df], axis=1)

        proportions = df["type"].value_counts()
        print(f'Counts per class:\n {proportions}\n')
        print("Anomaly Percentage", proportions[0] / proportions.sum())

        cols_to_norm = ["duration", "src_bytes", "dst_bytes", "wrong_fragment", "urgent",
                        "hot", "num_failed_logins", "num_compromised", "num_root",
                        "num_file_creations", "num_shells", "num_access_files", "count", "srv_count",
                        "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
                        "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
                        "dst_host_same_srv_rate",
                        "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
                        "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
                        "dst_host_srv_rerror_rate"]

        # Normalize data
        min_cols = df.loc[df["type"] == 0, cols_to_norm].min()
        max_cols = df.loc[df["type"] == 0, cols_to_norm].max()

        df.loc[:, cols_to_norm] = (df[cols_to_norm] - min_cols) / (max_cols - min_cols)

        kdd_numpy = np.array(df, dtype="float32")

        # np.savez('../data/kdd_cup.npz', kdd=kdd_numpy)

        return kdd_numpy

    def split_train_test(self, test_perc=.2, seed=0):
        # torch.manual_seed(seed)
        num_test_sample = int(self.n * test_perc)
        shuffled_idx = torch.randperm(self.n).long()
        train_set = Subset(self, shuffled_idx[num_test_sample:])
        test_set = Subset(self, shuffled_idx[:num_test_sample])

        return train_set, test_set

    def one_class_split_train_test(self, test_perc=.2, label=0, seed=0):
        # torch.manual_seed(seed)
        #
        label_data_index = self.get_data_index_by_label(label=label)
        num_test_sample = int(len(label_data_index) * test_perc)
        shuffled_idx = torch.randperm(len(label_data_index)).long()
        train_set = Subset(self, label_data_index[shuffled_idx[num_test_sample:]])

        remaining_index = np.concatenate([label_data_index[shuffled_idx[:num_test_sample]],
                                          self.get_data_index_by_label(label=0 if label == 1 else 1)])

        print(f'Size of data with label ={label} :', 100 * len(label_data_index) / self.n)

        test_set = Subset(self, remaining_index)

        return train_set, test_set

    def get_shape(self):
        return self.X.shape
