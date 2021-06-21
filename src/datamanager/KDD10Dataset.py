import os

import torch
from torch.utils.data import Dataset, Subset
from torch.utils.data.dataset import T_co
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
NPZ_FILENAME = 'kdd10_train.npz'
BASE_PATH = '../data'


class KDD10Dataset(Dataset):
    """
    This class is used to load KDD Cup 10% dataset as a pytorch Dataset
    """
    def __init__(self, path='../data/kdd10_train'):
        self.path = path

        # load data
        if path and os.path.exists(path):
            X = np.load(path)
        elif os.path.exists(f"{BASE_PATH}/{NPZ_FILENAME}"):
            X = np.load(f"{BASE_PATH}/{NPZ_FILENAME}")['kdd']
        else:
            df = self._import_data()
            X = self.preprocess(df)

        # Extract labels and features in two separate arrays
        self.X = X[:, :-1]
        self.y = X[:, -1]
        self.n = len(X)

    def _import_data(self):
        url_base = "https://archive.ics.uci.edu/ml/machine-learning-databases/kddcup99-mld"
        url_data = f"{url_base}/kddcup.data_10_percent.gz"
        url_info = f"{url_base}/kddcup.names"
        df_info = pd.read_csv(url_info, sep=":", skiprows=1, index_col=False, names=["colname", "type"])
        colnames = df_info.colname.values
        coltypes = np.where(df_info["type"].str.contains("continuous"), "float", "str")
        colnames = np.append(colnames, ["label"])
        coltypes = np.append(coltypes, ["str"])
        return pd.read_csv(url_data, names=colnames, index_col=False, dtype=dict(zip(colnames, coltypes)))

    def preprocess(self, df: pd.DataFrame):
        # Column "num_outbound_cmds" was not expressively dropped in the original paper.
        # However it contains only one distinct value (0.0). Hence it is an uninformative variable.
        # Dropping it also makes out feature space match the one of the paper.
        df = df.drop("num_outbound_cmds", axis=1)

        # One-hot encode the seven categorical attributes (except labels)
        # Assumes dtypes were previously assigned
        one_hot = pd.get_dummies(df.iloc[:, :-1])
        assert one_hot.shape[1] == 120

        # min-max scaling
        scaler = MinMaxScaler()
        cols = one_hot.select_dtypes("float").columns
        one_hot[cols] = scaler.fit_transform(one_hot[cols].values.astype(np.float64))

        # Extract and simplify labels (normal data is 1, attacks are labelled as 0)
        y = np.where(df.label == "normal.", 1, 0)

        X = np.concatenate(
            (one_hot.values, y.reshape(-1, 1)),
            axis=1
        )

        # Save data
        np.savez(f"{BASE_PATH}/{NPZ_FILENAME}", kdd=X.astype(np.float64))

        return X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index) -> T_co:
        return self.X[index], self.y[index]

    def get_data_index_by_label(self, label):
        indices = np.array([i for i in range(self.n)])
        return indices[self.y == label]

    def _load_data(self, path):
        column_names = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
                                'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
                                'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
                                'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
                                'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
                                'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
                                'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
                                'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                                'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'type']

        df = pd.read_csv(path, header=None, names=column_names)

        # regroup all the abnormal data to attack
        df['type'] = df['type'].map(lambda x: 0 if x == 'normal.' else 1)

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

        np.savez('../data/kdd_cup.npz', kdd=kdd_numpy)

        return kdd_numpy

    def split_train_test(self, test_perc=.2, seed=None):
        if seed:
            torch.manual_seed(seed)
        num_test_sample = int(self.n * test_perc)
        shuffled_idx = torch.randperm(self.n).long()
        train_set = Subset(self, shuffled_idx[num_test_sample:])
        test_set = Subset(self, shuffled_idx[:num_test_sample])

        return train_set, test_set

    def one_class_split_train_test(self, test_perc=.2, label=0, seed=None):
        if seed:
            torch.manual_seed(seed)
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
