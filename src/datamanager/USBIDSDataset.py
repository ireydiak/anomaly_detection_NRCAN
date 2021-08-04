import numpy as np
import torch

from torch.utils.data import Dataset, Subset
from torch.utils.data.dataset import T_co


class USBIDSDataset(Dataset):

    name = 'USBIDS'
    
    def __init__(self, path: str, pct: float=1.0):
        if path.endswith(".npz"):
            X = np.load(path)["usbids"]
            # Majority class must be labelled as 0
            # If the ratio of ones is greather than 0.5, we need to invert the labels
            if np.sum(X[:, -1]) / len(X) > .50 :
                X[:, -1] = (~X[:, -1].astype(np.bool))
        else:
            raise RuntimeError(f"Could not open {path}. USBIDSDataset can only read .npz files.")

        # Keep `pct` percent of the original data
        # Extract labels and features in two separate arrays
        if pct < 1.0:
            np.random.shuffle(X)
            self.X = X[0: int(len(X) * pct), :-1]
            self.y = X[0: int(len(X) * pct), -1]
        else:
            self.X = X[:, :-1]
            self.y = X[:, -1].astype(np.uint8)
        
        self.N = len(self.X)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index) -> T_co:
        return self.X[index], self.y[index]
    
    def get_shape(self):
        return self.X.shape
 
    def get_data_index_by_label(self, label):
        return np.argwhere(self.y == label).reshape(-1)

    def split_train_test(self, test_perc=.2, seed=None):
        if seed:
            torch.manual_seed(seed)
        num_test_sample = int(self.N * test_perc)
        shuffled_idx = torch.randperm(self.N).long()
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

        remaining_index = np.concatenate(
            [label_data_index[shuffled_idx[:num_test_sample]],
            self.get_data_index_by_label(label=0 if label == 1 else 1)]
        )

        print(f'Size of data with label ={label} :', 100 * len(label_data_index) / self.N)

        test_set = Subset(self, remaining_index)

        return train_set, test_set