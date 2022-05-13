from copy import deepcopy

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, Sampler, Subset


class MySubset(Subset):
    r"""
    Subset of a dataset at specified indices.
    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    def __getitem__(self, idx):
        data = self.dataset[self.indices[idx]]
        return data[0], data[1], idx


class DataManager:
    """
    class that yields dataloaders for train, test, and validation data
    """

    def __init__(self, train_dataset: torch.utils.data.Dataset,
                 test_dataset: torch.utils.data.Dataset,
                 batch_size: int,
                 num_classes: int = None,
                 input_shape: tuple = None,
                 validation: float = 0.,
                 seed: int = 0,
                 **kwargs):
        """
        Args:
            train_dataset: pytorch dataset used for training
            test_dataset: pytorch dataset used for testing
            batch_size: int, size of batches
            num_classes: int number of classes
            input_shape: tuple, shape of the input image
            validation: float, proportion of the train dataset used for the validation set
            seed: int, random seed for splitting train and validation set
            **kwargs: dict with keywords to be used
        """

        self.batch_size = batch_size
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.train_set = train_dataset
        self.test_set = test_dataset
        self.anomaly_ratio = sum(test_dataset.dataset.y == 1)/len(test_dataset)
        self.validation = validation
        self.kwargs = kwargs
        self.seed = seed

        # torch.manual_seed(seed)
        n = len(train_dataset)
        shuffled_idx = torch.randperm(n).long()

        # Create a mask to track selection process
        self.train_selection_mask = torch.ones_like(shuffled_idx)

        self.current_train_set = MySubset(train_dataset, self.train_selection_mask.nonzero().squeeze())

        # Create the loaders
        train_sampler, val_sampler = self.train_validation_split(
            len(self.train_set), self.validation, self.seed
        )

        self.init_train_loader = DataLoader(self.current_train_set, self.batch_size, sampler=train_sampler,
                                            **self.kwargs)
        self.train_loader = DataLoader(self.current_train_set, self.batch_size, sampler=train_sampler, **self.kwargs)
        self.validation_loader = DataLoader(self.train_set, self.batch_size, sampler=val_sampler, **self.kwargs)
        self.test_loader = DataLoader(test_dataset, batch_size, shuffle=True, **kwargs)

    def get_current_training_set(self):
        return self.current_train_set

    def get_init_train_loader(self):
        return self.init_train_loader

    def get_selected_indices(self):
        return self.train_selection_mask.nonzero().squeeze()

    def update_train_set(self, selected_indices):
        """
        This function update the training set with the new filter
        :param selected_indices: indices of new data to select
        :return:
        train_loader and validation_loader
        """
        self.train_selection_mask[:] = 0
        self.train_selection_mask[selected_indices] = 1
        lbl_sample_idx = self.train_selection_mask.nonzero().squeeze()
        self.current_train_set = MySubset(self.train_set, lbl_sample_idx)
        train_sampler, val_sampler = self.train_validation_split(len(self.current_train_set), self.validation,
                                                                 self.seed)
        self.train_loader = DataLoader(self.current_train_set, self.batch_size, sampler=train_sampler, **self.kwargs)
        self.validation_loader = DataLoader(self.current_train_set, self.batch_size, sampler=val_sampler, **self.kwargs)
        return self.train_loader, self.validation_loader

    @staticmethod
    def train_validation_split(num_samples, validation_ratio, seed=0):
        """
        This function returns two samplers for training and validation data.
        :param num_samples: total number of sample to split
        :param validation_ratio: percentage of validation dataset
        :param seed: random seed to use
        :return:
        """
        # torch.manual_seed(seed)
        num_val = int(num_samples * validation_ratio)
        shuffled_idx = torch.randperm(num_samples).long()
        train_idx = shuffled_idx[num_val:]
        val_idx = shuffled_idx[:num_val]
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        return train_sampler, val_sampler

    def get_train_set(self):
        return self.train_loader

    def get_validation_set(self):
        return self.validation_loader

    def get_test_set(self):
        return self.test_loader

    def get_classes(self):
        return range(self.num_classes)

    def get_input_shape(self):
        return self.input_shape

    def get_batch_size(self):
        return self.batch_size

    def get_random_sample_from_test_set(self):
        indice = np.random.randint(0, len(self.test_set))
        return self.test_set[indice]
