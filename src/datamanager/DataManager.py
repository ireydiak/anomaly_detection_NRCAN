import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, Sampler, Subset


class DataManager(object):
    """
    class that yields dataloaders for train, test, and validation data
    """

    def __init__(self, train_dataset: torch.utils.data.Dataset,
                 test_dataset: torch.utils.data.Dataset,
                 batch_size: int = 1,
                 num_classes: int = None,
                 input_shape: tuple = None,
                 validation: float = 0.1,
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
        self.validation = validation
        self.kwargs = kwargs
        self.seed = seed

        # torch.manual_seed(seed)
        n = len(train_dataset)
        # num_sample = int(n * initial_train_dataset_ratio)
        # shuffled_idx = torch.randperm(n).long()

        # Create the loaders
        train_sampler, val_sampler = self.train_validation_split(len(self.train_set), self.validation,
                                                                 self.seed)
        self.train_loader = DataLoader(self.train_set, self.batch_size, sampler=train_sampler, **self.kwargs)
        self.validation_loader = DataLoader(self.train_set, self.batch_size, sampler=val_sampler, **self.kwargs)
        self.test_loader = DataLoader(test_dataset, batch_size, shuffle=True, **kwargs)

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
