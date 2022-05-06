import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from typing import Type, Callable


def predict_proba(scores):
    """
    Predicts probability from the score

    Parameters
    ----------
    scores: the score values from the model

    Returns
    -------

    """
    prob = F.softmax(scores, dim=1)
    return prob


def check_dir(path):
    """
    This function ensure that a path exists or create it
    """
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def check_file_exists(path):
    """
    This function ensure that a path exists
    """
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_X_from_loader(loader):
    """
    This function returns the data set X from the provided pytorch @loader
    """
    X = []
    y = []
    for i, X_i in enumerate(loader, 0):
        X.append(X_i[0])
        y.append(X_i[1])
    X = torch.cat(X, axis=0)
    y = torch.cat(y, axis=0)
    return X.numpy(), y.numpy()


def average_results(results: dict):
    """
        Calculate Means and Stds of metrics in @results
    """

    final_results = defaultdict()
    for k, v in results.items():
        final_results[f'{k}'] = f"{np.mean(v):.4f}({np.std(v):.4f})"
        # final_results[f'{k}_std'] = np.std(v)
    return final_results


def optimizer_setup(optimizer_class: Type[torch.optim.Optimizer], **hyperparameters) -> Callable[
    [torch.nn.Module], torch.optim.Optimizer]:
    """
    Creates a factory method that can instanciate optimizer_class with the given
    hyperparameters.

    Why this? torch.optim.Optimizer takes the model's parameters as an argument.
    Thus we cannot pass an Optimizer to the CNNBase constructor.

    Parameters
    ----------
    optimizer_class: optimizer used to train the model
    hyperparameters: hyperparameters for the model

    Returns
    -------

    """

    def f(model):
        return optimizer_class(model.parameters(), **hyperparameters)

    return f


def random_split_to_two(table, ratio=.2):
    n1 = int(len(table) * (1 - ratio))
    shuffle_idx = torch.randperm(len(table)).long()

    t1 = table[shuffle_idx[:n1]]
    t2 = table[shuffle_idx[n1:]]

    return t1, t2
