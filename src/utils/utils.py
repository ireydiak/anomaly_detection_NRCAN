import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
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


def ids_misclf_per_label(y_pred: np.array, y_test_true: np.array, test_labels: np.array):
    # Misclassified rows
    mask = y_pred != y_test_true
    # Counts of unique rows in the test labels
    labels, counts = np.unique(test_labels, return_counts=True)
    # Counts of unique rows in the misclassified labels
    misclf_labels, misclf_counts = np.unique(test_labels[mask], return_counts=True)

    # Assemble the counts and their labels in a dictionary
    gt = {lbl: [cnt, 0, 0, 0] for lbl, cnt in zip(labels, counts)}
    misclf = {lbl: cnt for lbl, cnt in zip(misclf_labels, misclf_counts)}
    # Merge the two dictionaries
    for k, v in misclf.items():
        gt[k][1] = v
        gt[k][2] = v / gt[k][0]
        gt[k][3] = 1 - gt[k][2]
    # Create a dataframe from the merged dictionary
    return pd.DataFrame.from_dict(
        gt,
        orient="index",
        columns=["# Instances test set", "Misclassified count", "Misclassified ratio", "Accuracy"]
    )
