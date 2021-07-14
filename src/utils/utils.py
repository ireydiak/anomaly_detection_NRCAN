import os
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


def optimizer_setup(optimizer_class: Type[torch.optim.Optimizer], **hyperparameters) -> Callable[[torch.nn.Module], torch.optim.Optimizer]:
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
