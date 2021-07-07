import torch


def mean_square_loss(X: torch.Tensor, Y: torch.Tensor):
    return torch.mean(
        torch.square(X - Y)
    )
