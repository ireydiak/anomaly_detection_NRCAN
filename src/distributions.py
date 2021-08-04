import torch
import numpy as np


def multivariate_normal_pdf(X, phi, mu, Sigma, scaling=False):
    d, det, scaler = len(X), 1, 1
    if X.ndim > 1:
        d = X.shape[1]
    if scaling:
        if Sigma.ndim > 1 and Sigma.shape[1] > 1:
            det = torch.sqrt(torch.linalg.det(Sigma))
        two_pi_power = np.power(2. * np.pi, d / 2.)
        scaler = (1. / (det * two_pi_power))
    Sigma_inv = torch.eye(d) * 1e-6
    exp_term = torch.exp(-0.5 * (X - mu).T @ Sigma_inv @ (X - mu))
    res = phi * scaler * exp_term
    return res.item()


def multivariate_normal_cholesky_pdf(X, phi, mu, Sigma):
    d, det = len(X), 1
    if X.ndim > 1:
        d = X.shape[1]
    L = torch.linalg.cholesky(Sigma)
    det = torch.prod(torch.diag(L))
    two_pi_pwr = torch.pow(2. * np.pi, d / 2.)
    term = (torch.linalg.inv(L) @ (X - mu).T) @ torch.linalg.inv(L) @ (X - mu)
    return phi * (1. / (det * two_pi_pwr) * torch.exp(-0.5 * term))


def estimate_GMM_params(X, gamma) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    """
    Estimates GMM parameters :math:`\phi`, :math:`\mu` and :math:`\Sigma`
    Parameters
    ----------
    X: Samples
    gamma: Output of a softmax

    Returns
    -------
    :math:`\phi`: The mixture component
    :math:`\mu`: The mean vector
    :math:`\Sigma`: The covariance matrix
    """
    # K: number of mixtures
    # D: dimensionality
    # N: number of inputs
    K, D, N = gamma.shape[1], X.shape[1], X.shape[0]
    phi = torch.mean(gamma, dim=0)
    mu = torch.zeros([K, D])
    Sigma = torch.zeros([K, D, D])
    # TODO: replace loops by matrix operations
    for k in range(0, K):
        mu_tmp = torch.zeros([D])
        sig_tmp = torch.zeros([D, D])
        for i in range(0, N):
            mu_tmp = mu_tmp + gamma[i, k] * X[i, :]
        mu[k, :] = mu_tmp / N
        for i in range(0, N):
            sig_tmp = sig_tmp + gamma[i, k] * ((X[i, :] - mu[k, :]).T @ X[i, :] - mu[k, :])
        Sigma[k, :, :] = sig_tmp / N

    return phi, mu, Sigma
