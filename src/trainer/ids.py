import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC

from torch.nn import Parameter
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
from src.loss.EntropyLoss import EntropyLoss
from src.model.density import DSEBM
from src.model.one_class import DeepSVDD
from src.model.reconstruction import AutoEncoder, MemAutoEncoder
from src.model.transformers import NeuTraLAD
from src.trainer.base import BaseTrainer
from src.utils import metrics


class IDSTrainer(BaseTrainer, ABC):
    def __init__(self,
                 ckpt_root: str = None,
                 run_test_validation=False,
                 keep_ckpt=False,
                 thresh_mode="auto",
                 validation_ldr=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.metric_values = {"precision": [], "recall": [], "aupr": [], "f1-score": []}
        if ckpt_root:
            ckpt_root = ckpt_root[:-1] if ckpt_root.endswith("/") else ckpt_root
            self.ckpt_root = ckpt_root
        else:
            self.ckpt_root = "checkpoint"
        self.run_test_validation = run_test_validation
        self.keep_ckpt = keep_ckpt
        assert thresh_mode in ("auto", "optim"), "unknown option {} for `thresh_mode`".format(thresh_mode)
        self.thresh_mode = thresh_mode
        self.validation_ldr = validation_ldr

    def train(self, dataset: DataLoader):
        self.before_training(dataset)
        self.model.train(mode=True)

        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            self.epoch = epoch
            assert self.model.training, "model not in training mode, aborting"
            for sample in dataset:
                X, _, _ = sample
                X = X.to(self.device).float()

                # Reset gradient
                self.optimizer.zero_grad()

                loss = self.train_iter(X)

                # Backpropagation
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            if epoch % 10 == 0:
                print("Epoch={}\tLoss={:2.4f}".format(epoch, epoch_loss))

            if self.ckpt_root and epoch % 5 == 0:
                self.save_ckpt(
                    os.path.join(self.ckpt_root, "{}_epoch={}.pt".format(self.model.name.lower(), epoch + 1)))

            if self.run_test_validation and (epoch % 5 == 0 or epoch == 0):
                y_true, scores, _ = self.test(self.validation_ldr)
                if self.thresh_mode == "optim":
                    test_res = metrics.estimate_optimal_threshold(scores, y_true)
                else:
                    test_res, _ = metrics.score_recall_precision_w_threshold(scores, y_true)
                self.metric_values["precision"].append(test_res["Precision"])
                self.metric_values["recall"].append(test_res["Recall"])
                self.metric_values["aupr"].append(test_res["AUPR"])
                self.metric_values["f1-score"].append(test_res["F1-Score"])

        self.after_training()

    def test(self, dataset: DataLoader):
        self.model.eval()
        y_true, scores, labels = [], [], []
        with torch.no_grad():
            for row in dataset:
                X, y, label = row
                X = X.to(self.device).float()
                score = self.score(X)
                y_true.extend(y.cpu().tolist())
                scores.extend(score.cpu().tolist())
                labels.extend(list(label))
        self.model.train()
        return np.array(y_true), np.array(scores), np.array(labels)

    def inspect_gradient_wrt_input(self, test_ldr, all_labels):
        self.model.eval()
        y_true, scores, labels = [], [], []
        y_grad_wrt_X, label_grad_wrt_X, = {0: [], 1: []}, {label: [] for label in all_labels}
        losses = []
        for row in test_ldr:
            X, y, label = row
            # TODO: put in dataloader
            label = np.array(label)
            X = X.to(self.device).float()
            X.requires_grad = True
            self.optimizer.zero_grad()

            loss = self.train_iter(X)
            loss.backward()

            for y_c in [0, 1]:
                dsdx = X.grad[y == y_c].mean(dim=0).cpu().numpy()
                if len(X.grad[y == y_c]) > 0:
                    y_grad_wrt_X[y_c].append(dsdx)
            for y_c in all_labels:
                dsdx = X.grad[label == y_c].cpu().numpy()
                if len(X.grad[label == y_c]) > 0:
                    label_grad_wrt_X[y_c].append(dsdx)
            losses.append(loss.item())
            y_true.extend(y.cpu().tolist())
            labels.extend(list(label))
        self.model.train()
        y_grad_wrt_X[0], y_grad_wrt_X[1] = np.asarray(y_grad_wrt_X[0]), np.asarray(y_grad_wrt_X[1])
        for y_c in all_labels:
            label_grad_wrt_X[y_c] = np.concatenate(label_grad_wrt_X[y_c])

        return {
            "y_true": np.array(y_true),
            "labels": np.array(labels),
            "y_grad_wrt_X": y_grad_wrt_X,
            "label_grad_wrt_X": label_grad_wrt_X,
            "losses": np.asarray(losses),
        }


class DSEBMIDSTrainer(IDSTrainer):
    def __init__(self, score_metric="reconstruction", **kwargs):
        assert score_metric == "reconstruction" or score_metric == "energy"
        super(DSEBMIDSTrainer, self).__init__(**kwargs)
        self.score_metric = score_metric
        self.criterion = nn.BCEWithLogitsLoss()
        self.b_prime = Parameter(torch.Tensor(self.batch_size, self.model.in_features).to(self.device))
        torch.nn.init.xavier_normal_(self.b_prime)
        self.optim = optim.Adam(
            list(self.model.parameters()) + [self.b_prime],
            lr=self.lr, betas=(0.5, 0.999)
        )

    @staticmethod
    def load_from_file(fname: str, device: str = None):
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(fname, map_location=device)
        metric_values = ckpt["metric_values"]
        model = DSEBM.load_from_ckpt(ckpt)
        trainer = DSEBMIDSTrainer(model=model, batch_size=ckpt["batch_size"], device=device)
        trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        trainer.metric_values = metric_values

        return trainer, model

    def train_iter(self, X):
        noise = self.model.random_noise_like(X).to(self.device)
        X_noise = X + noise
        X.requires_grad_()
        X_noise.requires_grad_()
        out_noise = self.model(X_noise)
        energy_noise = self.energy(X_noise, out_noise)
        dEn_dX = torch.autograd.grad(energy_noise, X_noise, retain_graph=True, create_graph=True)
        fx_noise = (X_noise - dEn_dX[0])
        return self.loss(X, fx_noise)

    def score(self, sample: torch.Tensor):
        # Evaluation of the score based on the energy
        with torch.no_grad():
            flat = sample - self.b_prime
            out = self.model(sample)
            energies = 0.5 * torch.sum(torch.square(flat), dim=1) - torch.sum(out, dim=1)

        # Evaluation of the score based on the reconstruction error
        sample.requires_grad_()
        out = self.model(sample)
        energy = self.energy(sample, out)
        dEn_dX = torch.autograd.grad(energy, sample)[0]
        rec_errs = torch.linalg.norm(dEn_dX, 2, keepdim=False, dim=1)
        return energies.cpu().numpy(), rec_errs.cpu().numpy()

    def loss(self, X, fx_noise):
        out = torch.square(X - fx_noise)
        out = torch.sum(out, dim=-1)
        out = torch.mean(out)
        return out

    def energy(self, X, X_hat):
        return 0.5 * torch.sum(torch.square(X - self.b_prime)) - torch.sum(X_hat)


class NeuTraLADIDSTrainer(IDSTrainer):
    def __init__(self, **kwargs):
        super(NeuTraLADIDSTrainer, self).__init__(**kwargs)
        self.metric_hist = []

        mask_params = list()
        for mask in self.model.masks:
            mask_params += list(mask.parameters())
        self.optimizer = optim.Adam(
            list(self.model.enc.parameters()) + mask_params, lr=self.lr,
        )

        self.scheduler = StepLR(self.optimizer, step_size=20, gamma=0.9)
        self.criterion = nn.MSELoss()

    @staticmethod
    def load_from_file(fname: str, device: str = None):
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(fname, map_location=device)
        metric_values = ckpt["metric_values"]
        model = NeuTraLAD.load_from_ckpt(ckpt)
        trainer = NeuTraLADIDSTrainer(model=model, batch_size=ckpt["batch_size"], device=device)
        trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        trainer.metric_values = metric_values

        return trainer, model

    def score(self, sample: torch.Tensor):
        return self.model(sample)

    def train_iter(self, X):
        scores = self.model(X)
        return scores.mean()


class AEIDSTrainer(IDSTrainer):
    def __init__(self, reg: float = 0.5, **kwargs):
        super(AEIDSTrainer, self).__init__(**kwargs)
        self.reg = reg

    @staticmethod
    def load_from_file(fname: str, device: str = None):
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(fname, map_location=device)
        metric_values = ckpt["metric_values"]
        model = AutoEncoder.load_from_ckpt(ckpt)
        trainer = AEIDSTrainer(model=model, batch_size=ckpt["batch_size"], device=device)
        trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        trainer.metric_values = metric_values

        return trainer, model

    def score(self, sample: torch.Tensor):
        _, X_prime = self.model(sample)
        return ((sample - X_prime) ** 2).sum(axis=1)

    def train_iter(self, X):
        code, X_prime = self.model(X)
        l2_z = code.norm(2, dim=1).mean()
        loss = ((X - X_prime) ** 2).sum(axis=-1).mean() + self.reg * l2_z

        return loss


class MemAEIDSTrainer(IDSTrainer):
    def __init__(self, **kwargs) -> None:
        super(MemAEIDSTrainer, self).__init__(**kwargs)
        self.alpha = self.model.alpha
        self.recon_loss_fn = nn.MSELoss().to(self.device)
        self.entropy_loss_fn = EntropyLoss().to(self.device)

    @staticmethod
    def load_from_file(fname: str, device: str = None):
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(fname, map_location=device)
        metric_values = ckpt["metric_values"]
        model = MemAutoEncoder.load_from_ckpt(ckpt)
        trainer = MemAEIDSTrainer(model=model, batch_size=ckpt["batch_size"], device=device)
        trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        trainer.metric_values = metric_values

        return trainer, model

    def train_iter(self, sample: torch.Tensor):
        x_hat, w_hat = self.model(sample)
        R = self.recon_loss_fn(sample, x_hat)
        E = self.entropy_loss_fn(w_hat)
        return R + (self.alpha * E)

    def score(self, sample: torch.Tensor):
        x_hat, _ = self.model(sample)
        return torch.sum((sample - x_hat) ** 2, axis=1)

    def save_ckpt(self, fname: str):
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metric_values": self.metric_values,
            "alpha": self.model.alpha,
            "mem_dim": self.model.mem_dim,
            "shrink_thres": self.model.shrink_thres
        }, fname)


class DeepSVDDIDSTrainer(IDSTrainer):

    def __init__(self, R=None, c=None, **kwargs):
        super(DeepSVDDIDSTrainer, self).__init__(**kwargs)
        self.c = c
        self.R = R

    def train_iter(self, sample: torch.Tensor):
        assert torch.allclose(self.c, torch.zeros_like(self.c)) is False, "center c not initialized"
        outputs = self.model(sample)
        dist = torch.sum((outputs - self.c) ** 2, dim=1)
        return torch.mean(dist)

    def score(self, sample: torch.Tensor):
        assert torch.allclose(self.c, torch.zeros_like(self.c)) is False, "center c not initialized"
        outputs = self.model(sample)
        return torch.sum((outputs - self.c) ** 2, dim=1)

    def before_training(self, dataset: DataLoader):
        # Initialize hypersphere center c (if c not loaded)
        if self.c is None:
            print("Initializing center c...")
            self.c = self.init_center_c(dataset)
            print("Center c initialized.")

    def save_ckpt(self, fname: str):
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "c": self.c,
            "R": self.R,
            "metric_values": self.metric_values
        }, fname)

    @staticmethod
    def load_from_file(fname: str, device: str = None):
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(fname, map_location=device)
        metric_values = ckpt["metric_values"]
        model = DeepSVDD.load_from_ckpt(ckpt)
        trainer = DeepSVDDIDSTrainer(model=model, batch_size=ckpt["batch_size"], device=device)
        trainer.c = ckpt["c"]
        trainer.R = ckpt.get("R", None)
        trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        trainer.metric_values = metric_values

        return trainer, model

    def init_center_c(self, train_loader, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data.
           Code taken from https://github.com/lukasruff/Deep-SVDD-PyTorch/blob/master/src/optim/deepSVDD_trainer.py"""
        n_samples = 0
        c = torch.zeros(self.model.rep_dim, device=self.device)

        self.model.eval()
        with torch.no_grad():
            for sample in train_loader:
                # get the inputs of the batch
                X, _, _ = sample
                X = X.to(self.device).float()
                outputs = self.model(X)
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)

        if c.isnan().sum() > 0:
            raise Exception("NaN value encountered during init_center_c")

        if torch.allclose(c, torch.zeros_like(c)):
            raise Exception("Center c initialized at 0")

        c /= n_samples

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        self.model.train(mode=True)
        return c


class DAGMMIDSTrainer(IDSTrainer):
    def __init__(self, **kwargs) -> None:
        super(DAGMMIDSTrainer, self).__init__(**kwargs)
        self.lamb_1 = self.model.lambda_1
        self.lamb_2 = self.model.lambda_2
        self.phi = None
        self.mu = None
        self.cov_mat = None
        self.covs = None
        self.reg_covar = self.model.reg_covar

    def save_ckpt(self, fname: str):
        torch.save({
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "phi": self.phi,
            "mu": self.mu,
            "cov_mat": self.cov_mat,
            "covs": self.covs,
            "metric_values": self.metric_values,
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, fname)

    @staticmethod
    def load_from_file(fname: str, trainer, model, device: str = None):
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(fname, map_location=device)
        model = model.load_from_ckpt(ckpt, model)
        trainer.model = model
        trainer.phi = ckpt["phi"]
        trainer.cov_mat = ckpt["cov_mat"]
        trainer.covs = ckpt["covs"]
        trainer.mu = ckpt["mu"]
        trainer.metric_values = ckpt["metric_values"]
        trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        return trainer, model

    def train_iter(self, sample: torch.Tensor):
        z_c, x_prime, _, z_r, gamma_hat = self.model(sample)
        phi, mu, cov_mat = self.compute_params(z_r, gamma_hat)
        energy_result, pen_cov_mat = self.estimate_sample_energy(
            z_r, phi, mu, cov_mat
        )
        self.phi = phi.data
        self.mu = mu.data
        self.cov_mat = cov_mat
        return self.loss(sample, x_prime, energy_result, pen_cov_mat)

    def loss(self, x, x_prime, energy, pen_cov_mat):
        rec_err = ((x - x_prime) ** 2).mean()
        return rec_err + self.lamb_1 * energy + self.lamb_2 * pen_cov_mat

    def weighted_log_sum_exp(self, x, weights, dim):
        """
        Inspired by https://discuss.pytorch.org/t/moving-to-numerically-stable-log-sum-exp-leads-to-extremely-large-loss-values/61938

        Parameters
        ----------
        x
        weights
        dim

        Returns
        -------

        """
        m, idx = torch.max(x, dim=dim, keepdim=True)
        return m.squeeze(dim) + torch.log(torch.sum(torch.exp(x - m) * (weights.unsqueeze(2)), dim=dim))

    def relative_euclidean_dist(self, x, x_prime):
        return (x - x_prime).norm(2, dim=1) / x.norm(2, dim=1)

    def compute_params(self, z: torch.Tensor, gamma: torch.Tensor):
        r"""
        Estimates the parameters of the GMM.
        Implements the following formulas (p.5):
            :math:`\hat{\phi_k} = \sum_{i=1}^N \frac{\hat{\gamma_{ik}}}{N}`
            :math:`\hat{\mu}_k = \frac{\sum{i=1}^N \hat{\gamma_{ik} z_i}}{\sum{i=1}^N \hat{\gamma_{ik}}}`
            :math:`\hat{\Sigma_k} = \frac{
                \sum{i=1}^N \hat{\gamma_{ik}} (z_i - \hat{\mu_k}) (z_i - \hat{\mu_k})^T}
                {\sum{i=1}^N \hat{\gamma_{ik}}
            }`

        The second formula was modified to use matrices instead:
            :math:`\hat{\mu}_k = (I * \Gamma)^{-1} (\gamma^T z)`

        Parameters
        ----------
        z: N x D matrix (n_samples, n_features)
        gamma: N x K matrix (n_samples, n_mixtures)


        Returns
        -------

        """
        N = z.shape[0]

        # K
        gamma_sum = torch.sum(gamma, dim=0)
        phi = gamma_sum / N

        # phi = torch.mean(gamma, dim=0)

        # K x D
        # :math: `\mu = (I * gamma_sum)^{-1} * (\gamma^T * z)`
        mu = torch.sum(gamma.unsqueeze(-1) * z.unsqueeze(1), dim=0) / gamma_sum.unsqueeze(-1)
        # mu = torch.linalg.inv(torch.diag(gamma_sum)) @ (gamma.T @ z)

        mu_z = z.unsqueeze(1) - mu.unsqueeze(0)
        cov_mat = mu_z.unsqueeze(-1) @ mu_z.unsqueeze(-2)
        cov_mat = gamma.unsqueeze(-1).unsqueeze(-1) * cov_mat
        cov_mat = torch.sum(cov_mat, dim=0) / gamma_sum.unsqueeze(-1).unsqueeze(-1)

        return phi, mu, cov_mat

    def estimate_sample_energy(self, z, phi=None, mu=None, cov_mat=None, average_energy=True):
        if phi is None:
            phi = self.phi
        if mu is None:
            mu = self.mu
        if cov_mat is None:
            cov_mat = self.cov_mat

        # Avoid non-invertible covariance matrix by adding small values (self.reg_covar)
        d = z.shape[1]
        cov_mat = cov_mat + (torch.eye(d)).to(self.device) * self.reg_covar
        # N x K x D
        mu_z = z.unsqueeze(1) - mu.unsqueeze(0)

        # scaler
        inv_cov_mat = torch.cholesky_inverse(torch.linalg.cholesky(cov_mat))
        # inv_cov_mat = torch.linalg.inv(cov_mat)
        det_cov_mat = torch.linalg.cholesky(2 * np.pi * cov_mat)
        det_cov_mat = torch.diagonal(det_cov_mat, dim1=1, dim2=2)
        det_cov_mat = torch.prod(det_cov_mat, dim=1)

        exp_term = torch.matmul(mu_z.unsqueeze(-2), inv_cov_mat)
        exp_term = torch.matmul(exp_term, mu_z.unsqueeze(-1))
        exp_term = - 0.5 * exp_term.squeeze()

        # Applying log-sum-exp stability trick
        # https://www.xarg.org/2016/06/the-log-sum-exp-trick-in-machine-learning/
        if exp_term.ndim == 1:
            exp_term = exp_term.unsqueeze(0)
        max_val = torch.max(exp_term.clamp(min=0), dim=1, keepdim=True)[0]
        exp_result = torch.exp(exp_term - max_val)

        log_term = phi * exp_result
        log_term /= det_cov_mat
        log_term = log_term.sum(axis=-1)

        # energy computation
        energy_result = - max_val.squeeze() - torch.log(log_term + self.reg_covar)

        if average_energy:
            energy_result = energy_result.mean()

        # penalty term
        cov_diag = torch.diagonal(cov_mat, dim1=1, dim2=2)
        pen_cov_mat = (1 / cov_diag).sum()

        return energy_result, pen_cov_mat

    def score(self, sample: torch.Tensor):
        code, x_prime, cosim, z, gamma = self.model(sample)
        sample_energy, pen_cov_mat = self.estimate_sample_energy(
            z, self.phi, self.mu, self.cov_mat, average_energy=False
        )
        return sample_energy
