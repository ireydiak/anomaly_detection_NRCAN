import os
from typing import Union
import numpy as np
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import trange
from pyad.loss.TripletCenterLoss import TripletCenterLoss
from pyad.model.transformers import NeuTraLAD
from pyad.trainer.base import BaseTrainer


class NeuTraLADTrainer(BaseTrainer):

    def __init__(self, **kwargs):
        super(NeuTraLADTrainer, self).__init__(**kwargs)
        self.metric_hist = []

        mask_params = list()
        for mask in self.model.masks:
            mask_params += list(mask.parameters())
        self.optimizer = optim.Adam(
            list(self.model.enc.parameters()) + mask_params,
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        self.scheduler = StepLR(self.optimizer, step_size=20, gamma=0.9)
        self.criterion = nn.MSELoss()

    @staticmethod
    def load_from_file(fname: str, device: str = None):
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(fname, map_location=device)
        metric_values = ckpt["metric_values"]
        model = NeuTraLAD.load_from_ckpt(ckpt)
        trainer = NeuTraLADTrainer(model=model, batch_size=1, device=device)
        trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        trainer.metric_values = metric_values

        return trainer, model

    def score(self, sample: torch.Tensor):
        return self.model(sample)

    def train_iter(self, X):
        scores = self.model(X)
        return scores.mean()


class GOADTrainer(BaseTrainer):
    def __init__(self, model, batch_size, **kwargs):
        super(GOADTrainer, self).__init__(model, batch_size, **kwargs)
        # Regularizing terms
        self.margin = model.margin
        self.lamb = model.lamb
        self.eps = model.eps
        # Number of affine transformations
        self.n_transforms = model.n_transforms
        # Cross Entropy loss
        self.ce_loss = nn.CrossEntropyLoss()
        # Triplet loss
        self.tc_loss = TripletCenterLoss(margin=self.margin)
        # Transformation matrix
        self.trans_matrix = torch.randn(
            (self.n_transforms, model.in_features, model.feature_space)
        ).to(self.device)
        # Hypersphere centers
        self.centers = torch.zeros((self.model.feature_space, self.n_transforms)).to(self.device)

    def train(self, dataset: DataLoader):
        self.model.train(mode=True)
        print("Started training")
        for epoch in range(self.n_epochs):
            labels = torch.arange(
                self.n_transforms
            ).unsqueeze(0).expand((self.batch_size, self.n_transforms)).long().to(self.device)
            epoch_loss = 0.0
            n_batch = 0
            self.epoch = epoch
            assert self.model.training, "model not in training mode, aborting"
            self.centers = torch.zeros((self.model.num_hidden_nodes, self.n_transforms)).to(self.device)
            with trange(len(dataset)) as t:
                for sample in dataset:
                    X, _, _ = sample
                    X = X.to(self.device).float()
                    if len(X) != self.batch_size:
                        labels = torch.arange(
                            self.n_transforms
                        ).unsqueeze(0).expand((len(X), self.n_transforms)).long().to(self.device)

                    # Apply affine transformations
                    X_augmented = torch.vstack(
                        [X @ t for t in self.trans_matrix]
                    ).reshape(X.shape[0], self.model.feature_space, self.n_transforms)
                    # Forward pass
                    tc_zs, logits = self.model(X_augmented)
                    # Update enters estimates
                    self.centers += tc_zs.mean(0)

                    # Reset gradient
                    self.optimizer.zero_grad()

                    # Compute loss
                    ce_loss = self.ce_loss(logits, labels)
                    tc_loss = self.tc_loss(tc_zs)
                    loss = self.lamb * tc_loss + ce_loss

                    # Backpropagation
                    loss.backward()
                    self.optimizer.step()

                    n_batch += 1
                    epoch_loss += loss.item()
                    t.set_postfix(
                        loss='{:.3f}'.format(epoch_loss / (epoch + 1)),
                        epoch=epoch + 1
                    )
                    t.update()
            self.centers = (self.centers.T / n_batch).unsqueeze(0)
            if self.ckpt_root and epoch % 5 == 0:
                self.save_ckpt(
                    os.path.join(self.ckpt_root, "{}_epoch={}.pt".format(self.model.name.lower(), epoch + 1))
                )

            if self.validation_ldr is not None and (epoch % 5 == 0 or epoch == 0):
                self.validate()

    def test(self, dataset: DataLoader) -> Union[np.array, np.array]:
        self.model.eval()
        y_true, scores, labels = [], [], []
        with torch.no_grad():
            # val_probs_rots = np.zeros((len(dataset), self.n_transforms))
            for row in dataset:
                X, y, label = row
                X = X.to(self.device).float()
                # Apply affine transformations
                X_augmented = torch.vstack(
                    [X @ t for t in self.trans_matrix]
                ).reshape(X.shape[0], self.model.feature_space, self.n_transforms)
                # Forward pass & reshape
                zs, fs = self.model(X_augmented)
                zs = zs.permute(0, 2, 1)
                # Compute anomaly score
                score = self.score(zs)
                # val_probs_rots[idx] = -torch.diagonal(score, 0, 1, 2).cpu().data.numpy()

                y_true.extend(y.cpu().tolist())
                labels.extend(list(label))
                scores.extend(score.cpu().tolist())
        self.model.train()

        return np.array(y_true), np.array(scores), np.array(labels)

    def train_iter(self, sample: torch.Tensor):
        pass

    def score(self, sample: torch.Tensor):
        diffs = ((sample.unsqueeze(2) - self.centers) ** 2).sum(-1)
        diffs_eps = self.eps * torch.ones_like(diffs)
        diffs = torch.max(diffs, diffs_eps)
        logp_sz = torch.nn.functional.log_softmax(-diffs, dim=2)
        score = -torch.diagonal(logp_sz, 0, 1, 2).sum(dim=1)
        return score
