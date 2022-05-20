import os

import torch
import torch.nn as nn
from tqdm import trange
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from src.model.adversarial import ALAD
from src.trainer.base import BaseTrainer

torch.autograd.set_detect_anomaly(True)


class ALADTrainer(BaseTrainer):
    def __init__(self, **kwargs):
        self.optim_ge, self.optim_d = None, None
        super(ALADTrainer, self).__init__(**kwargs)
        self.criterion = nn.BCEWithLogitsLoss()

    @staticmethod
    def load_from_file(fname: str, device: str = None):
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(fname, map_location=device)
        metric_values = ckpt["metric_values"]
        model = ALAD.load_from_ckpt(ckpt)
        trainer = ALADTrainer(model=model, batch_size=ckpt["batch_size"], device=device)
        trainer.optim_ge.load_state_dict(ckpt["optim_ge"])
        trainer.optim_d.load_state_dict(ckpt["optim_d"])
        trainer.metric_values = metric_values

        return trainer, model

    def save_ckpt(self, fname: str):
        general_params = {
            "epoch": self.epoch,
            "batch_size": self.batch_size,
            "model_state_dict": self.model.state_dict(),
            "optim_ge": self.optim_ge.state_dict(),
            "optim_d": self.optim_d.state_dict(),
            "metric_values": self.metric_values
        }
        model_params = self.model.get_params()
        torch.save(dict(**general_params, **model_params), fname)

    def train_iter(self, sample: torch.Tensor):
        pass

    def score(self, sample: torch.Tensor):
        _, feature_real = self.model.D_xx(sample, sample)
        _, feature_gen = self.model.D_xx(sample, self.model.G(self.model.E(sample)))
        return torch.linalg.norm(feature_real - feature_gen, 2, keepdim=False, dim=1)

    def set_optimizer(self, weight_decay):
        self.optim_ge = optim.Adam(
            list(self.model.G.parameters()) + list(self.model.E.parameters()),
            lr=self.lr, betas=(0.5, 0.999)
        )
        self.optim_d = optim.Adam(
            list(self.model.D_xz.parameters()) + list(self.model.D_zz.parameters()) + list(
                self.model.D_xx.parameters()),
            lr=self.lr, betas=(0.5, 0.999)
        )

    def train_iter_dis(self, X):
        # Labels
        y_true = Variable(torch.zeros(X.size(0), 1)).to(self.device)
        y_fake = Variable(torch.ones(X.size(0), 1)).to(self.device)
        # Forward pass
        out_truexz, out_fakexz, out_truezz, out_fakezz, out_truexx, out_fakexx = self.model(X)
        # Compute loss
        # Discriminators Losses
        loss_dxz = self.criterion(out_truexz, y_true) + self.criterion(out_fakexz, y_fake)
        loss_dzz = self.criterion(out_truezz, y_true) + self.criterion(out_fakezz, y_fake)
        loss_dxx = self.criterion(out_truexx, y_true) + self.criterion(out_fakexx, y_fake)
        loss_d = loss_dxz + loss_dzz + loss_dxx

        return loss_d

    def train_iter_gen(self, X):
        # Labels
        y_true = Variable(torch.zeros(X.size(0), 1)).to(self.device)
        y_fake = Variable(torch.ones(X.size(0), 1)).to(self.device)
        # Forward pass
        out_truexz, out_fakexz, out_truezz, out_fakezz, out_truexx, out_fakexx = self.model(X)
        # Generator losses
        loss_gexz = self.criterion(out_fakexz, y_true) + self.criterion(out_truexz, y_fake)
        loss_gezz = self.criterion(out_fakezz, y_true) + self.criterion(out_truezz, y_fake)
        loss_gexx = self.criterion(out_fakexx, y_true) + self.criterion(out_truexx, y_fake)
        cycle_consistency = loss_gexx + loss_gezz
        loss_ge = loss_gexz + cycle_consistency

        return loss_ge

    def train(self, dataset: DataLoader):
        self.model.train()

        for epoch in range(self.n_epochs):
            ge_losses, d_losses = 0, 0
            self.epoch = epoch
            assert self.model.training, "model not in training mode, aborting"
            with trange(len(dataset)) as t:
                for sample in dataset:
                    X, _, _ = sample

                    if len(X) < self.batch_size:
                        break
                    X_dis, X_gen = X.to(self.device).float(), X.clone().to(self.device).float()
                    # Cleaning gradients
                    self.optim_ge.zero_grad()
                    self.optim_d.zero_grad()
                    # Forward pass
                    loss_d = self.train_iter_dis(X_dis)
                    loss_ge = self.train_iter_gen(X_gen)
                    # Backward pass
                    loss_d.backward()
                    loss_ge.backward()
                    self.optim_d.step()
                    self.optim_ge.step()
                    # Journaling
                    d_losses += loss_d.item()
                    ge_losses += loss_ge.item()
                    t.set_postfix(
                        loss_d='{:05.4f}'.format(loss_d),
                        loss_ge='{:05.4f}'.format(loss_ge),
                    )
                    t.update()

            if self.ckpt_root and epoch % 5 == 0:
                self.save_ckpt(
                    os.path.join(self.ckpt_root, "{}_epoch={}.pt".format(self.name, epoch + 1))
                )

            if self.validation_ldr is not None and (epoch % 5 == 0 or epoch == 0):
                self.validate()
