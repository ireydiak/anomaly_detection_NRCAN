import torch
import torch.nn as nn
import numpy as np
from tqdm import trange
from torch import optim
from torch.autograd import Variable
from sklearn.metrics import precision_recall_curve, roc_auc_score, auc
torch.autograd.set_detect_anomaly(True)

class ALADTrainer:
    def __init__(self, model: nn.Module, dm, device, batch_size, L, learning_rate, optimizer_factory=None):
        assert optimizer_factory is None
        self.model = model
        self.device = device
        self.dm = dm
        self.batch_size = batch_size
        self.L = L
        self.lr = learning_rate
        self.optim_ge = optim.Adam(
            list(self.model.G.parameters()) + list(self.model.E.parameters()),
            lr=self.lr, betas=(0.5, 0.999) 
        )
        self.optim_d = optim.Adam(
            list(self.model.D_xz.parameters()) + list(self.model.D_zz.parameters()) + list(self.model.D_xx.parameters()),
            lr=self.lr, betas=(0.5, 0.999)
        )

    def evaluate_on_test_set(self, **kwargs):
        labels, scores = [], []
        test_ldr = self.dm.get_test_set()
        self.model.eval()

        with torch.no_grad():
            for X_i, label in test_ldr[1]:
                X = X_i.float().to(self.device)
                _, feature_real = self.model.D_xx(X, X)
                _, feature_gen = self.model.D_xx(X, self.model.G(self.model.E(X)))
                score = torch.sum(torch.abs(feature_real - feature_gen), dim=1)
                scores.append(score.cpu())
                labels.append(label.cpu())
        scores = torch.cat(scores, dim=0)
        labels = torch.cat(labels, dim=0)
        precision, recall, thresholds = precision_recall_curve(labels, scores)
        print('ROC AUC score: {:.2f}'.format(roc_auc_score(labels, scores) * 100))

        return labels, scores

    def train_iter(self, X):
        # Cleaning gradients
        self.optim_ge.zero_grad()
        self.optim_d.zero_grad()
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
        # Generator losses
        loss_gexz = self.criterion(out_fakexz, y_true) + self.criterion(out_truexz, y_fake)
        loss_gezz = self.criterion(out_fakezz, y_true) + self.criterion(out_truezz, y_fake)
        loss_gexx = self.criterion(out_fakexx, y_true) + self.criterion(out_truexx, y_fake)
        cycle_consistency = loss_gexx + loss_gezz
        loss_ge = loss_gexz + cycle_consistency
        # Backward pass
        loss_d.backward(retain_graph=True)
        loss_ge.backward()

        self.optim_d.step()
        self.optim_ge.step()

        return loss_d.item(), loss_ge.item()

    def train(self, n_epochs):
        train_ldr = self.dm.get_train_set()
        # TODO: test with nn.BCE()
        self.criterion = nn.BCEWithLogitsLoss()
        for epoch in range(n_epochs):
            print(f"\nEpoch: {epoch + 1} of {n_epochs}")
            ge_losses = 0
            d_losses = 0
            with trange(len(train_ldr)) as t:
                 for _, X_i in enumerate(train_ldr, 0):
                    train_inputs = X_i[0].to(self.device).float()
                    loss_d, loss_ge = self.train_iter(train_inputs)
                    d_losses += loss_d
                    ge_losses += loss_ge
                    t.set_postfix(
                        loss_d='{:05.4f}'.format(loss_d),
                        loss_ge='{:05.4f}'.format(loss_ge),
                    )
                    t.update()
        return 0