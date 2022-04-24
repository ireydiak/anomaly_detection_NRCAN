import torch
import torch.nn as nn
import numpy as np
from tqdm import trange
from torch import optim
from torch.autograd import Variable
from sklearn.metrics import precision_recall_curve, roc_auc_score, auc, precision_recall_fscore_support

from utils import score_recall_precision_w_thresold, score_recall_precision

torch.autograd.set_detect_anomaly(True)


class ALADTrainer:
    def __init__(self, model: nn.Module, dm, device, L, learning_rate, optimizer_factory=None):
        assert optimizer_factory is None
        self.model = model
        self.device = device
        self.dm = dm
        # self.batch_size = batch_size
        self.L = L
        self.criterion = nn.BCEWithLogitsLoss()
        self.lr = learning_rate
        self.optim_ge = optim.Adam(
            list(self.model.G.parameters()) + list(self.model.E.parameters()),
            lr=self.lr, betas=(0.5, 0.999)
        )
        self.optim_d = optim.Adam(
            list(self.model.D_xz.parameters()) + list(self.model.D_zz.parameters()) + list(
                self.model.D_xx.parameters()),
            lr=self.lr, betas=(0.5, 0.999)
        )

    def evaluate_on_test_set(self, pos_label=1, **kwargs):
        labels, scores_l1, scores_l2 = [], [], []
        test_ldr = self.dm.get_test_set()
        energy_threshold = kwargs.get('threshold', 80)

        self.model.eval()

        with torch.no_grad():

            scores_l1_train = []
            scores_l2_train = []

            # Calculate score using estimated parameters on test set
            for X_i, label in test_ldr:
                X = X_i.float().to(self.device)
                _, feature_real = self.model.D_xx(X, X)
                _, feature_gen = self.model.D_xx(X, self.model.G(self.model.E(X)))
                score_l1 = torch.sum(torch.abs(feature_real - feature_gen), dim=1)
                score_l2 = torch.linalg.norm(feature_real - feature_gen, 2, keepdim=False, dim=1)

                scores_l1.append(score_l1.cpu().numpy())
                scores_l2.append(score_l2.cpu().numpy())
                labels.append(label.numpy())
            scores_l1 = np.concatenate(scores_l1, axis=0)
            scores_l2 = np.concatenate(scores_l2, axis=0)
            labels = np.concatenate(labels, axis=0)

            per_l1 = np.percentile(scores_l1, 80)
            y_pred_l1 = (scores_l1 >= per_l1)
            per_l2 = np.percentile(scores_l2, 80)
            y_pred_l2 = (scores_l2 >= per_l2)

            combined_scores_l1 = np.concatenate([scores_l1_train, scores_l1], axis=0)  # scores_l1 #

            comp_threshold = 100 * sum(labels == 0) / len(labels)

            res_max = score_recall_precision(combined_scores_l1, scores_l1, labels)
            res = score_recall_precision_w_thresold(combined_scores_l1, scores_l1, labels, pos_label=pos_label,
                                                    threshold=comp_threshold)
            res = dict(res, **res_max)

        # switch back to train mode
        self.model.train()


        return res, _, _, _

    def train_iter_dis(self, X):
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
        # Backward pass
        loss_d.backward()
        self.optim_d.step()

        return loss_d.item()

    def train_iter_gen(self, X):
        # Cleaning gradients
        self.optim_ge.zero_grad()
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
        # Backward pass
        loss_ge.backward()
        self.optim_ge.step()

        return loss_ge.item()

    def train(self, n_epochs):
        train_ldr = self.dm.get_train_set()
        # TODO: test with nn.BCE()
        # self.criterion = nn.BCEWithLogitsLoss()
        for epoch in range(n_epochs):
            print(f"\nEpoch: {epoch + 1} of {n_epochs}")
            ge_losses = 0
            d_losses = 0
            with trange(len(train_ldr)) as t:
                for i, (X_i, X_i_c) in enumerate(zip(train_ldr, train_ldr), 0):
                    train_inputs_dis = X_i[0].to(self.device).float()
                    train_inputs_gen = X_i_c[0].to(self.device).float()
                    loss_d = self.train_iter_dis(train_inputs_dis)
                    loss_ge = self.train_iter_gen(train_inputs_gen)
                    d_losses += loss_d
                    d_losses /= (i + 1)
                    ge_losses += loss_ge
                    ge_losses /= (i + 1)
                    t.set_postfix(
                        loss_d='{:05.4f}'.format(loss_d),
                        loss_ge='{:05.4f}'.format(loss_ge),
                    )
                    t.update()

            print(dict(loss_d='{:05.4f}'.format(ge_losses),
                       loss_ge='{:05.4f}'.format(d_losses), ))
        return 0
