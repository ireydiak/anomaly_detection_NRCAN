import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from torch import optim
from torch.nn import Parameter
from tqdm import trange

from utils import score_recall_precision_w_thresold, score_recall_precision


class DSEBMTrainer:
    def __init__(self, model: nn.Module, dm, device, batch, dim, learning_rate, optimizer_factory=None):
        assert optimizer_factory is None
        self.model = model
        self.device = device
        self.dm = dm
        self.criterion = nn.BCEWithLogitsLoss()
        self.lr = learning_rate
        self.batch = batch
        self.b_prime = Parameter(torch.Tensor(batch, dim).to(self.device))
        torch.nn.init.xavier_normal_(self.b_prime)
        self.optim = optim.Adam(
            list(self.model.parameters()) + [self.b_prime],
            lr=self.lr, betas=(0.5, 0.999)
        )

    def train(self, n_epochs):
        train_ldr = self.dm.get_train_set()
        for epoch in range(n_epochs):
            print(f"\nEpoch: {epoch + 1} of {n_epochs}")
            losses = 0
            energies = 0
            with trange(len(train_ldr)) as t:
                for i, data in enumerate(train_ldr, 0):

                    X = data[0].float().to(self.device)
                    if len(X) < self.batch:
                        break
                    noise = self.model.random_noise_like(X).to(self.device)

                    X_noise = X + noise
                    X.requires_grad_()
                    X_noise.requires_grad_()
                    # b_prime = tf.get_variable('b_prime', shape=[opts['batch_size'], 120])

                    # self.net_out = network(self.x_input, self.is_training)
                    # self.net_nosie_out = network(self.x_noise, self.is_training)
                    out = self.model(X)
                    out_noise = self.model(X_noise)

                    energy = self.energy(X, out)
                    energy_noise = self.energy(X_noise, out_noise)
                    # fx = self.x_input - tf.gradients(self.energy, self.x_input)
                    # self.fx = tf.squeeze(fx, axis=0)
                    # self.fx_noise = self.x_noise - tf.gradients(self.energy_noise, self.x_noise)
                    dEn_dX = torch.autograd.grad(energy_noise, X_noise, retain_graph=True, create_graph=True)
                    fx_noise = (X_noise - dEn_dX[0])
                    loss = self.loss(X, fx_noise)

                    self.optim.zero_grad()
                    # Use autograd to compute the backward pass.
                    loss.backward()

                    # updates the weights using gradient descent
                    self.optim.step()

                    losses += loss.item()
                    energies += energy.item()
                    losses /= (i + 1)
                    energies /= (i + 1)
                    t.set_postfix(
                        l='{:05.4f}'.format(loss),
                        e='{:05.4f}'.format(energy),

                    )
                    t.update()

    def test(self):
        fx = (X - torch.autograd.grad(energy, X, retain_graph=True)[0]).squeeze(0)

        # energy score
        # flat = tf.layers.flatten(self.x_input - b_prime)
        # self.list_score_energy = 0.5 * tf.reduce_sum(tf.square(flat), axis=1) - tf.reduce_sum(self.net_out, axis=1)
        flat = torch.flatten(X - self.b_prime)
        list_score_energy = 0.5 * torch.sum(torch.square(flat), dim=1) - torch.sum(out, dim=1)

        # recon score
        # delta = self.x_input - self.fx
        delta = X - fx
        # delta_flat = tf.layers.flatten(delta)
        delta_flat = torch.flatten(delta)

        list_score_recon = torch.norm(delta_flat, ord=2, dim=1, keepdim=False)
        # self.list_score_recon = tf.norm(delta_flat, ord=2, axis=1, keep_dims=False)

        # self.add_optimizers()
        # self.sess = tf.Session()
        # init = tf.global_variables_initializer()
        # self.sess.run(init)

    def evaluate_on_test_set(self, pos_label=1, **kwargs):
        labels, scores_r, scores_e = [], [], []
        test_ldr = self.dm.get_test_set()
        energy_threshold = kwargs.get('energy_threshold', 80)

        self.model.eval()

        # with torch.no_grad():

        scores_e_train = []
        scores_r_train = []

        # Create pytorch's train data_loader
        train_loader = self.dm.get_init_train_loader()
        for X_i, _, _ in train_loader:
            # transfer tensors to selected device
            X = X_i.float().to(self.device)
            if len(X) < self.batch:
                break

            # Evaluation of the score based on the energy
            with torch.no_grad():
                flat = X - self.b_prime
                out = self.model(X)
                energies = 0.5 * torch.sum(torch.square(flat), dim=1) - torch.sum(out, dim=1)
            scores_e_train.append(energies.cpu().numpy())

            # Evaluation of the score based on the reconstruction error
            X.requires_grad_()
            out = self.model(X)
            energy = self.energy(X, out)
            dEn_dX = torch.autograd.grad(energy, X)[0]
            # fx = X - dEn_dX
            # delta = X - fx
            rec_errs = torch.linalg.norm(dEn_dX, 2, keepdim=False, dim=1)
            scores_r_train.append(rec_errs.cpu().numpy())

        scores_e_train = np.concatenate(scores_e_train, axis=0)
        scores_r_train = np.concatenate(scores_r_train, axis=0)

        # Calculate score using estimated parameters on test set
        for X_i, label in test_ldr:
            # X = X_i.float().to(self.device)
            # _, feature_real = self.model.D_xx(X, X)
            # _, feature_gen = self.model.D_xx(X, self.model.G(self.model.E(X)))
            # score_l1 = torch.sum(torch.abs(feature_real - feature_gen), dim=1)
            # score_l2 = torch.linalg.norm(feature_real - feature_gen, 2, keepdim=False, dim=1)

            # transfer tensors to selected device
            X = X_i.float().to(self.device)
            if len(X) < self.batch:
                break

            # Evaluation of the score based on the energy
            with torch.no_grad():
                flat = X - self.b_prime
                out = self.model(X)
                energies = 0.5 * torch.sum(torch.square(flat), dim=1) - torch.sum(out, dim=1)
            scores_e.append(energies.cpu().numpy())

            # Evaluation of the score based on the reconstruction error
            X.requires_grad_()
            out = self.model(X)
            energy = self.energy(X, out)
            dEn_dX = torch.autograd.grad(energy, X)[0]
            # fx = X - dEn_dX
            # delta = X - fx
            rec_errs = torch.linalg.norm(dEn_dX, 2, keepdim=False, dim=1)
            scores_r.append(rec_errs.cpu().numpy())
            labels.append(label.numpy())

        scores_r = np.concatenate(scores_r, axis=0)
        scores_e = np.concatenate(scores_e, axis=0)
        labels = np.concatenate(labels, axis=0)

        # per_l1 = np.percentile(scores_r, 80)
        # y_pred_l1 = (scores_r >= per_l1)
        # per_l2 = np.percentile(scores_e, 80)
        # y_pred_l2 = (scores_e >= per_l2)

        combined_scores_e = np.concatenate([scores_e_train, scores_e], axis=0)  # scores_l1 #
        combined_scores_r = np.concatenate([scores_r_train, scores_r], axis=0)

        # print(precision_recall_fscore_support(labels.astype(int), y_pred_l1.astype(int),
        #                                       average='binary'))
        # print(precision_recall_fscore_support(labels.astype(int), y_pred_l2.astype(int),
        #                                       average='binary'))
        #
        # print('ROC AUC score l1: {:.2f}'.format(roc_auc_score(labels, scores_r) * 100))
        # print('ROC AUC score l2: {:.2f}'.format(roc_auc_score(labels, scores_e) * 100))

        res1 = score_recall_precision_w_thresold(combined_scores_e, scores_e, labels, pos_label=pos_label,
                                                 threshold=energy_threshold)
        res1 = self.ren_dict_keys(res1, 'en')
        print('Evaluation with energy')
        score_recall_precision(combined_scores_e, scores_e, labels)

        # Result based on reconstruction
        res2 = score_recall_precision_w_thresold(combined_scores_r, scores_r, labels, pos_label=pos_label,
                                                 threshold=energy_threshold)
        res2 = self.ren_dict_keys(res2, 'rec')
        print('Evaluation with reconstruction')
        score_recall_precision(combined_scores_r, scores_r, labels)

        res = dict(res1, **res2)
        # switch back to train mode
        self.model.train()

        return res, _, _, _

    def ren_dict_keys(self, d: dict, prefix=''):
        d_ = {}
        for k in d.keys():
            d_[f"{prefix}_{k}"] = d[k]

        return d_

    def loss(self, X, fx_noise):
        # self.loss = tf.reduce_mean(tf.square(self.x_input - self.fx_noise))
        out = torch.square(X - fx_noise)
        out = torch.sum(out, dim=-1)
        out = torch.mean(out)
        return out

    def energy(self, X, X_hat):
        # self.energy = 0.5 * tf.reduce_sum(tf.square(self.x_input - b_prime)) - tf.reduce_sum(self.net_out)
        # self.energy_noise = 0.5 * tf.reduce_sum(tf.square(self.x_noise - b_prime)) - tf.reduce_sum(self.net_nosie_out)
        return 0.5 * torch.sum(torch.square(X - self.b_prime)) - torch.sum(X_hat)
        # energy = 0.5 * tf.reduce_sum(tf.square(self.x_input - b_prime)) - tf.reduce_sum(self.net_out)
        # energy_noise = 0.5 * torch.sum(torch.square(self.x_noise - self.b_prime)) - torch.sum(X_hat_noise)
        # self.energy_noise = 0.5 * tf.reduce_sum(tf.square(self.x_noise - b_prime)) - tf.reduce_sum(self.net_nosie_out)
