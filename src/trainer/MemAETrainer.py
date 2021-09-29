import torch
import torch.nn as nn
import numpy as np

from tqdm import trange
from loss import EntropyLoss
from sklearn import metrics

from utils import score_recall_precision_w_thresold, score_recall_precision


class MemAETrainer:
    def __init__(self, model: nn.Module, dm, optimizer_factory, device: str, entropy_loss_weight: float = 0.0002):
        self.device = device
        self.model = model.to(device)
        self.dm = dm
        self.optim = optimizer_factory(self.model)
        self.recon_loss_fn = nn.MSELoss().to(device)
        self.entropy_loss_fn = EntropyLoss().to(device)
        self.entropy_loss_weight = entropy_loss_weight

    def train(self, n_epochs: int):
        print(f'Training with {self.__class__.__name__}')
        mean_loss = np.inf
        train_ldr = self.dm.get_train_set()

        for epoch in range(n_epochs):
            print(f"\nEpoch: {epoch + 1} of {n_epochs}")
            with trange(len(train_ldr)) as t:
                for i, X_i in enumerate(train_ldr, 0):
                    train_inputs = X_i[0].to(self.device).float()
                    loss, entropy_loss, recon_loss = self.train_iter(train_inputs)
                    t.set_postfix(
                        loss='{:05.3f}'.format(loss),
                        entropy_loss='{:05.3f}'.format(entropy_loss),
                        recon_loss='{:05.3f}'.format(recon_loss)
                    )
                    t.update()
        return mean_loss

    def train_iter(self, x):
        x_prime, att = self.model(x)

        recon_loss = self.recon_loss_fn(x_prime, x)
        entropy_loss = self.entropy_loss_fn(att)

        loss = recon_loss + self.entropy_loss_weight * entropy_loss

        self.optim.zero_grad()

        # Use autograd to compute the backward pass.
        loss.backward()

        # updates the weights using gradient descent
        self.optim.step()

        return loss.item(), entropy_loss.item(), recon_loss.item()

    def evaluate_on_test_set(self, pos_label=1, **kwargs):
        """
        function that evaluate the model on the test set every iteration of the
        active learning process
        """
        test_loader = self.dm.get_test_set()
        energy_threshold = kwargs.get('energy_threshold', 80)

        # Change the model to evaluation mode
        self.model.eval()

        train_score = []

        with torch.no_grad():
            # Create pytorch's train data_loader
            train_loader = self.dm.get_init_train_loader()
            for i, data in enumerate(train_loader, 0):
                # transfer tensors to selected device
                train_inputs = data[0].float().to(self.device)

                # forward pass
                X_prime, _ = self.model(train_inputs)

                # (X - X_prime)

                # train_score.append(h_x.cpu().numpy())
                train_score.append(((train_inputs - X_prime) ** 2).sum(axis=-1).squeeze().cpu().numpy())
            train_score = np.concatenate(train_score, axis=0)

            # Calculate score using estimated parameters
            test_score = []
            test_labels = []
            test_z = []

            for data in test_loader:
                test_inputs, label_inputs = data[0].float().to(self.device), data[1]

                # forward pass
                # forward pass
                X_prime, _ = self.model(test_inputs)

                test_score.append(((test_inputs - X_prime) ** 2).sum(axis=-1).squeeze().cpu().numpy())
                # test_score.append(h_x.cpu().numpy())
                # test_z.append(code.cpu().numpy())
                test_labels.append(label_inputs.numpy())

            test_score = np.concatenate(test_score, axis=0)
            # test_z = np.concatenate(test_z, axis=0)
            test_labels = np.concatenate(test_labels, axis=0)

            combined_score = np.concatenate([train_score, test_score], axis=0)

            res = score_recall_precision_w_thresold(combined_score, test_score, test_labels, pos_label=pos_label,
                                                    threshold=energy_threshold)

            score_recall_precision(combined_score, test_score, test_labels)

            # switch back to train mode
            self.model.train()

            return res, test_z, test_labels, combined_score

        # with torch.no_grad():
        #     errors = []
        #     rec_errors = []
        #     y_true = []
        #     for data in test_loader:
        #         X, y = data[0].float().to(self.device), data[1]
        #         # forward pass
        #         X_prime, att = self.model(X)
        #         err = torch.sqrt(
        #             torch.sum((X_prime - X) ** 2, dim=1)
        #         )
        #         y_true.extend(y.cpu().tolist())
        #         rec_errors.extend(err.cpu().tolist())
        #         errors.append(torch.mean(err).item())
        #     y_true = np.array(y_true)
        #     thresh = np.mean(np.array(errors))
        #     y_pred = np.where(rec_errors >= thresh, 1, 0)
        #     res['Accuracy'] = metrics.accuracy_score(y_true, y_pred)
        #     res['Precision'], res['Recall'], res['F1-Score'], _ = metrics.precision_recall_fscore_support(y_true, y_pred, average='binary', pos_label=pos_label)
        #     print(','.join([f'{key_name}: {key_val}' for key_name, key_val in res.items()]))
        #
        #     # switch back to train mode
        #     self.model.train()
        #
        #     return res, 0, 0, 0
