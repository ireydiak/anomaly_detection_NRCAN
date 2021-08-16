import torch
import torch.nn as nn
import numpy as np
from tqdm import trange


class ALADTrainer:
    def __init__(self, model: nn.Module, dm, optimizer_factory, device, batch_size, L):
        self.model = model
        self.device = device
        self.optim = optimizer_factory(self.model)
        self.dm = dm
        self.batch_size = batch_size
        self.L = L

    def train(self, n_epochs: int):
        mean_loss = 0
        train_ldr = self.dm.get_train_set()

        for epoch in range(n_epochs):
            print(f"\nEpoch: {epoch + 1} of {n_epochs}")
            with trange(len(train_ldr)) as t:
                for _, X_i in enumerate(train_ldr, 0):
                    train_inputs = X_i[0].to(self.device).float()
                    Z = torch.Tensor(np.random.normal(size=[self.batch_size, self.L])).to(self.device)
                    loss_gen, loss_enc, dis_loss_xz, dis_loss_xx, dis_loss_zz  = self.train_iter(train_inputs, Z)
                    t.set_postfix(
                        loss_gen='{:05.4f}'.format(loss_gen),
                        loss_enc='{:05.4f}'.format(loss_enc),
                        dis_loss_xz='{:05.4f}'.format(dis_loss_xz),
                        dis_loss_xx='{:05.4f}'.format(dis_loss_xx),
                        dis_loss_zz='{:05.4f}'.format(dis_loss_zz)
                    )
                    t.update()
                    mean_loss += (loss_gen + loss_enc + dis_loss_xz + dis_loss_xx + dis_loss_zz)
        return mean_loss / n_epochs
    
    def train_iter(self, X, Z):
        l_encoder, l_generator, x_logit_real, x_logit_fake, z_logit_real, z_logit_fake = self.model(X, Z)
        loss_gen, loss_enc, dis_loss_xz, dis_loss_xx, dis_loss_zz = self.model.compute_loss(l_encoder, l_generator, x_logit_real, x_logit_fake, z_logit_real, z_logit_fake)
        
        self.optim.zero_grad()

        # Use autograd to compute the backward pass.
        for loss in [loss_gen, loss_enc, dis_loss_xz, dis_loss_xx, dis_loss_zz]:
            loss.backward(retain_graph=True)

        # updates the weights using gradient descent
        self.optim.step()

        return loss_gen.item(), loss_enc.item(), dis_loss_xz.item(), dis_loss_xx.item(), dis_loss_zz.item()

    def evaluate_on_test_set(self, **kwargs):
        raise Exception("Unimplemented")
        # """
        # function that evaluate the model on the test set every iteration of the
        # active learning process
        # """
        # test_loader = self.dm.get_test_set()

        # # Change the model to evaluation mode
        # self.model.eval()
        # res = {"Accuracy": -1, "Precision": -1, "Recall": -1, "F1-Score": -1}

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

        #     # switch back to train mode
        #     self.model.train()

        #     return res, 0, 0, 0