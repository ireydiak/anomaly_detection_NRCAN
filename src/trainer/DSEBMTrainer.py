import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.nn import Parameter
from tqdm import trange


class DSEBMTrainer:
    def __init__(self, model: nn.Module, dm, device, batch, dim, learning_rate, optimizer_factory=None):
        assert optimizer_factory is None
        self.model = model
        self.device = device
        self.dm = dm
        self.criterion = nn.BCEWithLogitsLoss()
        self.lr = learning_rate
        self.batch = batch
        self.b_prime = Parameter(torch.Tensor(batch, dim))
        torch.nn.init.xavier_normal_(self.b_prime)
        self.optim = optim.Adam(
            list(self.model.parameters()) + [self.b_prime],
            lr=self.lr, betas=(0.5, 0.999)
        )
        self.b_prime = self.b_prime.to(self.device)
        pass

    # def train(self, n_epochs):
    #     train_ldr = self.dm.get_train_set()
    #     # TODO: test with nn.BCE()
    #     # self.criterion = nn.BCEWithLogitsLoss()
    #     for epoch in range(n_epochs):
    #         print(f"\nEpoch: {epoch + 1} of {n_epochs}")
    #
    #         losses = 0
    #
    #         with trange(len(train_ldr)) as t:
    #             for i, X_i in enumerate(train_ldr, 0):
    #                 train_inputs = X_i[0].to(self.device).float()
    #                 train_inputs_noise =
    #                 loss = 0
    #
    #                 # losses += loss
    #                 losses /= (i+1)
    #                 t.set_postfix(
    #                     loss='{:05.4f}'.format(loss),
    #
    #                 )
    #                 t.update()
    #
    #         print(dict(loss_d='{:05.4f}'.format(losses)))
    #
    #     return 0

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

                    fx_noise = (X_noise - torch.autograd.grad(energy_noise, X_noise, retain_graph=True)[0])
                    loss = self.loss(X, fx_noise)

                    self.optim.zero_grad()

                    # Use autograd to compute the backward pass.
                    loss.backward()

                    # updates the weights using gradient descent
                    self.optim.step()

                    losses += loss
                    energies += energy
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


    def loss(self, X, fx_noise):
        # self.loss = tf.reduce_mean(tf.square(self.x_input - self.fx_noise))
        return torch.mean(torch.sum(torch.square(X - fx_noise)))

    def energy(self, X, X_hat):
        # self.energy = 0.5 * tf.reduce_sum(tf.square(self.x_input - b_prime)) - tf.reduce_sum(self.net_out)
        # self.energy_noise = 0.5 * tf.reduce_sum(tf.square(self.x_noise - b_prime)) - tf.reduce_sum(self.net_nosie_out)
        return 0.5 * torch.sum(torch.square(X - self.b_prime.to(self.device))) - torch.sum(X_hat)
        # energy = 0.5 * tf.reduce_sum(tf.square(self.x_input - b_prime)) - tf.reduce_sum(self.net_out)
        # energy_noise = 0.5 * torch.sum(torch.square(self.x_noise - self.b_prime)) - torch.sum(X_hat_noise)
        # self.energy_noise = 0.5 * tf.reduce_sum(tf.square(self.x_noise - b_prime)) - tf.reduce_sum(self.net_nosie_out)
