import torch
import torch.nn as nn
import torch.nn.functional as F


class DSEBMTrainer:
    def __init__(self, model: nn.Module, device: str):
        self.model = model

    def train(self, train_ldr):
        for X, _ in train_ldr:
            noise = self.model.random_noise_like(X)
            X_noise = X + noise
            b_prime = tf.get_variable('b_prime', shape=[opts['batch_size'], 120])

            # self.net_out = network(self.x_input, self.is_training)
            # self.net_nosie_out = network(self.x_noise, self.is_training)
            out = self.model(X)
            out_noise = self.model(X_noise)

            energy = self.energy(X, out)
            energy_noise = self.energy(X_noise, out_noise)
            # fx = self.x_input - tf.gradients(self.energy, self.x_input)
            # self.fx = tf.squeeze(fx, axis=0)
            # self.fx_noise = self.x_noise - tf.gradients(self.energy_noise, self.x_noise)
            fx = (X - torch.autograd.grad(energy, X)).squeeze(0)
            fx_noise = (X_noise - torch.autograd.grad(energy_noise, X_noise))

            # energy score
            # flat = tf.layers.flatten(self.x_input - b_prime)
            # self.list_score_energy = 0.5 * tf.reduce_sum(tf.square(flat), axis=1) - tf.reduce_sum(self.net_out, axis=1)
            flat = torch.flatten(X - b_prime)
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
        return torch.sum(torch.square(X - self.b_prime)) - torch.sum(X_hat)
        # energy = 0.5 * tf.reduce_sum(tf.square(self.x_input - b_prime)) - tf.reduce_sum(self.net_out)
        # energy_noise = 0.5 * torch.sum(torch.square(self.x_noise - self.b_prime)) - torch.sum(X_hat_noise)
        # self.energy_noise = 0.5 * tf.reduce_sum(tf.square(self.x_noise - b_prime)) - tf.reduce_sum(self.net_nosie_out)

