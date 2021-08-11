import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable


class DSEBM(nn.Module):
    def __init__(self, D: int, N: int):
        super(DSEBM, self).__init__()
        self.D = D
        self.noise = None
        self._build_network()
        # self.x_input = tf.placeholder(tf.float32, shape=[None, 120], name='input')
        # self.is_training = tf.placeholder(tf.bool, name='is_training')

        # noise = tf.random_normal(shape=tf.shape(self.x_input), mean=0.0, stddev=1., dtype=tf.float32)
        # self.x_noise = self.x_input + noise
        self.b_prime = Variable(torch.ones(N, D), requires_grad=True)

        b_prime = tf.get_variable('b_prime', shape=[opts['batch_size'], 120])

        self.net_out = network(self.x_input, self.is_training)
        self.net_noise_out = network(self.x_noise, self.is_training)

        self.energy = 0.5 * tf.reduce_sum(tf.square(self.x_input - b_prime)) - tf.reduce_sum(self.net_out)
        self.energy_noise = 0.5 * tf.reduce_sum(tf.square(self.x_noise - b_prime)) - tf.reduce_sum(self.net_nosie_out)

        fx = self.x_input - tf.gradients(self.energy, self.x_input)
        self.fx = tf.squeeze(fx, axis=0)
        self.fx_noise = self.x_noise - tf.gradients(self.energy_noise, self.x_noise)

        self.loss = tf.reduce_mean(tf.square(self.x_input - self.fx_noise))

        ## energy score
        flat = tf.layers.flatten(self.x_input - b_prime)
        self.list_score_energy = 0.5 * tf.reduce_sum(tf.square(flat), axis=1) - tf.reduce_sum(self.net_out, axis=1)

        ## recon score
        delta = self.x_input - self.fx
        delta_flat = tf.layers.flatten(delta)
        self.list_score_recon = tf.norm(delta_flat, ord=2, axis=1, keep_dims=False)

        self.add_optimizers()
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def _build_network(self):
        self.fc_1 = nn.Sequential(
            nn.Linear(self.D, 128),
            nn.Softplus()
        )
        self.fc_2 = nn.Sequential(
            nn.Linear(128, 512),
            nn.Softplus()
        )

    def random_noise_like(self, X: torch.Tensor):
        return torch.normal(mean=0., std=1., size=X.shape).float()

    def forward(self, X: torch.Tensor):
        Variable()
        x = self.fc_2(self.fc_1(X))
        return self.fc_1(self.fc_2(x))

