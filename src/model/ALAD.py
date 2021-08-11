import torch
import torch.nn as nn

from src.model.TwoLayerMLP import TwoLayerMLP

"""

KDD ALAD architecture.

Generator (decoder), encoder and discriminator.

"""
import tensorflow as tf
from utils import sn

learning_rate = 1e-5
batch_size = 50
latent_dim = 32
init_kernel = tf.contrib.layers.xavier_initializer()


def leakyReLu(x, alpha=0.2, name=None):
    if name:
        with tf.variable_scope(name):
            return tf.nn.relu(x) - (alpha * tf.nn.relu(-x))
    else:
        return tf.nn.relu(x) - (alpha * tf.nn.relu(-x))


class ALAD(nn.Module):

    def __init__(self, D: int, L: int):
        super(ALAD, self).__init__()
        self.D = D
        self.L = L
        self._build_network()

    def _build_network(self):
        self.encoder = nn.Sequential(
            nn.Linear(self.D, 64),
            nn.LeakyReLU(),
            nn.Linear(64, self.L)
        )
        # TODO: Figure out why 3 layers instead of 2
        self.generator = nn.Sequential(
            nn.Linear(self.L, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, self.D)
        )
        self.D_x = nn.Sequential(
            nn.Linear(self.D, 128),
            nn.BatchNorm1d(num_features=128),
            nn.LeakyReLU()
        )
        self.D_z = nn.Sequential(
            nn.Linear(self.L, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.5)
        )
        self.D_xz = nn.Sequential(
            nn.Linear(self.D + self.L, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
        )
        self.discriminator_xx = TwoLayerMLP(self.D * 2, 128, p=0.2)
        self.discriminator_zz = TwoLayerMLP(self.L * 2, self.L, 0.2, 0.2)

    def discriminator_xz(self, X: torch.Tensor, Z: torch.Tensor):
        x = self.D_x(X)
        z = self.D_z(Z)
        inter_layer_inp_xz = torch.cat((x, z))
        l_enc = self.D_xz(inter_layer_inp_xz)
        return l_enc, inter_layer_inp_xz

    def forward(self, X: torch.Tensor, Z: torch.Tensor):
        x_pl = tf.placeholder(tf.float32, shape=data.get_shape_input(), name="input_x")
        z_pl = tf.placeholder(tf.float32, shape=[None, latent_dim], name="input_z")
        is_training_pl = tf.placeholder(tf.bool, [], name='is_training_pl')
        learning_rate = tf.placeholder(tf.float32, shape=(), name="lr_pl")

        # Data
        logger.info('Data loading...')
        trainx, trainy = data.get_train(label)
        if enable_early_stop: validx, validy = data.get_valid(label)
        trainx_copy = trainx.copy()
        testx, testy = data.get_test(label)

        rng = np.random.RandomState(random_seed)
        nr_batches_train = int(trainx.shape[0] / batch_size)
        nr_batches_test = int(testx.shape[0] / batch_size)

        logger.info('Building graph...')

        logger.warn("ALAD is training with the following parameters:")
        display_parameters(batch_size, starting_lr, ema_decay, degree, label,
                           allow_zz, score_method, do_spectral_norm)

        gen = network.decoder
        enc = network.encoder
        dis_xz = network.discriminator_xz
        dis_xx = network.discriminator_xx
        dis_zz = network.discriminator_zz

        # HEREEEE
        z_gen = self.encoder(X)
        x_gen = self.generator(Z)
        rec_x = self.generator(z_gen)
        rec_z = self.encoder(x_gen)

        l_enc, inter_layer_inp_xz = self.discriminator_xz(X, z_gen)
        l_gen, inter_layer_rct_xz = self.discriminator_xz(x_gen, Z)

        x_logit_real, inter_layer_inp_xx = self.discriminator_xx(X, X)
        x_logit_fake, inter_layer_rct_xx = self.discriminator_xx(X, rec_x)

        z_logit_real, _ = self.discriminator_zz(torch.cat((Z, Z)))
        z_logit_fake, _ = self.discriminator_zz(torch.cat((Z, rec_z)))

        with tf.variable_scope('encoder_model'):
            z_gen = enc(x_pl, is_training=is_training_pl,
                        do_spectral_norm=do_spectral_norm)

        with tf.variable_scope('generator_model'):
            x_gen = gen(z_pl, is_training=is_training_pl)
            rec_x = gen(z_gen, is_training=is_training_pl, reuse=True)

        with tf.variable_scope('encoder_model'):
            rec_z = enc(x_gen, is_training=is_training_pl, reuse=True,
                        do_spectral_norm=do_spectral_norm)

        with tf.variable_scope('discriminator_model_xz'):
            l_encoder, inter_layer_inp_xz = dis_xz(x_pl, z_gen,
                                                   is_training=is_training_pl,
                                                   do_spectral_norm=do_spectral_norm)
            l_generator, inter_layer_rct_xz = dis_xz(x_gen, z_pl,
                                                     is_training=is_training_pl,
                                                     reuse=True,
                                                     do_spectral_norm=do_spectral_norm)

        with tf.variable_scope('discriminator_model_xx'):
            x_logit_real, inter_layer_inp_xx = dis_xx(x_pl, x_pl,
                                                      is_training=is_training_pl,
                                                      do_spectral_norm=do_spectral_norm)
            x_logit_fake, inter_layer_rct_xx = dis_xx(x_pl, rec_x, is_training=is_training_pl,
                                                      reuse=True, do_spectral_norm=do_spectral_norm)

        with tf.variable_scope('discriminator_model_zz'):
            z_logit_real, _ = dis_zz(z_pl, z_pl, is_training=is_training_pl,
                                     do_spectral_norm=do_spectral_norm)
            z_logit_fake, _ = dis_zz(z_pl, rec_z, is_training=is_training_pl,
                                     reuse=True, do_spectral_norm=do_spectral_norm)

    # TODO: Figure out weights initialization
    # def weight_init(self, m):
    #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
    #         nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
    #         nn.init.zeros_(m.bias)
    #     self.apply(weight_init)
