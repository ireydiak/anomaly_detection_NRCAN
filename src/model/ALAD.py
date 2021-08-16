import torch
import torch.nn as nn
import torch.nn.functional as F
from .TwoLayerMLP import TwoLayerMLP

# learning_rate = 1e-5
# batch_size = 50
# latent_dim = 32
# init_kernel = tf.contrib.layers.xavier_initializer()

class ALAD(nn.Module):

    def __init__(self, D: int, L: int):
        # TODO: weight init
        super(ALAD, self).__init__()
        self.D = D
        self.L = L
        self._build_network()

    def _build_network(self):
        self.encoder = nn.Sequential(
            nn.Linear(self.D, self.D // 2),
            nn.LeakyReLU(),
            nn.Linear(self.D // 2, self.L)
        )
        # TODO: Figure out why 3 layers instead of 2
        self.generator = nn.Sequential(
            nn.Linear(self.L, self.L * 2),
            nn.ReLU(),
            nn.Linear(self.L * 2, 128),
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
        # self.D_xz = nn.Sequential(
        #     nn.Linear(self.D + self.L, 128),
        #     nn.LeakyReLU(),
        #     nn.Dropout(0.5),
        # )
        self.D_xz = TwoLayerMLP(128 * 2, 128, p=0.5)
        self.D_xx = TwoLayerMLP(self.D * 2, 128, p=0.2)
        self.D_zz = TwoLayerMLP(self.L * 2, self.L, 0.2, 0.2)

    def forward(self, X, Z):
        # Encode X
        z_gen = self.encoder(X)
        # Decode Z
        x_gen = self.generator(Z)
        # Decode X
        rec_x = self.generator(z_gen)
        # Encode Z'
        rec_z = self.encoder(x_gen)
        # Discriminator xz
        l_encoder, _ = self.forward_Dxz(X, z_gen)
        l_generator, _ = self.forward_Dxz(x_gen, Z)
        # Discriminator xx
        x_logit_real, _ = self.D_xx(torch.cat((X, X), dim=1))
        x_logit_fake, _ = self.D_xx(torch.cat((X, rec_x), dim=1))
        # Discriminator zz
        z_logit_real, _ = self.D_zz(torch.cat((Z, Z), dim=1))
        z_logit_fake, _ = self.D_zz(torch.cat((Z, rec_z), dim=1))
        return l_encoder, l_generator, x_logit_real, x_logit_fake, z_logit_real, z_logit_fake

    def compute_loss(self, l_encoder, l_generator, x_logit_real, x_logit_fake, z_logit_real, z_logit_fake):
        # Discriminator xz
        loss_dis_enc = F.binary_cross_entropy_with_logits(l_encoder, torch.ones_like(l_encoder))
        loss_dis_gen = F.binary_cross_entropy_with_logits(l_generator, torch.ones_like(l_generator))
        dis_loss_xz = loss_dis_enc + loss_dis_gen
        # Discriminator xx TODO: reduce mean?
        x_real_dis = F.binary_cross_entropy_with_logits(x_logit_real, torch.ones_like(x_logit_real))
        x_fake_dis = F.binary_cross_entropy_with_logits(x_logit_fake, torch.zeros_like(x_logit_fake))
        dis_loss_xx = x_real_dis + x_fake_dis
        # Discriminator zz TODO: reduce mean?
        z_real_dis = F.binary_cross_entropy_with_logits(z_logit_real, torch.ones_like(z_logit_real))
        z_fake_dis = F.binary_cross_entropy_with_logits(z_logit_fake, torch.zeros_like(z_logit_fake))
        dis_loss_zz = z_real_dis + z_fake_dis
        # Combined loss
        loss_D = dis_loss_xz + dis_loss_xx + dis_loss_zz
        # Generator and Encoder
        gen_loss_xz = F.binary_cross_entropy_with_logits(l_generator, torch.ones_like(l_generator))
        enc_loss_xz = F.binary_cross_entropy_with_logits(l_encoder, torch.zeros_like(l_encoder))
        x_real_gen = F.binary_cross_entropy_with_logits(x_logit_real, torch.zeros_like(x_logit_real))
        x_fake_gen = F.binary_cross_entropy_with_logits(x_logit_fake, torch.ones_like(x_logit_fake))
        z_real_gen = F.binary_cross_entropy_with_logits(z_logit_real, torch.zeros_like(z_logit_real))
        z_fake_gen = F.binary_cross_entropy_with_logits(z_logit_fake, torch.ones_like(z_logit_fake))
        # Combined loss
        cost_x = x_real_gen + x_fake_gen
        cost_z = z_real_gen + z_fake_gen
        cycle_consistency_loss = cost_x + cost_z
        loss_gen = gen_loss_xz + cycle_consistency_loss
        loss_enc = enc_loss_xz + cycle_consistency_loss

        return loss_gen, loss_enc, dis_loss_xz, dis_loss_xx, dis_loss_zz    
    
    def forward_Dxz(self, X, Z):
        x = self.D_x(X)
        z = self.D_z(Z)
        xz = torch.cat((x, z), dim=1)
        logit, mid_layer = self.D_xz(xz)
        return logit, mid_layer

    # TODO: Figure out weights initialization
    # def weight_init(self, m):
    #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
    #         nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
    #         nn.init.zeros_(m.bias)
    #     self.apply(weight_init)
