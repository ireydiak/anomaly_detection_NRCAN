import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import LeakyReLU
from torch.autograd import Variable

from .base import BaseModel
from .utils import weights_init_xavier


# learning_rate = 1e-5
# batch_size = 50
# latent_dim = 32
# init_kernel = tf.contrib.layers.xavier_initializer()

class ALAD(BaseModel):
    def __init__(self, **kwargs):
        self.G = None
        self.E = None
        self.D_zz = None
        self.D_xx = None
        self.D_xz = None
        self.latent_dim = None
        self.name = "ALAD"
        super(ALAD, self).__init__(**kwargs)

    def resolve_params(self, dataset_name: str):
        if dataset_name in ("KDD10", "NSLKDD", "Thyroid"):
            self.latent_dim = 32
        elif dataset_name == "Arrhythmia":
            self.latent_dim = 64
        else:
            self.latent_dim = self.in_features // 2
        self.D_xz = DiscriminatorXZ(self.in_features, 128, self.latent_dim, negative_slope=0.2, p=0.5)
        self.D_xx = DiscriminatorXX(self.in_features, 128, negative_slope=0.2, p=0.2)
        self.D_zz = DiscriminatorZZ(self.latent_dim, self.latent_dim, negative_slope=0.2, p=0.2)
        self.G = Generator(self.latent_dim, self.in_features, negative_slope=1e-4)
        self.E = Encoder(self.in_features, self.latent_dim)
        self.D_xz.apply(weights_init_xavier)
        self.D_xx.apply(weights_init_xavier)
        self.D_zz.apply(weights_init_xavier)
        self.G.apply(weights_init_xavier)
        self.E.apply(weights_init_xavier)

    def forward(self, X):
        # Encoder
        z_gen = self.E(X)

        # Generator
        z_real = Variable(
            torch.randn((X.size(0), self.latent_dim)).to(self.device),
            requires_grad=False
        )
        x_gen = self.G(z_real)

        # DiscriminatorXZ
        out_truexz, _ = self.D_xz(X, z_gen)
        out_fakexz, _ = self.D_xz(x_gen, z_real)

        # DiscriminatorZZ
        out_truezz, _ = self.D_zz(z_real, z_real)
        out_fakezz, _ = self.D_zz(z_real, self.E(self.G(z_real)))

        # DiscriminatorXX
        out_truexx, _ = self.D_xx(X, X)
        out_fakexx, _ = self.D_xx(X, self.G(self.E(X)))

        return out_truexz, out_fakexz, out_truezz, out_fakezz, out_truexx, out_fakexx

    def get_params(self):
        return {
            'latent_dim': self.latent_dim,
            'in_features': self.in_features
        }


class Encoder(nn.Module):
    def __init__(self, in_features, latent_dim, negative_slope=0.2):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.fc_1 = nn.Sequential(
            nn.Linear(in_features, latent_dim * 2),
            nn.LeakyReLU(negative_slope),
            nn.Linear(latent_dim * 2, latent_dim)
        )

    def forward(self, X):
        return self.fc_1(X)


class Generator(nn.Module):
    def __init__(self, latent_dim, feature_dim, negative_slope=1e-4):
        super(Generator, self).__init__()
        self.fc_1 = nn.Sequential(
            nn.Linear(latent_dim, 2 * latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(2 * latent_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, feature_dim)
        )

    def forward(self, Z):
        return self.fc_1(Z)


class DiscriminatorZZ(nn.Module):
    def __init__(self, in_features, out_features, negative_slope=0.2, p=0.5, n_classes=1):
        super(DiscriminatorZZ, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.negative_slope = negative_slope
        self.p = p
        self.n_classes = n_classes
        self._build_network()

    def _build_network(self):
        self.fc_1 = nn.Sequential(
            nn.Linear(2 * self.in_features, self.out_features),
            nn.LeakyReLU(self.negative_slope),
            nn.Dropout(self.p)
        )
        self.fc_2 = nn.Linear(self.out_features, self.n_classes)

    def forward(self, Z, rec_Z):
        ZZ = torch.cat((Z, rec_Z), dim=1)
        mid_layer = self.fc_1(ZZ)
        logits = self.fc_2(mid_layer)
        return logits, mid_layer


class DiscriminatorXX(nn.Module):
    def __init__(self, in_features, out_features, negative_slope=0.2, p=0.5, n_classes=1):
        super(DiscriminatorXX, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_classes = n_classes
        self.negative_slope = negative_slope
        self.p = p
        self._build_network()

    def _build_network(self):
        self.fc_1 = nn.Sequential(
            nn.Linear(self.in_features * 2, self.out_features),
            nn.LeakyReLU(self.negative_slope),
            nn.Dropout(self.p)
        )
        self.fc_2 = nn.Linear(
            self.out_features, self.n_classes
        )

    def forward(self, X, rec_X):
        XX = torch.cat((X, rec_X), dim=1)
        mid_layer = self.fc_1(XX)
        logits = self.fc_2(mid_layer)
        return logits, mid_layer


class DiscriminatorXZ(nn.Module):
    def __init__(self, in_features, out_features, latent_dim, negative_slope=0.2, p=0.5, n_classes=1):
        super(DiscriminatorXZ, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.negative_slope = negative_slope
        self.p = p
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self._build_network()

    def _build_network(self):
        # Inference over x
        self.fc_1x = nn.Sequential(
            nn.Linear(self.in_features, self.out_features),
            nn.BatchNorm1d(self.out_features),
            nn.LeakyReLU(self.negative_slope),
        )
        # Inference over z
        self.fc_1z = nn.Sequential(
            nn.Linear(self.latent_dim, self.out_features),
            nn.LeakyReLU(self.negative_slope),
            nn.Dropout(self.p)
        )
        # Joint inference
        self.fc_1xz = nn.Sequential(
            nn.Linear(2 * self.out_features, self.out_features),
            nn.LeakyReLU(self.negative_slope),
            nn.Dropout(self.p)
        )
        self.fc_2xz = nn.Linear(self.out_features, self.n_classes)

    def forward_xz(self, xz):
        mid_layer = self.fc_1xz(xz)
        logits = self.fc_2xz(mid_layer)
        return logits, mid_layer

    def forward(self, X, Z):
        x = self.fc_1x(X)
        z = self.fc_1z(Z)
        xz = torch.cat((x, z), dim=1)
        logits, mid_layer = self.forward_xz(xz)
        return logits, mid_layer
