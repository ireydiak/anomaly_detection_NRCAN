import torch.nn as nn
from .memory_module import MemoryUnit


class MemAutoEncoder(nn.Module):

    def __init__(self, D, L, mem_dim, shrink_thres=0.0025, device='cpu'):
        """
        Implements model Memory AutoEncoder as described in the paper
        `Memorizing Normality to Detect Anomaly: Memory-augmented Deep Autoencoder (MemAE) for Unsupervised Anomaly Detection`.
        A few adjustments were made to train the model on matrices instead of tensors.
        This version is not meant to be trained on image datasets.

        - Original github repo: https://github.com/donggong1/memae-anomaly-detection
        - Paper citation:
            @inproceedings{
            gong2019memorizing,
            title={Memorizing Normality to Detect Anomaly: Memory-augmented Deep Autoencoder for Unsupervised Anomaly Detection},
            author={Gong, Dong and Liu, Lingqiao and Le, Vuong and Saha, Budhaditya and Mansour, Moussa Reda and Venkatesh, Svetha and Hengel, Anton van den},
            booktitle={IEEE International Conference on Computer Vision (ICCV)},
            year={2019}
            }

        Parameters
        ----------
        D: Feature space dimension
        L: Latent space dimension
        mem_dim: Dimension of the memory matrix
        shrink_thres: The shrink threshold used in the memory module
        device: The Torch-compatible device used during training
        """
        super(MemAutoEncoder, self).__init__()
        self.D = D
        self.L = L
        self.encoder = nn.Sequential(
            nn.Linear(D, 60),
            nn.Tanh(),
            nn.Linear(60, 30),
            nn.Tanh(),
            nn.Linear(30, 10),
            nn.Tanh(),
            nn.Linear(10, L)
        )  # .to(device)
        self.mem_rep = MemoryUnit(mem_dim, L, shrink_thres, device=device).to(device)
        self.decoder = nn.Sequential(
            nn.Linear(L, 10),
            nn.Tanh(),
            nn.Linear(10, 30),
            nn.Tanh(),
            nn.Linear(30, 60),
            nn.Tanh(),
            nn.Linear(60, D)
        )  # .to(device)

    def forward(self, x):
        f_e = self.encoder(x)
        f_mem, att = self.mem_rep(f_e)
        f_d = self.decoder(f_mem)
        return f_d, att

    def get_params(self):
        return {
            'L': self.L,
            'D': self.D
        }
