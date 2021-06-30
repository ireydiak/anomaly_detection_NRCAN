import torch
import numpy as np
from sklearn.cluster import KMeans
from src.datamanager import DataManager
from tqdm import tqdm
from src.model.MLAD import MLAD


class MLADTrainManager:
    def __init__(self, model: MLAD, dm: DataManager, optim, use_cuda: bool = False, **params):
        device_name = 'cuda:0' if use_cuda else 'cpu'
        self.device = torch.device(device_name)
        self.model = model.to(self.device)
        self.dm = dm
        self.optim = optim(self.model)
        self.use_cuda = use_cuda
        self.K = params.get('K', 4)

    def initialize(self, X: torch.Tensor):
        latent_features = self.model.common_net.forward(X)
        clusters = KMeans(n_clusters=self.K, random_state=0).fit(latent_features)
        labels = clusters.labels_
        for k in range(0, self.K):
            idx = np.where(labels == k)
            x_1 = X[idx, :]
            np.random.shuffle(idx)
            x_2 = X[idx, :]

    def train(self, n_epochs):
        train_ldr = self.dm.get_train_set()
        train_loss = 0.0

        for epoch in range(n_epochs):
            print("Epoch: {} of {}".format(epoch + 1, n_epochs))
            with tqdm(range(len(train_ldr))) as t:
                for i, data in enumerate(train_ldr):
                    self.optim.zero_grad()
                    loss = self.forward(data[0].float().to(self.device))
                    loss.backward()
                    self.optim.step()
                    t.set_postfix(loss='{:05.3f}'.format(train_loss / (i + 1)))
                    t.update()

    def forward(self, X: torch.Tensor):
        common_tup, err_tup, gmm_tup, ex_tup, rec_tup = self.model.forward(X)
        return self.model.loss(common_tup, err_tup, gmm_tup, ex_tup, rec_tup)