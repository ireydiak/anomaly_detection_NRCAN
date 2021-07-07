import torch
import numpy as np
from sklearn.cluster import KMeans
from src.datamanager import DataManager
from tqdm import tqdm
from src.model.MLAD import MLAD


class MLADTrainManager:
    def __init__(self, model: MLAD, dm: DataManager, optim, use_cuda: bool = False, **kwargs):
        device_name = 'cuda:0' if use_cuda else 'cpu'
        self.device = torch.device(device_name)
        self.model = model.to(self.device)
        self.dm = dm
        self.optim = optim(self.model)
        self.use_cuda = use_cuda
        self.K = kwargs.get('K', 4)
        self.L = kwargs.get('L', None)
        self.train_set = None
        self.batch_size = None

    def initialize(self, X: torch.Tensor):
        latent_features = self.model.common_net.forward(X)
        clusters = KMeans(n_clusters=self.K, random_state=0).fit(latent_features)
        labels = clusters.labels_
        for k in range(0, self.K):
            idx = np.where(labels == k)
            x_1 = X[idx, :]
            np.random.shuffle(idx)
            x_2 = X[idx, :]

    def fit_clusters(self, X: torch.Tensor):
        Z = self.model.common_net.forward(X)
        clusters = KMeans(n_clusters=self.K, random_state=0).fit(Z)
        return clusters.labels_

    def create_batches(self):

        pass

    def create_samples(self, clusters):
        X_1 = X_2 = None
        N = self.batch_size
        for k in range(0, self.K):
            [coding_idx] = np.where(clusters == k)
            X_1 = torch.cat((X_1, self.train_set[coding_idx, :])) if X_1 else self.train_set[coding_idx, :]
            np.random.shuffle(coding_idx)
            # TODO: we run the risk of training the model on the same data
            X_2 = torch.cat((X_2, self.train_set[coding_idx, :])) if X_2 else self.train_set[coding_idx, :]
        n_batch = np.int(len(X_1) // self.batch_size)
        return [(X_1[i * N:(i+1) * N, :], X_2[i * N:(i+1) * N, :]) for i in range(0, n_batch)]

    def train(self, n_epochs):
        train_ldr = self.dm.get_train_set()
        train_loss = 0.0
        loss_history = []

        for epoch in range(n_epochs):
            print("Epoch: {} of {}".format(epoch + 1, n_epochs))
            clusters = self.fit_clusters(self.train_set)
            batches = self.create_samples(clusters)
            with tqdm(range(len(batches))) as t:
                for i, X_1, X_2 in enumerate(batches):
                    self.optim.zero_grad()
                    loss = self.forward(X_1, X_2)
                    loss_history.append(loss)
                    loss.backward()
                    self.optim.step()
                    train_loss += loss.items()
                    t.set_postfix(loss='{:05.3f}'.format(train_loss / (i + 1)))
                    t.update()

    def forward(self, X_1: torch.Tensor, X_2: torch.Tensor):
        common_tup, gmm_tup, ex_tup, rec_tup = self.model.forward(X_1, X_2)
        return self.model.loss(common_tup, gmm_tup, ex_tup, rec_tup, X_1, X_2)