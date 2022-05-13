import numpy as np
import torch

from abc import ABC, abstractmethod
from typing import Union
from sklearn import metrics as sk_metrics
from torch.utils.data.dataloader import DataLoader
from torch import optim
from tqdm import trange


class BaseTrainer(ABC):

    def __init__(self, model,
                 batch_size,
                 lr: float = 1e-4,
                 n_epochs: int = 200,
                 n_jobs_dataloader: int = 0,
                 device: str = "cuda",
                 **kwargs):
        self.device = device
        self.model = model.to(device)
        self.batch_size = batch_size
        self.n_jobs_dataloader = n_jobs_dataloader
        self.n_epochs = n_epochs
        self.lr = lr
        self.weight_decay = kwargs.get('weight_decay', 0)
        self.optimizer = self.set_optimizer(self.weight_decay)

    @abstractmethod
    def train_iter(self, sample: torch.Tensor):
        pass

    @abstractmethod
    def score(self, sample: torch.Tensor):
        pass

    def after_training(self):
        """
        Perform any action after training is done
        """
        pass

    def before_training(self, dataset: DataLoader):
        """
        Optionally perform pre-training or other operations.
        """
        pass

    def set_optimizer(self, weight_decay):
        return optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=weight_decay)

    def train(self, train_loader: DataLoader, val_loader: DataLoader = None):
        self.model.train()

        self.before_training(train_loader)

        print("Started training")
        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            # if not val_loader is None:
            #     loss = self.eval(train_loader)
            #     print(f'validation score: loss:{loss:.3f}, score:{np.mean(self.test(val_loader)[1]):.3f}')
            with trange(len(train_loader)) as t:
                for sample in train_loader:
                    X = sample[0]
                    X = X.to(self.device).float()
                    # TODO handle this just for trainer DBESM
                    # if len(X) < self.batch_size:
                    #     t.update()
                    #     break

                    # Reset gradient
                    self.optimizer.zero_grad()

                    loss = self.train_iter(X)

                    # Backpropagation
                    loss.backward()
                    self.optimizer.step()

                    epoch_loss += loss.item()
                    t.set_postfix(
                        loss='{:.3f}'.format(epoch_loss / (epoch + 1)),
                        epoch=epoch + 1
                    )
                    t.update()

        self.after_training()

    def eval(self, dataset: DataLoader):
        self.model.eval()
        with torch.no_grad():
            loss = 0
            for row in dataset:
                X = row[0]
                X = X.to(self.device).float()
                loss += self.train_iter(X)
            loss /= len(dataset)
        self.model.train()

        return loss

    def test(self, dataset: DataLoader) -> Union[np.array, np.array]:
        self.model.eval()
        y_true, scores = [], []
        with torch.no_grad():
            for row in dataset:
                X, y = row[0], row[1]
                X = X.to(self.device).float()
                # if len(X) < self.batch_size:
                #     break
                score = self.score(X)
                y_true.extend(y.cpu().tolist())
                scores.extend(score.cpu().tolist())
        self.model.train()

        return np.array(y_true), np.array(scores)

    def get_params(self) -> dict:
        return {
            "learning_rate": self.lr,
            "epochs": self.n_epochs,
            "batch_size": self.batch_size,
            **self.model.get_params()
        }

    def predict(self, scores: np.array, thresh: float):
        return (scores >= thresh).astype(int)

    def evaluate(self, combined_scores, y_test: np.array, test_scores: np.array, threshold: float,
                 pos_label: int = 1) -> dict:
        res = {"Precision": -1, "Recall": -1, "F1-Score": -1, "AUROC": -1, "AUPR": -1}

        thresh = np.percentile(combined_scores, threshold)
        y_pred = self.predict(test_scores, thresh)
        res["Precision"], res["Recall"], res["F1-Score"], _ = sk_metrics.precision_recall_fscore_support(
            y_test, y_pred, average='binary', pos_label=pos_label
        )

        res["AUROC"] = sk_metrics.roc_auc_score(y_test, test_scores)
        res["AUPR"] = sk_metrics.average_precision_score(y_test, test_scores)
        return res


class BaseShallowTrainer(ABC):

    def __init__(self, model,
                 batch_size,
                 lr: float = 1e-4,
                 n_epochs: int = 200,
                 n_jobs_dataloader: int = 0,
                 device: str = "cuda"):
        """
        Parameters are mostly ignored but kept for better code consistency

        Parameters
        ----------
        model
        batch_size
        lr
        n_epochs
        n_jobs_dataloader
        device
        """
        self.device = None
        self.model = model
        self.batch_size = None
        self.n_jobs_dataloader = None
        self.n_epochs = None
        self.lr = None

    def train(self, dataset: DataLoader):
        self.model.clf.fit(dataset.dataset.dataset.X)

    def score(self, sample: torch.Tensor):
        return self.model.predict(sample.numpy())

    def test(self, dataset: DataLoader) -> Union[np.array, np.array]:
        y_true, scores = [], []
        for row in dataset:
            X, y = row[0], row[1]
            score = self.score(X)
            y_true.extend(y.cpu().tolist())
            scores.extend(score)

        return np.array(y_true), np.array(scores)

    def get_params(self) -> dict:
        return {
            **self.model.get_params()
        }

    def predict(self, scores: np.array, thresh: float):
        return (scores >= thresh).astype(int)

    def evaluate(self, y_true: np.array, scores: np.array, threshold: float, pos_label: int = 1) -> dict:
        res = {"Precision": -1, "Recall": -1, "F1-Score": -1, "AUROC": -1, "AUPR": -1}

        thresh = np.percentile(scores, threshold)
        y_pred = self.predict(scores, thresh)
        res["Precision"], res["Recall"], res["F1-Score"], _ = sk_metrics.precision_recall_fscore_support(
            y_true, y_pred, average='binary', pos_label=pos_label
        )

        res["AUROC"] = sk_metrics.roc_auc_score(y_true, scores)
        res["AUPR"] = sk_metrics.average_precision_score(y_true, scores)
        return res
