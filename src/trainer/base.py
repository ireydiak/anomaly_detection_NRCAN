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
                 device: str = "cuda"):
        self.device = device
        self.model = model.to(device)
        self.batch_size = batch_size
        self.n_jobs_dataloader = n_jobs_dataloader
        self.n_epochs = n_epochs
        self.lr = lr

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

    def set_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.lr)

    def train(self, dataset: DataLoader):
        self.model.train()

        self.before_training(dataset)

        optimizer = self.set_optimizer()

        print('Started training')
        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            with trange(len(dataset)) as t:
                for sample in dataset:
                    X, _, _ = sample
                    X = X.to(self.device).float()

                    if len(X) < self.batch_size:
                        break

                    # Reset gradient
                    optimizer.zero_grad()

                    loss = self.train_iter(X)

                    # Backpropagation
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    t.set_postfix(
                        loss='{:05.3f}'.format(epoch_loss),
                        epoch=epoch + 1
                    )
                    t.update()
        self.after_training()

    def test(self, dataset: DataLoader) -> Union[np.array, np.array]:
        self.model.eval()
        y_true, scores = [], []
        with torch.no_grad():
            for row in dataset:
                X, y = row
                X = X.to(self.device).float()
                score = self.score(X)
                y_true.extend(y.cpu().tolist())
                scores.extend(score.cpu().tolist())

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

    def estimate_optimal_threshold(self, combined_scores, test_score, y_test, pos_label=1):
        q = np.linspace(0, 99, 100)
        thresholds = np.percentile(combined_scores, q)
        res = {"Precision": -1, "Recall": -1, "F1-Score": -1, "AUROC": -1, "AUPR": -1, "Thresh_star": -1}

        for thresh, qi in zip(thresholds, q):
            # Prediction using the threshold value
            y_pred = (test_score >= thresh).astype(int)
            y_true = y_test.astype(int)

            precision, recall, f_score, _ = sk_metrics.precision_recall_fscore_support(
                y_true, y_pred, average='binary', pos_label=pos_label
            )

            if f_score > res["F1-Score"]:
                res["Precision"] = precision
                res["Recall"] = recall
                res["AUPR"] = sk_metrics.average_precision_score(y_true, test_score)
                res["AUROC"] = sk_metrics.roc_auc_score(y_true, test_score)
                res["Thresh_star"] = thresh

        return res

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

    @abstractmethod
    def score(self, sample: torch.Tensor):
        pass

    @abstractmethod
    def train(self, dataset: DataLoader):
        pass

    def test(self, dataset: DataLoader) -> Union[np.array, np.array]:
        self.model.eval()
        y_true, scores = [], []
        with torch.no_grad():
            for row in dataset:
                X, y, _ = row
                X = X.to(self.device).float()
                score = self.score(X)
                y_true.extend(y.cpu().tolist())
                scores.extend(score.cpu().tolist())

        return np.array(y_true), np.array(scores)

    def get_params(self) -> dict:
        return {
            **self.model.get_params()
        }

    def predict(self, scores: np.array, thresh: float):
        return (scores >= thresh).astype(int)

    def estimate_optimal_threshold(self, combined_scores, test_score, y_test, pos_label=1):
        q = np.linspace(0, 99, 100)
        thresholds = np.percentile(combined_scores, q)
        res = {"Precision": -1, "Recall": -1, "F1-Score": -1, "AUROC": -1, "AUPR": -1, "Thresh_star": -1}

        for thresh, qi in zip(thresholds, q):
            # Prediction using the threshold value
            y_pred = (test_score >= thresh).astype(int)
            y_true = y_test.astype(int)

            precision, recall, f_score, _ = sk_metrics.precision_recall_fscore_support(
                y_true, y_pred, average='binary', pos_label=pos_label
            )

            if f_score > res["F1-Score"]:
                res["Precision"] = precision
                res["Recall"] = recall
                res["AUPR"] = sk_metrics.average_precision_score(y_true, test_score)
                res["AUROC"] = sk_metrics.roc_auc_score(y_true, test_score)
                res["Thresh_star"] = thresh

        return res

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
