import numpy as np
import torch

from abc import ABC, abstractmethod
from typing import Union
from sklearn import metrics as sk_metrics
from torch.utils.data.dataloader import DataLoader
from torch import optim
from tqdm import trange
from src.model.base import BaseModel


class BaseTrainer(ABC):

    def __init__(self,
                 model,
                 batch_size,
                 lr: float = 1e-4,
                 n_epochs: int = 200,
                 n_jobs_dataloader: int = 0,
                 device: str = "cuda",
                 anomaly_label=1,
                 ckpt_fname: str = None,
                 **kwargs):
        self.device = device
        self.model = model.to(device)
        self.batch_size = batch_size
        self.n_jobs_dataloader = n_jobs_dataloader
        self.n_epochs = n_epochs
        self.lr = lr
        self.anomaly_label = anomaly_label
        self.weight_decay = kwargs.get('weight_decay', 0)
        self.optimizer = self.set_optimizer(weight_decay=kwargs.get('weight_decay', 0))
        self.cur_epoch = None
        self.use_cuda = "cuda" in device
        self.ckpt_fname = ckpt_fname

    @staticmethod
    def load_from_file(fname: str, trainer, model: BaseModel, device: str = None):
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(fname, map_location=device)
        metric_values = ckpt["metric_values"]
        model = model.load_from_ckpt(ckpt, model)
        trainer.model = model
        trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        trainer.metric_values = metric_values

        return trainer, model

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

    def train(self, dataset: DataLoader):
        self.model.train(mode=True)
        self.before_training(dataset)
        assert self.model.training, "Model not in training mode. Aborting"

        print("Started training")
        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            self.cur_epoch = epoch
            with trange(len(dataset)) as t:
                for sample in dataset:
                    X, _ = sample
                    X = X.to(self.device).float()

                    # Reset gradient
                    self.optimizer.zero_grad()

                    loss = self.train_iter(X)

                    # Backpropagation
                    loss.backward()
                    self.optimizer.step()

                    epoch_loss += loss.item()
                    t.set_postfix(
                        loss='{:05.3f}'.format(epoch_loss/(epoch + 1)),
                        epoch=epoch + 1
                    )
                    t.update()
            if self.ckpt_fname and epoch % 5 == 0:
                self.save_ckpt(self.ckpt_fname + "_epoch={}.pt".format(epoch+1))

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

    def evaluate(
            self,
            y_true: np.array,
            scores: np.array,
            thresh: float = None,
            pos_label: int = 1
    ) -> dict:
        res = {"Precision": -1, "Recall": -1, "F1-Score": -1, "AUROC": -1, "AUPR": -1}

        thresh = thresh or (y_true == self.anomaly_label) / len(y_true)
        thresh = np.percentile(scores, thresh)
        y_pred = self.predict(scores, thresh)
        res["Precision"], res["Recall"], res["F1-Score"], _ = sk_metrics.precision_recall_fscore_support(
            y_true, y_pred, average='binary', pos_label=pos_label
        )
        res["AUROC"] = sk_metrics.roc_auc_score(y_true, scores)
        res["AUPR"] = sk_metrics.average_precision_score(y_true, scores)
        return res

    def save_ckpt(self, fname: str):
        torch.save({
            "cur_epoch": self.cur_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, fname)


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
            X, y = row
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
