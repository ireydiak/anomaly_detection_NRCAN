import pytorch_lightning as pl
import torch
import numpy as np
from pyad.utils import metrics


class BaseLightningModel(pl.LightningModule):

    def __init__(
            self,
            weight_decay: float,
            lr: float,
            in_features: int = -1,
            n_instances: int = -1,
            threshold: float = None
    ):
        super(BaseLightningModel, self).__init__()
        assert 0. <= threshold <= 100., "`threshold` must be inclusively between 0 and 1"
        self.in_features = in_features
        self.n_instances = n_instances
        self.threshold = threshold
        # call this to save hyper-parameters to the checkpoint
        # will save children parameters as well
        self.save_hyperparameters(
            ignore=["in_features", "n_instances"]
        )

    def score(self, X: torch.Tensor, y: torch.Tensor = None):
        raise NotImplementedError

    def test_epoch_end(self, outputs) -> None:
        scores, y_true, labels = np.array([]), np.array([]), np.array([])
        for output in outputs:
            scores = np.append(scores, output["scores"].cpu().detach().numpy())
            y_true = np.append(y_true, output["y_true"].cpu().detach().numpy())
            labels = np.append(labels, output["labels"].cpu().detach().numpy())
        results, _ = metrics.score_recall_precision_w_threshold(scores, y_true, threshold=self.threshold)
        self.log("performance", results)

    def test_step(self, batch, batch_idx):
        X, y_true, labels = batch
        X = X.float()
        scores = self.score(X)

        return {
            "scores": scores,
            "y_true": y_true,
            "labels": labels
        }
