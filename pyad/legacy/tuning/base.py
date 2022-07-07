import torch
import numpy as np
from ray import tune as ray_tune
from sklearn.metrics import average_precision_score


class BaseTuner(ray_tune.Trainable):

    def step(self):
        train_loss = self.forward(self.train_ldr)
        y_true, scores = self.test(self.val_ldr)
        aupr = average_precision_score(y_true, scores)

        return {
            "train_loss": train_loss,
            "val_loss": scores.sum(),
            "aupr": aupr
        }

    def test(self, dataset):
        self.model.eval()
        y_true, scores = [], []
        with torch.no_grad():
            for row in dataset:
                X, y, label = row
                X = X.to(self.device).float()
                score = self.score(X)
                y_true.extend(y.cpu().tolist())
                scores.extend(score.cpu().tolist())
        self.model.train()

        return np.array(y_true), np.array(scores)

    def forward(self, dataset):
        epoch_loss = 0.
        for sample in dataset:
            X, _, _ = sample
            X = X.to(self.device).float()

            # Reset gradient
            self.optimizer.zero_grad()

            loss = self.train_iter(X)

            # Backpropagation
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
        return epoch_loss

    def train_iter(self, X: torch.Tensor):
        raise NotImplementedError

    def score(self, X: torch.Tensor):
        raise NotImplementedError

    @staticmethod
    def get_tunable_params(n_instances: int, in_features: int):
        raise NotImplementedError
