import numpy as np

from typing import Tuple, List
from sklearn import metrics as sk_metrics


def compute_metrics(test_score, y_test, thresh, pos_label=1):
    """
    This function compute metrics for a given threshold

    Parameters
    ----------
    test_score
    y_test
    thresh
    pos_label

    Returns
    -------

    """
    y_pred = (test_score >= thresh).astype(int)
    y_true = y_test.astype(int)

    precision, recall, f_score, _ = sk_metrics.precision_recall_fscore_support(
        y_true, y_pred, average='binary', pos_label=pos_label
    )
    avgpr = sk_metrics.average_precision_score(y_true, test_score)
    roc = sk_metrics.roc_auc_score(y_true, test_score)

    return precision, recall, f_score, roc, avgpr, y_pred


def estimate_optimal_threshold(scores, y_true, ratio=None, pos_label=1, nq=100):
    ratio = ratio or 100 * sum(y_true == 0) / len(y_true)
    print(f"Ratio of normal data:{ratio}")
    q = np.linspace(ratio - 5, min(ratio + 5, 100), nq)
    thresholds = np.percentile(scores, q)

    f1 = np.zeros(shape=nq)
    r = np.zeros(shape=nq)
    p = np.zeros(shape=nq)
    auc = np.zeros(shape=nq)
    aupr = np.zeros(shape=nq)
    qis = np.zeros(shape=nq)

    for i, (thresh, qi) in enumerate(zip(thresholds, q)):
        # print(f"Threshold :{thresh:.3f}--> {qi:.3f}")
        # Prediction using the threshold value
        precision, recall, f_score, roc, avgpr, y_pred = compute_metrics(scores, y_true, thresh, pos_label)

        # print(f"qi:{qi:.3f} ==> p:{precision:.3f}  r:{recall:.3f}  f1:{f_score:.3f}")
        f1[i] = f_score * 100
        r[i] = recall * 100
        p[i] = precision * 100
        auc[i] = roc * 100
        aupr[i] = avgpr * 100
        qis[i] = qi

    arm = np.argmax(f1)
    _, _, _, _, _, y_pred = compute_metrics(scores, y_true, thresholds[arm], pos_label)

    return {
               "Precision": p[arm],
               "Recall": r[arm],
               "F1-Score": f1[arm],
               "AUPR": aupr[arm],
               "AUROC": auc[arm],
               "Thresh": thresholds[arm],
               "Quantile_star": qis[arm]
           }, y_pred


def score_recall_precision_w_threshold(scores, y_true, ratio=None, pos_label=1) -> Tuple[dict, np.ndarray]:
    ratio = ratio or ((y_true == 0).sum() / len(y_true)) * 100
    thresh = np.percentile(scores, ratio)
    # Prediction using the threshold value
    y_pred = (scores >= thresh).astype(int)
    y_true = y_true.astype(int)

    precision, recall, f_score, _ = sk_metrics.precision_recall_fscore_support(
        y_true, y_pred, average='binary', pos_label=pos_label
    )

    return {"Precision": precision * 100,
            "Recall": recall * 100,
            "F1-Score": f_score * 100,
            "AUROC": sk_metrics.roc_auc_score(y_true, scores) * 100,
            "AUPR": sk_metrics.average_precision_score(y_true, scores) * 100,
            "Thresh": thresh
            }, y_pred


def per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray, normal_label: str):
    results = dict()
    for label in np.unique(labels):
        # select current label
        mask = (labels == label)
        n = mask.sum()
        # normal data expect binary label 0
        if label == normal_label:
            c = ((y_true[mask]) == 0 & (y_pred[mask] == 0)).sum()
        # abnormal data expect binary label 1
        else:
            c = ((y_true[mask]) == 1 & (y_pred[mask] == 1)).sum()
        # compute per-class accuracy
        results[label] = (c / n) * 100
    return results


class AggregatorDict(dict):
    def __init__(self, keys: List[str]):
        super(AggregatorDict, self).__init__()
        self.keys = keys
        for key in keys:
            super().__setitem__(key, [])

    def __setitem__(self, __k, __v) -> None:
        self[__k].append(__v)

    def add(self, __d) -> None:
        for k, v in __d.items():
            if k in set(self.keys):
                self.__setitem__(k, v)

    def aggregate(self) -> dict:
        new_dict = {}
        for k, v in self.items():
            new_dict[k] = "{:2.4f} ({:2.4f})".format(np.mean(v), np.std(v))
        return new_dict
