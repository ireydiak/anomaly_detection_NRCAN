from random import shuffle
from typing import Tuple

import numpy as np
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


def estimate_optimal_threshold(test_score, y_test, pos_label=1, nq=100, ratio=None):
    ratio = ratio or 100 * sum(y_test == 0) / len(y_test)
    print(f"Ratio of normal data:{ratio}")
    q = np.linspace(ratio - 5, min(ratio + 5, 100), nq)
    thresholds = np.percentile(test_score, q)

    f1 = np.zeros(shape=nq)
    r = np.zeros(shape=nq)
    p = np.zeros(shape=nq)
    auc = np.zeros(shape=nq)
    aupr = np.zeros(shape=nq)
    qis = np.zeros(shape=nq)

    for i, (thresh, qi) in enumerate(zip(thresholds, q)):
        # print(f"Threshold :{thresh:.3f}--> {qi:.3f}")
        # Prediction using the threshold value
        precision, recall, f_score, roc, avgpr, y_pred = compute_metrics(test_score, y_test, thresh, pos_label)

        # print(f"qi:{qi:.3f} ==> p:{precision:.3f}  r:{recall:.3f}  f1:{f_score:.3f}")
        f1[i] = f_score * 100
        r[i] = recall * 100
        p[i] = precision * 100
        auc[i] = roc * 100
        aupr[i] = avgpr * 100
        qis[i] = qi

    arm = np.argmax(f1)
    _, _, _, _, _, y_pred = compute_metrics(test_score, y_test, thresholds[arm], pos_label)

    return {
        "Precision": p[arm],
        "Recall": r[arm],
        "F1-Score": f1[arm],
        "AUPR": aupr[arm],
        "AUROC": auc[arm],
        "Thresh_star": thresholds[arm],
        "Quantile_star": qis[arm]
    }, y_pred


def score_recall_precision_w_threshold(scores, y_true, threshold=None, pos_label=1) -> Tuple[dict, np.ndarray]:
    if threshold is None:
        anomaly_ratio = (y_true == pos_label).sum() / len(y_true)
        threshold = threshold or int(np.ceil((1 - anomaly_ratio) * 100))
    thresh = np.percentile(scores, threshold)

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
