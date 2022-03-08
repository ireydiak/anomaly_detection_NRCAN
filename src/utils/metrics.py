import numpy as np
from sklearn import metrics as sk_metrics


def estimate_optimal_threshold(combined_scores, test_score, y_test, pos_label=1):
    q = np.linspace(0, 99, 100)
    thresholds = np.percentile(combined_scores, q)
    res = {"Precision": -1, "Recall": -1, "F1-Score": -1, "AUROC": -1, "AUPR": -1, "Thresh_star": -1}
    y_true = y_test.astype(int)
    res["AUPR"] = sk_metrics.average_precision_score(y_true, test_score)
    res["AUROC"] = sk_metrics.roc_auc_score(y_true, test_score)

    for thresh, qi in zip(thresholds, q):
        # Prediction using the threshold value
        y_pred = (test_score >= thresh).astype(int)

        precision, recall, f_score, _ = sk_metrics.precision_recall_fscore_support(
            y_true, y_pred, average="binary", pos_label=pos_label
        )

        if f_score > res["F1-Score"]:
            res["F1-Score"] = f_score
            res["Precision"] = precision
            res["Recall"] = recall
            res["Thresh_star"] = thresh

    return res


def score_recall_precision(combined_scores, test_score, y_test, pos_label=1):
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


def score_recall_precision_w_thresold(combined_score, test_score, test_labels, threshold, pos_label=1):
    thresh = np.percentile(combined_score, threshold)
    print("Threshold :", thresh)

    # Prediction using the threshold value
    y_pred = (test_score >= thresh).astype(int)
    y_true = test_labels.astype(int)

    precision, recall, f_score, _ = sk_metrics.precision_recall_fscore_support(
        y_true, y_pred, average='binary', pos_label=pos_label
    )

    return {"Precision": precision,
            "Recall": recall,
            "F1-Score": f_score,
            "AUROC": sk_metrics.roc_auc_score(y_true, test_score),
            "AUPR": sk_metrics.average_precision_score(y_true, test_score)}


def precision_recall_f1_roc_pr(y_true: np.array, scores: np.array, pos_label: int = 1, threshold: int = 80) -> dict:
    res = {"Precision": -1, "Recall": -1, "F1-Score": -1, "AUROC": -1, "AUPR": -1}

    thresh = np.percentile(scores, threshold)
    y_pred = (scores >= thresh).astype(int)
    res["Precision"], res["Recall"], res["F1-Score"], _ = sk_metrics.precision_recall_fscore_support(
        y_true, y_pred, average='binary', pos_label=pos_label
    )
    res["AUROC"] = sk_metrics.roc_auc_score(y_true, scores)
    res["AUPR"] = sk_metrics.average_precision_score(y_true, scores)

    return res
