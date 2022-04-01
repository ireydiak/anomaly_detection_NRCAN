import numpy as np
from sklearn import metrics as sk_metrics


def estimate_optimal_threshold(combined_scores, test_score, y_test, pos_label=1, nq=100):
    # def score_recall_precision(combined_score, test_score, test_labels, pos_label=1, nq=100):
    ratio = 100 * sum(y_test == 0) / len(y_test)
    q = np.linspace(ratio - 5, min(ratio + 5, 100), nq)
    thresholds = np.percentile(test_score, q)

    result_search = []
    confusion_matrices = []
    f1 = np.zeros(shape=nq)
    r = np.zeros(shape=nq)
    p = np.zeros(shape=nq)
    auc = np.zeros(shape=nq)
    aupr = np.zeros(shape=nq)
    qis = np.zeros(shape=nq)

    for i, (thresh, qi) in enumerate(zip(thresholds, q)):
        # print(f"Threshold :{thresh:.3f}--> {qi:.3f}")
        # Prediction using the threshold value
        y_pred = (test_score >= thresh).astype(int)
        y_true = y_test.astype(int)

        accuracy = sk_metrics.accuracy_score(y_true, y_pred)
        precision, recall, f_score, _ = sk_metrics.precision_recall_fscore_support(
            y_true, y_pred, average='binary', pos_label=pos_label
        )
        avgpr = sk_metrics.average_precision_score(y_true, test_score)
        roc = sk_metrics.roc_auc_score(y_true, test_score)
        cm = sk_metrics.confusion_matrix(y_true, y_pred, labels=[1, 0])
        confusion_matrices.append(cm)
        result_search.append([accuracy, precision, recall, f_score])
        f1[i] = f_score
        r[i] = recall
        p[i] = precision
        auc[i] = roc
        aupr[i] = avgpr
        qis[i] = qi

    arm = np.argmax(f1)

    return {
        "Precision": p[arm],
        "Recall": r[arm],
        "F1-Score": f1[arm],
        "AUPR": aupr[arm],
        "AUROC": auc[arm],
        "Thresh_star": thresholds[arm],
        "Quantile_star": thresholds[arm]
    }


def score_recall_precision_w_thresold(combined_score, test_score, test_labels, threshold, pos_label=1):
    thresh = np.percentile(combined_score, threshold)

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

