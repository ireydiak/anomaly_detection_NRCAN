import numpy as np
from sklearn import metrics as sk_metrics


def accuracy(outputs, labels):
    """
    Computes the accuracy of the model
    Args:
        outputs: outputs predicted by the model
        labels: real outputs of the data
    Returns:
        Accuracy of the model
    """
    predicted = outputs.argmax(dim=1)
    correct = (predicted == labels).sum().item()
    return correct / labels.size(0)


def score_recall_precision(combined_score, test_score, test_labels, pos_label=1):
    ratio = 100 * sum(test_labels == 0)/len(test_labels)
    nq = 100
    q = np.linspace(ratio - 5, min(ratio + 5, 100), nq)
    thresholds = np.percentile(test_score, q)

    result_search = []
    confusion_matrices = []
    f1 = np.zeros(shape=nq)
    r = np.zeros(shape=nq)
    p = np.zeros(shape=nq)

    for i, (thresh, qi) in enumerate(zip(thresholds, q)):
        # print(f"Threshold :{thresh:.3f}--> {qi:.3f}")

        # Prediction using the threshold value
        y_pred = (test_score >= thresh).astype(int)
        y_true = test_labels.astype(int)

        accuracy = sk_metrics.accuracy_score(y_true, y_pred)
        precision, recall, f_score, _ = sk_metrics.precision_recall_fscore_support(y_true, y_pred,
                                                                        average='binary',
                                                                        pos_label=pos_label)
        cm = sk_metrics.confusion_matrix(y_true, y_pred, labels=[1, 0])
        confusion_matrices.append(cm)
        result_search.append([accuracy, precision, recall, f_score])
        f1[i] = f_score
        r[i] = recall
        p[i] = precision

        # print(f"\nAccuracy:{accuracy:.3f}, "
        #       f"Precision:{precision:.3f}, "
        #       f"Recall:{recall:.3f}, "
        #       f"F-score:{f_score:.3f}, "
        #       f"\nconfusion-matrix: {cm}\n\n")
    arm = np.argmax(f1)
    print(f"\np max:{p[arm]:.3f}, r max:{r[arm]:.3f}, f1 max:{f1[arm]:.3f}\n\n")

    return dict(p_fmax=p[arm], r_fmax=r[arm], f_max=f1[arm])



def score_recall_precision_w_thresold(combined_score, test_score, test_labels, pos_label=1, threshold=80.0):
    thresh = np.percentile(test_score, threshold)
    print("Threshold :", thresh)

    # Prediction using the threshold value
    y_pred = (test_score >= thresh).astype(int)
    y_true = test_labels.astype(int)

    accuracy = sk_metrics.accuracy_score(y_true, y_pred)
    precision, recall, f_score, _ = sk_metrics.precision_recall_fscore_support(
      y_true, y_pred, average='binary', pos_label=pos_label
    )

    res = {"Accuracy": accuracy,
           "Precision": precision,
           "Recall": recall,
           "F1-Score": f_score,
           "AUROC": sk_metrics.roc_auc_score(y_true, test_score),
           "AUPR": sk_metrics.average_precision_score(y_true, test_score)}

    return res


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
