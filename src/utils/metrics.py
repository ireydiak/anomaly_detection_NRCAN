import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score, average_precision_score, roc_auc_score


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
    q = np.linspace(0, 99, 100)
    thresholds = np.percentile(combined_score, q)

    result_search = []
    confusion_matrices = []
    for thresh, qi in zip(thresholds, q):
        print(f"Threshold :{thresh:.3f}--> {qi:.3f}")

        # Prediction using the threshold value
        y_pred = (test_score >= thresh).astype(int)
        y_true = test_labels.astype(int)

        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f_score, _ = precision_recall_fscore_support(y_true, y_pred,
                                                                        average='binary',
                                                                        pos_label=pos_label)
        cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
        confusion_matrices.append(cm)
        result_search.append([accuracy, precision, recall, f_score])

        print(f"\nAccuracy:{accuracy:.3f}, "
              f"Precision:{precision:.3f}, "
              f"Recall:{recall:.3f}, "
              f"F-score:{f_score:.3f}, "
              f"\nconfusion-matrix: {cm}\n\n")


def score_recall_precision_w_thresold(combined_score, test_score, test_labels, pos_label=1, threshold=80.0):
    thresh = np.percentile(combined_score, threshold)
    print("Threshold :", thresh)

    # Prediction using the threshold value
    y_pred = (test_score >= thresh).astype(int)
    y_true = test_labels.astype(int)

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f_score, _ = precision_recall_fscore_support(y_true, y_pred, average='binary',
                                                                            pos_label=pos_label)
    res = {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1-Score": f_score}

    res = precision_recall_f1_roc_pr(y_true, y_pred, pos_label, threshold)
    return res

def precision_recall_f1_roc_pr(y_true: np.array, scores: np.array, pos_label: int = 1, threshold: int = 80) -> dict:
    res = {"Precision": -1, "Recall": -1, "F1-Score": -1, "AUROC": -1, "AUPR": -1}

    thresh = np.percentile(scores, threshold)
    y_pred = (scores >= thresh).astype(int)
    res["Precision"], res["Recall"], res["F1-Score"], _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', pos_label=pos_label
    )
    res["AUROC"] = roc_auc_score(y_true, scores)
    res["AUPR"] = average_precision_score(y_true, scores)

    return res
