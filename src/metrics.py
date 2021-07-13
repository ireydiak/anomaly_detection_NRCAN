from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def accuracy_precision_recall_f1_scores(y_true, y_pred):
    return accuracy_score(y_true, y_pred),\
           precision_score(y_true, y_pred),\
           recall_score(y_true, y_pred),\
           f1_score(y_true, y_pred)
