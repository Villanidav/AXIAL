import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, matthews_corrcoef


def compute_metrics(y_true, y_pred, num_classes):
    """
    Compute various classification metrics for a PyTorch model.
    """
    # Convert tensors to numpy arrays
    y_true = y_true.numpy()
    y_pred = y_pred.numpy()
    # Compute accuracy
    accuracy = accuracy_score(y_true, y_pred)
    if num_classes == 2:
        # Compute specificity and sensitivity
        specificity, sensitivity = compute_binary_metrics(y_true, y_pred)
        # Compute Matthews correlation coefficient
        mcc = matthews_corrcoef(y_true, y_pred)
        # Add metrics to dictionary
        metrics = {
            'accuracy': accuracy,
            'specificity': specificity,
            'sensitivity': sensitivity,
            'mcc': mcc
        }
        return metrics
    else:
        balanced_accuracy = compute_balanced_accuracy(y_true, y_pred)
        # Compute F1 score
        f1 = f1_score(y_true, y_pred, average='weighted')
        # Compute precision and recall and PR AUC
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        pr_auc = roc_auc_score(y_true, y_pred, average='weighted')
        # Compute Matthews correlation coefficient
        mcc = matthews_corrcoef(y_true, y_pred)
        # Add metrics to dictionary
        metrics = {
            'accuracy': accuracy,
            'balanced_accuracy': balanced_accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'pr_auc': pr_auc,
            'mcc': mcc
        }
        return metrics


def compute_balanced_accuracy(y_true, y_pred):
    """
    Compute the balanced accuracy for a multi-class classification problem.
    """
    num_classes = len(np.unique(y_true))
    class_weights = [np.sum(y_true == i) for i in range(num_classes)]
    class_accuracies = [np.mean(y_pred[y_true == i] == i) for i in range(num_classes)]
    balanced_accuracy = np.average(class_accuracies, weights=class_weights)
    return balanced_accuracy


def compute_binary_metrics(y_true, y_pred):
    """
    Compute specificity and sensitivity for binary classification.
    """
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    return specificity, sensitivity
