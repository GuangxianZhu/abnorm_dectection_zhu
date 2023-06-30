from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class Label:
    ok: int
    ng: int

    def __init__(self, ok: int):
        self.ok = ok
        self.ng = -(ok - 1)  # 0->1, 1->0


def accuracy(predictions: list[Label], labels: list[Label]) -> float:
    """
    calculates and returns the accuracy metric on a list of predicted labels and true labels

    Parameters
    ----------
    predictions : list[Label]
        list of label objects containing the predicted label OK or NG
    labels : list[Label]
        list of label objects containing the true label OK or NG

    Returns
    -------
    float
        returns accuracy of predictions

    """
    if len(predictions) != len(labels):
        raise ValueError("length of predictions and labels are not same")
    correct_predictions = 0
    total_predictions = 0

    for prediction, label in zip(predictions, labels):
        if prediction == label:
            correct_predictions += 1
        total_predictions += 1

    acc = correct_predictions / total_predictions

    return acc


def iou(mask_predictions: list[np.array], mask_labels: list[np.array]) -> float:
    """
    calculates and returns a list of IoU's for each sample in a list of predicted binary masks and labelled binary masks

    Parameters
    ----------
    mask_predictions : list[np.array]
        list of 0/1 binary arrays that represent the predicted mask with 1 values
    mask_labels : list[np.array]
        list of 0/1 binary arrays that represent the labelled mask with 1 values

    Returns
    -------
    list[float]
        returns a list of calculated IoU scores

    """
    if len(mask_predictions) != len(mask_labels):
        raise ValueError("length of mask_predictions and mask_labels are not same")
    ious = []
    for prediction, label in zip(mask_predictions, mask_labels):
        pred_area = np.count_nonzero(prediction == 1)
        label_area = np.count_nonzero(label == 1)
        intersect_area = np.count_nonzero(np.logical_and(prediction, label))

        if (pred_area + label_area - intersect_area) == 0:
            ioveru = 0
        else:
            ioveru = intersect_area / (pred_area + label_area - intersect_area)
        ious.append(ioveru)

    return np.mean(ious)


# ZHU --------------------------------------------------------------------------------------------------------------
# AUC(high)

from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

def auc(predictions: list[float], labels: list[Label]) -> float:
    """
    Calculates and returns the AUC (Area Under the Curve) on a list of predicted scores and true labels.
    
    Parameters
    ----------
    predictions : list[float]
        list of predicted scores
    labels : list[Label]
        list of Label objects containing the true labels

    Returns
    -------
    float
        returns AUC of predictions

    Raises
    -------
    ValueError
        If length of predictions and labels are not same.
    """
    
    #print('auc using v1.1')
    if len(predictions) != len(labels):
        raise ValueError("length of predictions and labels are not same")

    # Extract the 'ok' attribute from the labels
    true_labels = [label.ok for label in labels]

    # Calculate the AUC score
    auc_score = roc_auc_score(true_labels, predictions)

    return auc_score


# over-detection rate(low)

from sklearn.metrics import roc_curve

def odr(predictions: list[float], labels: list[Label]) -> float:
    
    df_dict = {'Anomaly':predictions, 'y_true':[label.ok for label in labels]} # 0: ok, 1: ng
    csv = pd.DataFrame(df_dict)
    min_fp = csv[csv.y_true == 1].Anomaly.min()
    zero_threshold = min_fp - 0.0001
    # print(f'zero_threshold: {zero_threshold}')
    # print('ok imgs')
    # print(len(csv[(csv.y_true == 0)]))
    # print('ok imgs over zero threshold')
    # print(len(csv[(csv.y_true == 0)&(csv.Anomaly > zero_threshold)]))
    # csv.to_csv('odr.csv')
    zero_odr = len(csv[(csv.y_true == 0)&(csv.Anomaly > zero_threshold)])/len(csv[(csv.y_true == 0)])
    return zero_odr

def balanced_metrics(predictions: list[float], labels: list[Label]):
    if len(predictions) != len(labels):
        raise ValueError("length of predictions and labels are not same")
    true_labels = [label.ok for label in labels]
    fpr, tpr, thresholds = roc_curve(true_labels, predictions)
    best_idx = np.argmax(tpr-fpr)
    best_threshold = round(thresholds[best_idx],4)
    best_fpr, best_tpr = round(fpr[best_idx],4), round(tpr[best_idx],4)
    gu_preds = []
    for label, anomaly in zip(labels, predictions):
        if label.ok == 0:
            gu_preds.append(1 if anomaly > best_threshold else 0)
    odr = round(sum(gu_preds)/len(gu_preds),4)
    # check true_labels and predictions type
    assert isinstance(true_labels, list) or isinstance(true_labels, np.ndarray), "true_labels should be list or np.ndarray"
    assert isinstance(gu_preds, list) or isinstance(gu_preds, np.ndarray), "predictions should be list or np.ndarray"
    ok_acc = round(1-best_fpr,4)
    ng_acc = round(best_tpr,4)
    return {'balanced_threshold':best_threshold, 'ok_acc':ok_acc, 'ng_acc':ng_acc, 'balanced_odr':odr}

def make_cls_report(predictions: list[float], labels: list[Label]):
    if len(predictions) != len(labels):
        raise ValueError("length of predictions and labels are not same")
    true_labels = [label.ok for label in labels]
    fpr, tpr, thresholds = roc_curve(true_labels, predictions)
    best_idx = np.argmax(tpr-fpr)
    best_threshold = round(thresholds[best_idx],4)
    gu_preds = [1 if prediction > best_threshold else 0 for prediction in predictions] #todo FIX
    # check true_labels and predictions type
    assert isinstance(true_labels, list) or isinstance(true_labels, np.ndarray), "true_labels should be list or np.ndarray"
    assert isinstance(gu_preds, list) or isinstance(gu_preds, np.ndarray), "predictions should be list or np.ndarray"
    cls_report = classification_report(true_labels, gu_preds)
    return cls_report

def make_confu_matrix(predictions: list[float], labels: list[Label]):
    if len(predictions) != len(labels):
        raise ValueError("length of predictions and labels are not same")
    true_labels = [label.ok for label in labels]
    fpr, tpr, thresholds = roc_curve(true_labels, predictions)
    best_idx = np.argmax(tpr-fpr)
    best_threshold = round(thresholds[best_idx],4)
    # check true_labels and predictions type
    gu_preds = [1 if prediction > best_threshold else 0 for prediction in predictions] #todo FIX
    assert isinstance(true_labels, list) or isinstance(true_labels, np.ndarray), "true_labels should be list or np.ndarray"
    assert isinstance(gu_preds, list) or isinstance(gu_preds, np.ndarray), "predictions should be list or np.ndarray"
    confu_matrix = confusion_matrix(true_labels, gu_preds)
    return confu_matrix

def auc_odr(predictions: list[float], labels: list[int]) -> float:

    if len(predictions) != len(labels):
        raise ValueError("length of predictions and labels are not same")

    return auc(predictions, labels), odr(predictions, labels), balanced_metrics(predictions, labels), make_cls_report(predictions, labels), make_confu_matrix(predictions, labels)
