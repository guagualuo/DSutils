""" This module implements visualisation tools for classification problems """
import matplotlib.pyplot as plt
import numpy as np

from utils import _to_numpy


# https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
def plot_roc_curve(y_true,
                   y_score,
                   sample_weight=None,
                   multiclass: bool = False,
                   ax=None,
                   ):
    """
    Plot ROC curve of classification.

    For multiclass prediction, consider both 'macro' and 'micro' averaging of the ROC curve.
    y_true and y_score are assumed to be in the sense of OVR (one-versus-rest) fashion.
    In the 'macro' average, the ROC curves is interpolated at all false positive rates and then averaged.
    In the 'micro' average, y is set to be OVR form, and with y_score together,
    flattened to a 1D array to compute the ROC curve.

    Parameters
    -----
    y_true: array of shape (n_samples, ) or (n_samples, n_classes)
        True labels.

    y_score: array of shape (n_samples, ) or (n_samples, n_classes)
        Decision function or probability of prediction.

    sample_weight: array of shape (n_samples, ),
        Sample weights.

    multiclass: bool = False

    ax:

    Returns
    -----
    ax

    """
    from sklearn.metrics import roc_curve, auc

    y_true = _to_numpy(y_true)
    y_score = _to_numpy(y_score)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    if multiclass:
        assert len(y_score.shape) == 2
        n_classes = y_score.shape[1]

        fpr, recall = {}, {}
        roc_auc = {}
        for i in range(n_classes):
            fpr[i], recall[i], _ = roc_curve(y_true[:, i], y_score[:, i], sample_weight=sample_weight)
            roc_auc[i] = auc(fpr[i], recall[i])

        # micro average
        fpr['micro'], recall['micro'], _ = roc_curve(y_true.ravel(), y_score.ravel(), sample_weight=sample_weight)
        roc_auc['micro'] = auc(fpr['micro'], recall['micro'])

        # macro average
        fpr['macro'] = sorted(np.unique(np.concatenate([fpr[i] for i in range(n_classes)])))
        recall['macro'] = np.zeros_like(fpr['macro'])
        for i in range(n_classes):
            recall['macro'] += np.interp(fpr['macro'], fpr[i], recall[i]) / n_classes
        roc_auc['macro'] = auc(fpr['micro'], recall['micro'])

        ax.plot(fpr["micro"], recall["micro"],
                label=f"micro-average ROC curve (area = {roc_auc['micro']:.2f})",
                color="deeppink", linestyle=":", linewidth=4,)
        ax.plot(fpr["macro"], recall["macro"],
                label=f"macro-average ROC curve (area = {roc_auc['macro']:.2f})",
                color="navy", linestyle=":", linewidth=4,)

        colors = plt.get_cmap('tab10', min(n_classes, 10)).colors
        for i, c in zip(range(n_classes), colors):
            plt.plot(fpr[i], recall[i], color=c, lw=2,
                     label=f"ROC curve of class {i} (area = {roc_auc[i]:.2f})")

    else:
        fpr, recall, _ = roc_curve(y_true, y_score, sample_weight=sample_weight)
        roc_auc = auc(fpr, recall)
        ax.plot(fpr, recall, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")

    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")


def plot_precison_recall(y_true,
                         y_score,
                         sample_weight=None,
                         multiclass: bool = False,
                         ax=None,
                         ):
    """
    Plot precision-recall curve of classification.

    For multiclass prediction, consider both 'macro' and 'micro' averaging of the precision-recall curve.
    y_true and y_score are assumed to be in the sense of OVR (one-versus-rest) fashion.
    In the 'macro' average, the curves is interpolated at all recall values and then averaged.
    In the 'micro' average, y is set to be OVR form, and with y_score together,
    flattened to a 1D array to compute the precision-recall curve.

    Parameters
    -----
    y_true: array of shape (n_samples, ) or (n_samples, n_classes)
        True labels.

    y_score: array of shape (n_samples, ) or (n_samples, n_classes)
        Decision function or probability of prediction.

    sample_weight: array of shape (n_samples, ),
        Sample weights.

    multiclass: bool = False

    ax:

    Returns
    -----
    ax

    """
    from sklearn.metrics import precision_recall_curve, average_precision_score

    y_true = _to_numpy(y_true)
    y_score = _to_numpy(y_score)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    if multiclass:
        assert len(y_score.shape) == 2
        n_classes = y_score.shape[1]

        recall, precision = {}, {}
        average_precision = {}
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(y_true[:, i], y_score[:, i], sample_weight=sample_weight)
            average_precision[i] = average_precision_score(y_true[:, i], y_score[:, i])

        # micro average
        precision['micro'], recall['micro'], _ = precision_recall_curve(y_true.ravel(), y_score.ravel(), sample_weight=sample_weight)
        average_precision['micro'] = average_precision_score(y_true.ravel(), y_score.ravel())

        # macro average
        recall['macro'] = sorted(np.unique(np.concatenate([recall[i] for i in range(n_classes)])), reverse=True)
        precision['macro'] = np.zeros_like(recall['macro'])
        for i in range(n_classes):
            precision['macro'] += np.interp(recall['macro'][::-1], recall[i][::-1], precision[i][::-1])[::-1] / n_classes
        average_precision['macro'] = -np.sum(np.diff(recall['macro']) * np.array(precision['macro'])[:-1])

        ax.plot(recall["micro"], precision["micro"],
                label=f"micro-average precision-recall curve (AP = {average_precision['micro']:.2f})",
                color="deeppink", linestyle=":", linewidth=4,)
        ax.plot(recall["macro"], precision["macro"],
                label=f"macro-average precision-recall curve (AP = {average_precision['macro']:.2f})",
                color="navy", linestyle=":", linewidth=4,)

        colors = plt.get_cmap('tab10', min(n_classes, 10)).colors
        for i, c in zip(range(n_classes), colors):
            plt.plot(recall[i], precision[i], color=c, lw=2,
                     label=f"precision-recall curve of class {i} (AP = {average_precision[i]:.2f})")

    else:
        precision, recall, _ = precision_recall_curve(y_true, y_score, sample_weight=sample_weight)
        average_precision = average_precision_score(y_true, y_score)
        ax.plot(recall, precision, color="darkorange", lw=2, label=f"precision-recall curve (AP = {average_precision:.2f})")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="upper right")
