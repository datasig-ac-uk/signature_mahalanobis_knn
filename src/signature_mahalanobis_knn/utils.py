from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    roc_auc: float | None = None,
    title: str = "",
) -> float:
    """
    Plot the ROC curve given the true labels and the scores.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_score : np.ndarray
        Target scores.
    roc_auc : float | None, optional
        ROC AUC, by default None.
        If None, then it will be computed.
    title : str, optional
        Title for the ROC curve plot, by default "".

    Returns
    -------
    float
        ROC AUC score.
    """

    # compute and plot metrics
    fp_rate, tp_rate, _ = roc_curve(y_true, y_score)
    if roc_auc is None:
        roc_auc = roc_auc_score(y_true, y_score)

    plt.title(f"Receiver Operating Characteristic {title}")
    plt.plot(fp_rate, tp_rate, "b", label=f"AUC = {round(roc_auc, 2)}")
    plt.legend(loc="lower right")
    plt.plot([0, 1], [0, 1], "r--")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.show()

    return roc_auc
