from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from signature_mahalanobis_knn.sig_mahal_knn import SignatureMahalanobisKNN


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
    plt.plot(fp_rate, tp_rate, "b", label=f"AUC = {round(roc_auc, 2)}", linewidth=2)
    plt.legend(loc="lower right")
    plt.plot([0, 1], [0, 1], "r--", linewidth=2)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.show()

    return roc_auc


def compute_auc_given_dists(
    distances_in: np.ndarray,
    distances_out: np.ndarray,
    plot: bool = False,
    title: str = "",
) -> float:
    """
    Compute ROC AUC given the distances of inliers and outliers.

    Parameters
    ----------
    distances_in : np.ndarray
        KNN distances for the inlier data points.
    distances_out : np.ndarray
        KNN distances for the outlier data points.
    plot : bool, optional
        Whether to plot the ROC curve, by default False
    title : str, optional
        Title for the ROC curve plot, by default "".
        Only used when plot is True.

    Returns
    -------
    float
        ROC AUC score.
    """
    # replace infinity with twice of the maximum value, hacky, may need more thoughts
    distances_in[distances_in == np.inf] = np.nan
    distances_out[distances_out == np.inf] = np.nan
    max_val = max(np.nanmax(distances_in), np.nanmax(distances_out))
    two_times_max = 2 * max_val
    distances_in = np.nan_to_num(distances_in, nan=two_times_max)
    distances_out = np.nan_to_num(distances_out, nan=two_times_max)

    y_true = [0] * len(distances_in) + [1] * len(distances_out)
    y_score = np.concatenate([distances_in, distances_out])
    roc_auc = roc_auc_score(
        y_true=y_true,
        y_score=y_score,
    )

    if plot:
        plot_roc_curve(
            y_true=y_true,
            y_score=y_score,
            roc_auc=roc_auc,
            title=title,
        )

    return roc_auc


def compute_auc(
    signature_mahalanobis_knn: SignatureMahalanobisKNN,
    test_in: np.ndarray,
    test_out: np.ndarray,
    is_signature: bool,
    plot: bool = False,
    title: str = "",
) -> float:
    """
    Compute ROC AUC given the data points of inliers and outliers.

    Parameters
    ----------
    test_in : np.ndarray
        Data points from the inlier class.
    test_out : np.ndarray
        Data points from the outlier class.
    is_signature : bool
        Whether the data provided are signatures or not.
        If True, then test_in and test_out are signatures.
        If False, then test_in and test_out are data points
        and the signatures will be computed.
    plot : bool, optional
        Whether to plot the ROC curve, by default False
    title : str, optional
        Title for the ROC curve plot, by default "".
        Only used when plot is True.

    Returns
    -------
    float
        ROC AUC score.
    """
    # compute KNN distances for the signatures of the data points
    # for both inliers and outliers of distribution data
    if is_signature:
        distances_in = signature_mahalanobis_knn.conformance(signatures=test_in)
        distances_out = signature_mahalanobis_knn.conformance(signatures=test_out)
    else:
        distances_in = signature_mahalanobis_knn.conformance(X=test_in)
        distances_out = signature_mahalanobis_knn.conformance(X=test_out)

    # convert to the default data type of the arrays in
    # the mahalanobis distance object
    distances_in = distances_in.astype(
        signature_mahalanobis_knn.mahal_distance.default_dtype
    )
    distances_out = distances_out.astype(
        signature_mahalanobis_knn.mahal_distance.default_dtype
    )

    # compute AUC for the inliers and outliers
    return compute_auc_given_dists(
        distances_in=distances_in,
        distances_out=distances_out,
        plot=plot,
        title=title,
    )


def plot_cdf_given_dists(
    distances_in: np.ndarray,
    distances_out: np.ndarray,
    bins: int | np.array = 50,
    xrange: tuple[float, float] | None = None,
    xticks: list[float] | None = None,
    xlog: bool = False,
    xlog_base: int = 10,
    title: str = "",
):
    """
    Plot an empirical cumulative distributions of the
    distances of inliers and outliers.

    Parameters
    ----------
    distances_in : np.ndarray
        KNN distances for the inlier data points.
    distances_out : np.ndarray
        KNN distances for the outlier data points.
    bins : int | np.array, optional
        Number of bins or the bins, by default 10.
        If int, then the bins will be equally spaced.
        If array, then the sequence defines the bin edges.
    xrange : tuple[float, float] | None, optional
        Range of the x-axis, by default None.
        If None, then the range will be the minimum and maximum
        of the distances.
    xticks : list[float] | None, optional
        Tick values of the x-axis, by default None.
        If None, then the ticks will be automatically generated.
    xlog : bool, optional
        Whether to use log scale for the x-axis, by default False.
    xlog_base : int, optional
        Base of the log scale, by default 10.
        Only used when xlog is True.
    title : str, optional
        Title for the ROC curve plot, by default "".
        Only used when plot is True.
    """
    # obtain the empirical cumulative distribution functions
    sorted_inlier = np.sort(distances_in)
    cumulative_inliers = np.arange(1, len(sorted_inlier) + 1) / len(sorted_inlier)
    sorted_outlier = np.sort(distances_out)
    cumulative_outliers = np.arange(1, len(sorted_outlier) + 1) / len(sorted_outlier)

    # define the empirical cumulative distribution functions
    def empirical_cdf_inliers(x):
        return np.interp(x, sorted_inlier, cumulative_inliers, left=0.0, right=1.0)

    def empirical_cdf_outliers(x):
        return np.interp(x, sorted_outlier, cumulative_outliers, left=0.0, right=1.0)

    # obtain the range of x to evaluate the ECDFs
    xmin = min(np.min(distances_in), np.min(distances_out))
    xmax = max(np.max(distances_in), np.max(distances_out))
    x = np.linspace(xmin, xmax, bins)

    # plot the cumulative functions
    plt.plot(
        x,
        empirical_cdf_inliers(x),
        c="blue",
        label="inliers",
        linestyle="--",
        linewidth=2,
    )
    plt.plot(x, empirical_cdf_outliers(x), c="orange", label="anomalies", linewidth=2)
    plt.grid()
    if title != "":
        plt.title(title)
    plt.legend(loc="lower right")
    if xrange is not None:
        plt.xlim(xrange)
    if xticks is not None:
        plt.xticks(xticks)
    if xlog:
        plt.xscale("log", base=xlog_base)
    plt.ylim([0, 1])
    plt.ylabel("Cumulative probability")
    plt.xlabel("Conformance")
    plt.show()
