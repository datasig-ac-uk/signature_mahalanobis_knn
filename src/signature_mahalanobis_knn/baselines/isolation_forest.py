from __future__ import annotations

import time

import numpy as np
from sklearn.ensemble import IsolationForest

from signature_mahalanobis_knn.baselines.utils import compute_moment_features


def isolation_forest_scores(
    corpus_features: np.array,
    inlier_features: np.array,
    outlier_features: np.array,
    random_state: int = 0,
    verbose: bool = False,
    **kwargs,
) -> dict[str, np.array | float]:
    """
    Compute the Isolation Forest scores for inliers and outliers
    given the features for the corpus.

    Parameters
    ----------
    corpus_features : np.array
        Two-dimensional array of shape (n_samples, n_features)
    inlier_features : np.array
        Two-dimensional array of shape (n_samples, n_features)
    outlier_features : np.array
        Two-dimensional array of shape (n_samples, n_features)
    random_state : int, optional
        Random seed, by default 0
    verbose : bool, optional
        Whether to print computation times on fit and scores computation,
        by default False
    **kwargs
        Keyword arguments passed to the IsolationForest model

    Returns
    -------
    dict[str, np.array | float]
        Dictionary with keys:
            - "scores_inliers": np.array of shape (n_samples,)
            - "scores_outliers": np.array of shape (n_samples,)
            - "fit_time": float
            - "compute_score_time": float
    """
    # initalise IsolationForest model
    # measure the time spent on fit
    start_time = time.time()
    detector = IsolationForest(
        random_state=random_state,
        **kwargs,
    ).fit(corpus_features)
    fit_time = time.time() - start_time
    if verbose:
        print(f"fit_time: {fit_time}")  # noqa: T201

    # obtain scores for inliers and outliers
    # negate the scores so that larger values correspond to outliers
    # to be consistent with our method
    # IsolationForest returns scores where the lower, the more abnormal
    # measure the time spent on conformance
    start_time = time.time()
    scores_inliers = -1 * detector.score_samples(inlier_features)
    scores_outliers = -1 * detector.score_samples(outlier_features)
    compute_score_time = time.time() - start_time
    if verbose:
        print(f"compute_score_time: {compute_score_time}")  # noqa: T201

    return {
        "scores_inliers": scores_inliers,
        "scores_outliers": scores_outliers,
        "fit_time": fit_time,
        "compute_score_time": compute_score_time,
    }


def isolation_forest_based_on_moments(
    corpus_streams: np.array | list[np.array],
    inlier_streams: np.array | list[np.array],
    outlier_streams: np.array | list[np.array],
    random_state: int = 0,
    verbose: bool = False,
    **kwargs,
) -> dict[str, np.array | float]:
    """
    Compute the Isolation Forest scores for inliers and outliers
    given the streams for the corpus. The features are computed by
    computing the mean and covariances of the streams.

    Parameters
    ----------
    corpus_streams : np.array | list[np.array]
        Three-dimensional array of shape (batch, length, channels),
        or a list of two-dimensional arrays of shape (length, channels).
    inlier_streams : np.array | list[np.array]
        Three-dimensional array of shape (batch, length, channels),
        or a list of two-dimensional arrays of shape (length, channels).
    outlier_streams : np.array | list[np.array]
        Three-dimensional array of shape (batch, length, channels),
        or a list of two-dimensional arrays of shape (length, channels).
    verbose : bool, optional
        Whether to print computation times on fit and scores computation,
        by default False
    **kwargs
        Keyword arguments passed to the IsolationForest model

    Returns
    -------
    dict[str, np.array | float]
        Dictionary with keys:
            - "scores_inliers": np.array of shape (n_samples,)
            - "scores_outliers": np.array of shape (n_samples,)
            - "fit_time": float
            - "compute_score_time": float
    """
    # compute features for corpus, inliers and outliers
    corpus_features, inlier_features, outlier_features = map(
        compute_moment_features,
        (corpus_streams, inlier_streams, outlier_streams),
    )

    return isolation_forest_scores(
        corpus_features=corpus_features,
        inlier_features=inlier_features,
        outlier_features=outlier_features,
        random_state=random_state,
        verbose=verbose,
        **kwargs,
    )
