from __future__ import annotations

import time

import numpy as np
from sklearn.neighbors import LocalOutlierFactor


def local_outlier_factor_scores(
    corpus_features: np.array,
    inlier_features: np.array,
    outlier_features: np.array,
    verbose: bool = False,
    **kwargs,
) -> dict[str, np.array | float]:
    """
    Compute the Local Outlier Factor scores for inliers and outliers
    given the features for the corpus.

    Parameters
    ----------
    corpus_features : np.array
        Two-dimensional array of shape (n_samples, n_features)
    inlier_features : np.array
        Two-dimensional array of shape (n_samples, n_features)
    outlier_features : np.array
        Two-dimensional array of shape (n_samples, n_features)
    verbose : bool, optional
        Whether to print computation times on fit and scores computation,
        by default False
    **kwargs
        Keyword arguments passed to the LocalOutlierFactor model

    Returns
    -------
    dict[str, np.array | float]
        Dictionary with keys:
            - "scores_inliers": np.array of shape (n_samples,)
            - "scores_outliers": np.array of shape (n_samples,)
            - "fit_time": float
            - "compute_score_time": float
    """
    # initalise LocalOutlierFactor model
    # and fit on all the data
    # measure the time spent on fit
    start_time = time.time()
    detector = LocalOutlierFactor(
        **kwargs,
    ).fit(np.concatenate([corpus_features, inlier_features, outlier_features], axis=0))
    fit_time = time.time() - start_time
    if verbose:
        print(f"fit_time: {fit_time}")  # noqa: T201

    # obtain scores for inliers and outliers
    # negate the scores so that larger values correspond to outliers
    # to be consistent with our method
    # LocalOutlierFactor returns scores where the lower, the more abnormal
    # measure the time spent on conformance
    start_time = time.time()
    X_scores = detector.negative_outlier_factor_[len(corpus_features) :]
    # slice the scores to obtain the scores for inliers and outliers
    scores_inliers = -1 * X_scores[: len(inlier_features)]
    scores_outliers = -1 * X_scores[len(inlier_features) :]
    compute_score_time = time.time() - start_time
    if verbose:
        print(f"compute_score_time: {compute_score_time}")  # noqa: T201

    return {
        "scores_inliers": scores_inliers,
        "scores_outliers": scores_outliers,
        "fit_time": fit_time,
        "compute_score_time": compute_score_time,
    }


def local_outlier_factor_based_on_moments(
    corpus_streams: np.array | list[np.array],
    inlier_streams: np.array | list[np.array],
    outlier_streams: np.array | list[np.array],
    verbose: bool = False,
    **kwargs,
) -> dict[str, np.array | float]:
    """
    Compute the Local Outlier Factor scores for inliers and outliers
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
        Keyword arguments passed to the LocalOutlierFactor model

    Returns
    -------
    dict[str, np.array | float]
        Dictionary with keys:
            - "scores_inliers": np.array of shape (n_samples,)
            - "scores_outliers": np.array of shape (n_samples,)
            - "fit_time": float
            - "compute_score_time": float
    """

    # obtain features by computing the mean and covariances
    def compute_moment_features(streams):
        features = []
        for stream in streams:
            mean = np.mean(stream, axis=0)
            cov = np.cov(stream, rowvar=False)[np.triu_indices(len(mean))]
            features.append(np.append(mean, np.ravel(cov)))

        return np.array(features)

    # compute features for corpus, inliers and outliers
    corpus_features, inlier_features, outlier_features = map(
        compute_moment_features,
        (corpus_streams, inlier_streams, outlier_streams),
    )

    return local_outlier_factor_scores(
        corpus_features=corpus_features,
        inlier_features=inlier_features,
        outlier_features=outlier_features,
        verbose=verbose,
        **kwargs,
    )
