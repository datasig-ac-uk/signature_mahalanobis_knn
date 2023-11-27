from __future__ import annotations

import numpy as np
from sklearn.neighbors import LocalOutlierFactor


def local_outlier_factor_scores(
    corpus_features: np.array,
    inlier_features: np.array,
    outlier_features: np.array,
    **kwargs,
) -> tuple[np.array, np.array]:
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
    **kwargs
        Keyword arguments passed to the LocalOutlierFactor model

    Returns
    -------
    tuple[np.array, np.array]
        Local Outlier Factor scores for inliers and outliers
    """
    # initalise IsolationForest model
    detector = LocalOutlierFactor(
        **kwargs,
    ).fit(corpus_features)

    # obtain scores for inliers and outliers
    # negate the scores so that larger values correspond to outliers
    # to be consistent with our method
    # LocalOutlierFactor returns scores where the lower, the more abnormal
    scores_inliers = -1 * detector.score_samples(inlier_features)
    scores_outliers = -1 * detector.score_samples(outlier_features)

    return scores_inliers, scores_outliers


def local_outlier_factor_based_on_moments(
    corpus_streams: np.array,
    inlier_streams: np.array,
    outlier_streams: np.array,
    **kwargs,
) -> tuple[np.array, np.array]:
    """
    Compute the Local Outlier Factor scores for inliers and outliers
    given the streams for the corpus. The features are computed by
    computing the mean and covariances of the streams.

    Parameters
    ----------
    corpus_streams : np.array
        Three-dimensional array of shape (batch, length, channels)
    inlier_streams : np.array
        Three-dimensional array of shape (batch, length, channels)
    outlier_streams : np.array
        Three-dimensional array of shape (batch, length, channels)
    **kwargs
        Keyword arguments passed to the LocalOutlierFactor model

    Returns
    -------
    tuple[np.array, np.array]
        Local Outlier Factor scores for inliers and outliers
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
        **kwargs,
    )
