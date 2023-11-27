from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest


def isolation_forest_scores(
    corpus_features: np.array,
    inlier_features: np.array,
    outlier_features: np.array,
    random_state: int = 0,
    **kwargs,
):
    # initalise IsolationForest model
    detector = IsolationForest(
        random_state=random_state,
        **kwargs,
    ).fit(corpus_features)

    # obtain scores for inliers and outliers
    # negate the scores so that larger values correspond to outliers
    # to be consistent with our method
    # IsolationForest returns scores where the lower, the more abnormal
    scores_inliers = -1 * detector.score_samples(inlier_features)
    scores_outliers = -1 * detector.score_samples(outlier_features)

    return scores_inliers, scores_outliers


def isolation_forest_based_on_moments(
    corpus_streams: np.array,
    inlier_streams: np.array,
    outlier_streams: np.array,
    random_state: int = 0,
    **kwargs,
) -> tuple[np.array, np.array]:
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

    return isolation_forest_scores(
        corpus_features=corpus_features,
        inlier_features=inlier_features,
        outlier_features=outlier_features,
        random_state=random_state,
        **kwargs,
    )
