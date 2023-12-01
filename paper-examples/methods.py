from __future__ import annotations

import time

import Data
import numpy as np

from signature_mahalanobis_knn import SignatureMahalanobisKNN
from signature_mahalanobis_knn.baselines.isolation_forest import (
    isolation_forest_based_on_moments,
    isolation_forest_scores,
)
from signature_mahalanobis_knn.baselines.local_outlier_factor import (
    local_outlier_factor_based_on_moments,
    local_outlier_factor_scores,
)
from signature_mahalanobis_knn.baselines.utils import compute_signatures


def compute_signature_mahalanbois_knn_scores(
    data: Data.Data,
    knn_library: str,
    depth: int,
    n_neighbours: int,
    random_state: int,
    backend: str = "esig",
) -> dict[str, np.array | float]:
    signature_maha_knn = SignatureMahalanobisKNN(n_jobs=-1, random_state=random_state)

    # measure the time spent on fit
    start_time = time.time()
    signature_maha_knn.fit(
        knn_library=knn_library,
        X_train=data.corpus,
        signature_kwargs={
            "augmentation_list": None,
            "depth": depth,
            "backend": backend,
        },
    )
    fit_time = time.time() - start_time
    print(f"fit_time: {fit_time}")  # noqa: T201

    # measure the time spent on conformance
    start_time = time.time()
    scores_inliers = signature_maha_knn.conformance(
        data.test_inlier, n_neighbors=n_neighbours
    )
    scores_outliers = signature_maha_knn.conformance(
        data.test_outlier, n_neighbors=n_neighbours
    )
    compute_score_time = time.time() - start_time
    print(f"compute_score_time: {compute_score_time}")  # noqa: T201

    return {
        "scores_inliers": scores_inliers,
        "scores_outliers": scores_outliers,
        "fit_time": fit_time,
        "compute_score_time": compute_score_time,
    }


def compute_isolation_forest_scores_based_on_moments(
    data: Data.Data,
    random_state: int,
    **kwargs,
) -> dict[str, np.array | float]:
    return isolation_forest_based_on_moments(
        corpus_streams=data.corpus,
        inlier_streams=data.test_inlier,
        outlier_streams=data.test_outlier,
        random_state=random_state,
        verbose=True,
        **kwargs,
    )


def compute_isolation_forest_scores_signatures(
    data: Data.Data,
    depth: int,
    random_state: int,
    backend: str = "esig",
    **kwargs,
) -> dict[str, np.array | float]:
    # compute signatures for corpus, inlier and outlier
    corpus_signatures = compute_signatures(
        X=data.corpus,
        n_jobs=-1,
        signature_kwargs={
            "augmentation_list": None,
            "depth": depth,
            "backend": backend,
        },
    )
    inlier_signatures = compute_signatures(
        X=data.test_inlier,
        n_jobs=-1,
        signature_kwargs={
            "augmentation_list": None,
            "depth": depth,
            "backend": backend,
        },
    )
    outlier_signatures = compute_signatures(
        X=data.test_outlier,
        n_jobs=-1,
        signature_kwargs={
            "augmentation_list": None,
            "depth": depth,
            "backend": backend,
        },
    )

    return isolation_forest_scores(
        corpus_features=corpus_signatures,
        inlier_features=inlier_signatures,
        outlier_features=outlier_signatures,
        random_state=random_state,
        verbose=True,
        **kwargs,
    )


def compute_local_outlier_factor_scores_based_on_moments(
    data: Data.Data,
    **kwargs,
) -> dict[str, np.array | float]:
    return local_outlier_factor_based_on_moments(
        corpus_streams=data.corpus,
        inlier_streams=data.test_inlier,
        outlier_streams=data.test_outlier,
        verbose=True,
        **kwargs,
    )


def compute_local_outlier_factor_scores_signatures(
    data: Data.Data,
    depth: int,
    backend: str = "esig",
    **kwargs,
) -> dict[str, np.array | float]:
    # compute signatures for corpus, inlier and outlier
    corpus_signatures = compute_signatures(
        X=data.corpus,
        n_jobs=-1,
        signature_kwargs={
            "augmentation_list": None,
            "depth": depth,
            "backend": backend,
        },
    )
    inlier_signatures = compute_signatures(
        X=data.test_inlier,
        n_jobs=-1,
        signature_kwargs={
            "augmentation_list": None,
            "depth": depth,
            "backend": backend,
        },
    )
    outlier_signatures = compute_signatures(
        X=data.test_outlier,
        n_jobs=-1,
        signature_kwargs={
            "augmentation_list": None,
            "depth": depth,
            "backend": backend,
        },
    )

    return local_outlier_factor_scores(
        corpus_features=corpus_signatures,
        inlier_features=inlier_signatures,
        outlier_features=outlier_signatures,
        verbose=True,
        **kwargs,
    )
