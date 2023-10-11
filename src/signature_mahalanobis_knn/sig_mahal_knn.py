from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

from signature_mahalanobis_knn.mahal_distance import Mahalanobis
from signature_mahalanobis_knn.utils import plot_roc_curve


class SignatureMahalanobisKNN:
    def __init__(
        self,
        n_jobs: int = -2,
    ):
        """
        Parameters
        ----------
        n_jobs : int, optional
            Parameter for joblib, number of parallel processors to use, by default -2.
            -1 means using all processors, -2 means using all processors but one.
        """
        self.signature_transform = None
        self.n_jobs = n_jobs
        self.signatures = None
        self.knn = None

    def fit(
        self,
        X: np.ndarray | None = None,
        signatures: np.ndarray | None = None,
        knn_algorithm: str = "auto",
        **kwargs,
    ) -> None:
        """
        Fit the KNN model with the corpus of signatures.
        If signatures is not provided, then X must be provided
        to compute the signatures.
        If signatures is provided, then X is ignored.

        Parameters
        ----------
        X : np.ndarray | None, optional
            Data points, by default None.
            Must support index operation X[i] where
            each X[i] returns a data point in the corpus.
        signatures : np.ndarray | None, optional
            Signatures of the data points, by default None.
            Must support index operation X[i] where
            each X[i] returns a data point in the corpus.
        knn_algorithm : str, optional
            Algorithm used to compute the nearest neighbors
            (see scikit-learn documentation for `sklearn.neighbors.NearestNeighbors`),
            by default "auto".
        **kwargs
            Keyword arguments passed to the signature transformer if
            signatures are not provided and X is provided.
            See sktime documentation for
            `sktime.transformations.panel.signature_based.SignatureTransformer`.
        """
        if signatures is None:
            if X is None:
                msg = "Either X or signatures must be provided"
                raise ValueError(msg)

            from sktime.transformations.panel.signature_based import (
                SignatureTransformer,
            )

            # set default kwargs for signature transformer if not provided
            if kwargs == {}:
                kwargs = {
                    "augmentation_list": ("addtime",),
                    "window_name": "global",
                    "window_depth": None,
                    "window_length": None,
                    "window_step": None,
                    "rescaling": None,
                    "sig_tfm": "signature",
                    "depth": 2,
                }

            self.signature_transform = SignatureTransformer(
                **kwargs,
            )

            # compute signatures
            sigs = Parallel(n_jobs=self.n_jobs)(
                delayed(self.signature_transform.fit_transform)(X[i])
                for i in range(len(X))
            )
            self.signatures = pd.concat(sigs)
        else:
            self.signatures = signatures

        # fit mahalanobis distance for the signatures
        mahal_distance = Mahalanobis()
        mahal_distance.fit(self.signatures)

        # fit knn for the mahalanobis distance
        knn = NearestNeighbors(
            metric=mahal_distance.distance,
            n_jobs=self.n_jobs,
            algorithm=knn_algorithm,
        )
        knn.fit(self.signatures)
        self.knn = knn

    def conformance(
        self,
        X: np.ndarray | None = None,
        signatures: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Compute the conformance scores for the data points either passed in
        directly as X or the signatures of the data points in signatures.
        If signatures is not provided, then X must be provided
        to compute the signatures.
        If signatures is provided, then X is ignored.

        Must call fit() method first.

        Parameters
        ----------
        X : np.ndarray | None, optional
            Data points, by default None.
            Must support index operation X[i] where
            each X[i] returns a data point in the corpus.
        signatures : np.ndarray | None, optional
            Signatures of the data points, by default None.
            Must support index operation X[i] where
            each X[i] returns a data point in the corpus.

        Returns
        -------
        np.ndarray
            Conformance scores for data points provided.
        """
        if self.knn is None:
            msg = "Must fit the model first"
            raise ValueError(msg)

        if signatures is None:
            if X is None:
                msg = "Either X or signatures must be provided"
                raise ValueError(msg)
            if self.signature_transform is None:
                msg = "Must fit the model first"
                raise ValueError(msg)

            # compute signatures
            sigs = Parallel(n_jobs=self.n_jobs)(
                delayed(self.signature_transform.fit_transform)(X[i])
                for i in range(len(X))
            )
            signatures = pd.concat(sigs)

        # compute KNN distances for the signatures of the data points
        # against the signatures of the corpus
        distances, _ = self.knn.kneighbors(
            signatures, n_neighbors=1, return_distance=True
        )

        return distances

    def compute_auc_given_dists(
        self,
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
        distances_in = np.nan_to_num(distances_in, max_val * 2)
        distances_out = np.nan_to_num(distances_out, max_val * 2)

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
        self,
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
            distances_in = self.conformance(signatures=test_in)
            distances_out = self.conformance(signatures=test_out)
        else:
            distances_in = self.conformance(X=test_in)
            distances_out = self.conformance(X=test_out)

        # compute AUC for the inliers and outliers
        return self.compute_auc_given_dists(
            distances_in=distances_in,
            distances_out=distances_out,
            plot=plot,
            title=title,
        )
