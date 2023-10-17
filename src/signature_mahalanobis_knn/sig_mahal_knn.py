from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.neighbors import NearestNeighbors

from signature_mahalanobis_knn.mahal_distance import Mahalanobis

X_OR_SIGNATURE_ERROR_MSG = "Either X or signatures must be provided"
MODEL_NOT_FITTED_ERROR_MSG = "Must fit the model first"


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
        self.mahal_distance = None
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
            `sktime.transformations.panel.signature_based.SignatureTransformer`
            for more details on what arguments are available.
            Some notable options are:
                - augmentation_list: tuple[str], Possible augmentation strings are
                  ['leadlag', 'ir', 'addtime', 'cumsum', 'basepoint']
                - window_name: str, String from
                  ['global', 'sliding', 'expanding', 'dyadic']
                - window_depth: int, The depth of the dyadic window.
                  (Active only if `window_name == 'dyadic']`.
                - window_length: int, The length of the sliding/expanding window.
                  (Active only if `window_name in ['sliding, 'expanding'].
                - window_step: int, The step of the sliding/expanding window.
                  (Active only if `window_name in ['sliding, 'expanding'].
                - rescaling: "pre" or "post",
                    - "pre": rescale the path last signature term should
                      be roughly O(1)
                    - "post": Rescals the output signature by multiplying
                      the depth-d term by d!. Aim is that every term become ~O(1).
                - sig_tfm: One of: ['signature', 'logsignature']).
                - depth: int, Signature truncation depth.
            By default, the following arguments are used:
                - augmentation_list: ("addtime",)
                - window_name: "global"
                - window_depth: None
                - window_length: None
                - window_step: None
                - rescaling: None
                - sig_tfm: "signature"
                - depth: 2
        """
        if signatures is None:
            if X is None:
                raise ValueError(X_OR_SIGNATURE_ERROR_MSG)

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
        self.mahal_distance = Mahalanobis()
        self.mahal_distance.fit(self.signatures)

        # fit knn for the mahalanobis distance
        knn = NearestNeighbors(
            metric=self.mahal_distance.distance,
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
            raise ValueError(MODEL_NOT_FITTED_ERROR_MSG)

        if signatures is None:
            if X is None:
                raise ValueError(X_OR_SIGNATURE_ERROR_MSG)
            if self.signature_transform is None:
                raise ValueError(MODEL_NOT_FITTED_ERROR_MSG)

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
