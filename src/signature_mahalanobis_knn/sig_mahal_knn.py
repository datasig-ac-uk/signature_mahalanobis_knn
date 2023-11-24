from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pynndescent import NNDescent
from sklearn.neighbors import NearestNeighbors

from signature_mahalanobis_knn.mahal_distance import Mahalanobis

X_OR_SIGNATURE_ERROR_MSG = "Either X or signatures must be provided"
MODEL_NOT_FITTED_ERROR_MSG = "Must fit the model first"


class SignatureMahalanobisKNN:
    def __init__(
        self,
        n_jobs: int = 1,
        random_state: int | None = None,
    ):
        """
        Parameters
        ----------
        n_jobs : int, optional
            Parameter for joblib, number of parallel processors to use, by default 1.
            -1 means using all processors, -2 means using all processors but one.
        random_state : int | None, optional
            Random state for the knn library, by default None.
        """
        self.signature_transform: object | None = None
        self.n_jobs: int = n_jobs
        self.mahal_distance: Mahalanobis | None = None
        self.signatures_train: np.array | None = None
        self.knn: NearestNeighbors | NNDescent | None = None
        self.random_state: int | None = random_state

    def fit(
        self,
        knn_library: str = "sklearn",
        X_train: np.ndarray | None = None,
        signatures_train: np.ndarray | None = None,
        knn_algorithm: str = "auto",
        signature_kwargs: dict | None = None,
        **kwargs,
    ) -> None:
        """
        Fit the KNN model with the corpus of signatures_train.
        If signatures_train is not provided, then X_train must be provided
        to compute the signatures_train.
        If signatures_train is provided, then X_train is ignored.

        Parameters
        ----------
        X_train : np.ndarray | None, optional
            Data points, by default None.
            Must support index operation X_train[i] where
            each X_train[i] returns a data point in the corpus.
            Typically a three-dimensional array of shape
            (batch, length, channels).
        signatures_train : np.ndarray | None, optional
            Signatures of the data points, by default None.
            Two dimensional array of shape (n_samples, sig_dim).
        knn_algorithm : str, optional
            Algorithm used to compute the nearest neighbors
            (see scikit-learn documentation for `sklearn.neighbors.NearestNeighbors`),
            by default "auto".
        signature_kwargs : dict | None, optional
            Keyword arguments passed to the signature transformer if
            signatures_train are not provided and X_train is provided.
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
                - rescaling: str, "pre" or "post",
                    - "pre": rescale the path last signature term should
                      be roughly O(1)
                    - "post": Rescals the output signature by multiplying
                      the depth-d term by d!. Aim is that every term become ~O(1).
                - sig_tfm: str, One of: ['signature', 'logsignature']).
                - depth: int, Signature truncation depth.
                - backend: str, one of: `'esig'` (default), or `'iisignature'`.
                  The backend to use for signature computation.
            By default, the following arguments are used:
                - augmentation_list: ("addtime",)
                - window_name: "global"
                - window_depth: None
                - window_length: None
                - window_step: None
                - rescaling: None
                - sig_tfm: "signature"
                - depth: 2
        **kwargs
            Keyword arguments passed to the knn library.
            See scikit-learn documentation for `sklearn.neighbors.NearestNeighbors`
            and pynndescent documentation for `pynndescent.NNDescent`.
        """
        if signatures_train is None:
            if X_train is None:
                raise ValueError(X_OR_SIGNATURE_ERROR_MSG)

            from sktime.transformations.panel.signature_based import (
                SignatureTransformer,
            )

            # set default kwargs for signature transformer if not provided
            sig_defaults = {
                "augmentation_list": ("addtime",),
                "window_name": "global",
                "window_depth": None,
                "window_length": None,
                "window_step": None,
                "rescaling": None,
                "sig_tfm": "signature",
                "depth": 2,
                "backend": "esig",
            }

            if signature_kwargs is None:
                # set all defaults
                signature_kwargs = sig_defaults
            else:
                # set defaults for any missing kwargs
                for key, value in sig_defaults.items():
                    if key not in signature_kwargs:
                        signature_kwargs[key] = value

            self.signature_transform = SignatureTransformer(
                **signature_kwargs,
            )

            # compute signatures
            sigs = Parallel(n_jobs=self.n_jobs)(
                delayed(self.signature_transform.fit_transform)(X_train[i])
                for i in range(len(X_train))
            )
            self.signatures_train = np.array(pd.concat(sigs))
        else:
            self.signatures_train = signatures_train

        # fit mahalanobis distance for the signatures_train
        self.mahal_distance = Mahalanobis()
        self.mahal_distance.fit(self.signatures_train)

        if knn_library == "sklearn":
            # fit knn for the mahalanobis distance
            knn = NearestNeighbors(
                metric="euclidean",
                n_jobs=self.n_jobs,
                algorithm=knn_algorithm,
                **kwargs,
            )
            knn.fit(self.mahal_distance.U)
            self.knn = knn
        elif knn_library == "pynndescent":
            # fit pynndescent for the mahalanobis distance
            knn = NNDescent(
                data=self.mahal_distance.U,
                metric="euclidean",
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                **kwargs,
            )
            self.knn = knn

    def conformance(
        self,
        X_test: np.ndarray | None = None,
        signatures_test: np.ndarray | None = None,
        n_neighbors: int = 20,
        debug=False,
    ) -> np.ndarray:
        """
        Compute the conformance scores for the data points either passed in
        directly as X_test or the signatures_test of the data points in signatures_test.
        If signatures_test is not provided, then X_test must be provided
        to compute the signatures_test.
        If signatures_test is provided, then X_test is ignored.

        Must call fit() method first.

        Parameters
        ----------
        X_test : np.ndarray | None, optional
            Data points, by default None.
            Must support index operation X_test[i] where
            each X_test[i] returns a data point in the corpus.
            Typically a three-dimensional array of shape
            (batch, length, channels).
        signatures_test : np.ndarray | None, optional
            Signatures of the data points, by default None.
            Two dimensional array of shape (n_samples, sig_dim).

        Returns
        -------
        np.ndarray
            Conformance scores for data points provided.
        """
        if self.knn is None:
            raise ValueError(MODEL_NOT_FITTED_ERROR_MSG)

        if signatures_test is None:
            if X_test is None:
                raise ValueError(X_OR_SIGNATURE_ERROR_MSG)
            if self.signature_transform is None:
                raise ValueError(MODEL_NOT_FITTED_ERROR_MSG)

            # compute signatures
            sigs = Parallel(n_jobs=self.n_jobs)(
                delayed(self.signature_transform.fit_transform)(X_test[i])
                for i in range(len(X_test))
            )
            signatures_test = np.array(pd.concat(sigs))

        # pre-process the signatures
        sig_dim = signatures_test.shape[1]
        modified_signatures = (
            (signatures_test - self.mahal_distance.mu)
            @ self.mahal_distance.Vt.T
            @ np.diag(self.mahal_distance.S ** (-1))
        )

        # compute Euclidean NNs
        if isinstance(self.knn, NearestNeighbors):
            # compute KNN distances for the modified_signatures of the data points
            # against the modified_signatures of the corpus
            candidate_distances, train_indices = self.knn.kneighbors(
                modified_signatures, n_neighbors=n_neighbors, return_distance=True
            )
        elif isinstance(self.knn, NNDescent):
            # compute KNN distances for the modified_signatures of the data points
            # against the modified_signatures of the corpus
            train_indices, candidate_distances = self.knn.query(
                modified_signatures, k=n_neighbors
            )

        # post-process the candidate distances
        test_indices = np.tile(
            np.arange(train_indices.shape[0]), (train_indices.shape[1], 1)
        ).T
        # differences has shape (n_test x n_neighbors x sig_dim)
        differences = (
            self.signatures_train[train_indices] - signatures_test[test_indices]
        )

        denominator = np.linalg.norm(differences, axis=-1)
        numerator = np.linalg.norm(
            differences
            @ (
                np.identity(sig_dim) - self.mahal_distance.Vt.T @ self.mahal_distance.Vt
            ),
            axis=-1,
        )

        rho = numerator / denominator
        # get rid of nans from zero denominator
        rho[denominator == 0] = 0

        candidate_distances[denominator < self.mahal_distance.zero_thres] = 0
        candidate_distances[rho > self.mahal_distance.subspace_thres] = np.inf

        # compute the minimum of the candidate distances for each data point
        if debug:
            return np.min(candidate_distances, axis=-1), train_indices

        return np.min(candidate_distances, axis=-1)
