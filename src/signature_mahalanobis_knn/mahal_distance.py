from __future__ import annotations

import numpy as np
from numba import jit


class Mahalanobis:
    def __init__(self):
        """
        After fit is called, becomes callable and intended to be used
        as a distance function in sklearn nearest neighbour.
        """
        self.Vt: np.ndarray = np.empty(
            0
        )  # Truncated right singular matrix transposed of the corpus
        self.mu: np.ndarray = np.empty(0)  # Mean of the corpus
        self.S: np.ndarray = np.empty(0)  # Truncated singular values of the corpus
        self.subspace_thres: float = (
            1e-3  # Threshold to decide whether a point is in the data subspace
        )
        self.svd_thres: float = (
            1e-12  # Threshold to decide numerical rank of the data matrix
        )
        self.numerical_rank: int = -1  # Numerical rank
        self.default_dtype = np.float64

    def fit(self, X: np.ndarray, y: None = None, **kwargs) -> None:  # noqa: ARG002
        """
        Fit the object to a corpus X.

        Parameters
        ----------
        X : np.ndarray
            Panel data representing the corpus, each row is a data point.
        y: None
            Not used, present for API consistency by convention.
        """
        # mean centering
        self.mu = np.mean(X, axis=0)
        X = X - self.mu

        U, S, Vt = np.linalg.svd(X)
        k = np.sum(self.svd_thres <= S)  # detected numerical rank
        self.numerical_rank = k
        self.Vt = Vt[:k].astype(self.default_dtype)
        self.S = S[:k].astype(self.default_dtype)

    @staticmethod
    @jit(nopython=True)  # Observe 6 times speed up on pen-digit dataset
    def calc_distance(
        x1: np.ndarray,
        x2: np.ndarray,
        Vt: np.ndarray,
        S: np.ndarray,
        subspace_thres: float,
    ) -> float:
        """
        Compute the variance norm between x1 and x2 using the precomputed SVD.

        Parameters
        ----------
        x1 : np.ndarray
            One-dimensional array.
        x2 : np.ndarray
            One-dimensional array.
        Vt : np.ndarray
            Two-dimensional arrat, truncated right singular matrix transposed of the corpus.
        S : np.ndarray
            One-dimensional array, truncated singular values of the corpus.
        subspace_thres : float
            Threshold to decide whether a point is in the data subspace.

        Returns
        -------
        float
            Value representing distance between x, y.
        """
        x = x1 - x2
        norm_x = np.linalg.norm(x)
        if norm_x < 1e-15:
            return 0.0

        # quantifies the amount that x is outside the row-subspace
        rho = np.linalg.norm(x - x @ Vt.T @ Vt) / norm_x
        if rho > subspace_thres:
            return np.inf

        return x @ Vt.T @ np.diag(S ** (-2)) @ Vt @ x.T

    def distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Compute the variance norm between x1 and x2 using the precomputed SVD.

        Parameters
        ----------
        x1 : np.ndarray
            One-dimensional array.
        x2 : np.ndarray
            One-dimensional array.

        Returns
        -------
        float
            Value representing distance between x, y.
        """
        # ensure inputs are the right data type
        x1 = x1.astype(self.default_dtype)
        x2 = x2.astype(self.default_dtype)

        return self.calc_distance(x1, x2, self.Vt, self.S, self.subspace_thres)
