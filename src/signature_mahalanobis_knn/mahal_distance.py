from __future__ import annotations

import numpy as np
from numba import njit


class Mahalanobis:
    def __init__(
        self,
        subspace_thres: float = 1e-3,
        svd_thres: float = 1e-12,
        zero_thres: float = 1e-15,
        default_dtype: np.dtype = np.float64,
    ):
        """
        After fit is called, becomes callable and intended to be used
        as a distance function in sklearn nearest neighbour.

        Parameters
        ----------
        subspace_thres : float, optional
            Threshold to decide whether a point is in the data subspace,
            by default 1e-3.
        svd_thres : float, optional
            Threshold to decide numerical rank of the data matrix,
            by default 1e-12.
        zero_thres : float, optional
            Threshold to decide whether the distance is zero,
            by default 1e-15.
        """
        self.subspace_thres: float = subspace_thres
        self.svd_thres: float = svd_thres
        self.zero_thres: float = zero_thres

        # set the following after fit() is called - None means not fitted yet
        # nean of the corpus
        self.mu: np.ndarray | None = None
        # truncated right singular matrix transposed of the corpus
        self.Vt: np.ndarray | None = None
        # truncated singular values of the corpus
        self.S: np.ndarray | None = None
        # numerical rank of the corpus
        self.numerical_rank: int | None = None
        self.default_dtype = default_dtype

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
        self.U = U[:, :k].astype(self.default_dtype)
        self.Vt = Vt[:k].astype(self.default_dtype)
        self.S = S[:k].astype(self.default_dtype)

    @staticmethod
    @njit(fastmath=True)
    def calc_distance(
        x1: np.ndarray,
        x2: np.ndarray,
        Vt: np.ndarray,
        S: np.ndarray,
        subspace_thres: float,
        zero_thres: float,
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
            Two-dimensional array, truncated right singular matrix transposed of the corpus.
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
        if norm_x < zero_thres:
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
        if self.numerical_rank is None:
            msg = "Mahalanobis distance is not fitted yet."
            raise ValueError(msg)

        # ensure inputs are the right data type
        x1 = x1.astype(self.default_dtype)
        x2 = x2.astype(self.default_dtype)

        return self.calc_distance(
            x1=x1,
            x2=x2,
            Vt=self.Vt,
            S=self.S,
            subspace_thres=self.subspace_thres,
            zero_thres=self.zero_thres,
        )
