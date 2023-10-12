from __future__ import annotations

import numpy as np
from numba import jit


class Mahalanobis:
    def __init__(
        self,
        subspace_thres: float = 1e-3,
        svd_thres: float = 1e-12,
        zero_thres: float = 1e-15,
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
        # truncated right singular matrix transposed of the corpus
        self.Vt: np.ndarray | None = None
        # nean of the corpus
        self.mu: np.ndarray | None = None
        # truncated singular values of the corpus
        self.S: np.ndarray | None = None
        # numerical rank of the corpus
        self.numerical_rank: int | None = None

    def fit(self, X: np.ndarray, **kwargs) -> None:
        """
        Fit the object to a corpus X.

        :param X: numpy array, panel data representing the corpus, each row is a data point
        :param y: No use, here for interface consistency

        :return: None
        """
        # mean centering
        self.mu = np.mean(X, axis=0)
        X = X - self.mu

        U, S, Vt = np.linalg.svd(X)
        k = np.sum(self.svd_thres <= S)  # detected numerical rank
        self.numerical_rank = k
        self.Vt = Vt[:k]
        self.S = S[:k]

    @staticmethod
    @jit(nopython=True)  # Observe 6 times speed up on pen-digit dataset
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

        :param x1: 1D array, row vector
        :param x2: 1D array, row vector
        :param Vt: 2D array, truncated right singular matrix transposed of the corpus
        :param S: 1D array, truncated singular values of the corpus
        :subspace_thres: float, threshold to decide whether a point is in the data subspace

        :return: a value representing distance between x, y
        """
        x = x1 - x2
        # quantifies the amount that x is outside the row-subspace
        if np.linalg.norm(x) < zero_thres:
            return 0.0
        rho = np.linalg.norm(x - x @ Vt.T @ Vt) / np.linalg.norm(x)

        if rho > subspace_thres:
            return np.inf

        return x @ Vt.T @ np.diag(S ** (-2)) @ Vt @ x.T

    def distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Compute the variance norm between x1 and x2.

        :param x1: 1D array, row vector
        :param x2: 1D array, row vector

        :return: a value representing distance between x, y
        """
        if self.numerical_rank is None:
            msg = "Mahalanobis distance is not fitted yet."
            raise ValueError(msg)

        return self.calc_distance(
            x1=x1,
            x2=x2,
            Vt=self.Vt,
            S=self.S,
            subspace_thres=self.subspace_thres,
            zero_thres=self.zero_thres,
        )
