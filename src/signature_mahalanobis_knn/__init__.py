"""
Copyright (c) 2023 DataSig, Zhen Shao, Ryan Chan. All rights reserved.

signature_mahalanobis_knn: Using Nearest Neighbour-Variance Norm with Path Signatures for anomaly detection of streams
"""


from __future__ import annotations

__version__ = "0.0.2"

from .mahal_distance import Mahalanobis
from .sig_mahal_knn import SignatureMahalanobisKNN

__all__ = (
    "__version__",
    "Mahalanobis",
    "SignatureMahalanobisKNN",
)
