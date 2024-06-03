from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sktime.transformations.panel.signature_based import (
    SignatureTransformer,
)


def compute_moment_features(streams: np.array | list[np.array]) -> np.array:
    """
    For each stream in streams, compute the mean and covariances (upper triangle)
    of the stream and concatenate them into a single feature vector.

    Parameters
    ----------
    streams : np.array
        Array of streams, with shape (batch, length, channels),
        or a list of arrays of shape (length, channels).

    Returns
    -------
    np.array
        Array of features, with shape (samples, features).
    """
    features = []
    for stream in streams:
        mean = np.mean(stream, axis=0)
        cov = np.cov(stream, rowvar=False)[np.triu_indices(len(mean))]
        features.append(np.concatenate((mean, cov)))

    return np.array(features)


def compute_signatures(
    X: np.array,
    n_jobs: int = 1,
    signature_kwargs: dict | None = None,
) -> np.array:
    """
    Helper function to compute signatures in parallel.
    For use when computing baselines with signatures.

    Parameters
    ----------
    X : np.array
        Data points, by default None.
        Must support index operation X[i] where
        each X[i] returns a data point in the corpus.
        Typically a three-dimensional array of shape
        (batch, length, channels).
    n_jobs : int, optional
        Parameter for joblib, number of parallel processors to use, by default 1.
        -1 means using all processors, -2 means using all processors but one.
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

    Returns
    -------
    np.array
        Two-dimensional array of signatures.
    """
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

    # initialize signature transformer
    signature_transform = SignatureTransformer(
        **signature_kwargs,
    )

    # compute signatures
    sigs = Parallel(n_jobs=n_jobs)(
        delayed(signature_transform.fit_transform)(X[i]) for i in range(len(X))
    )

    return np.array(pd.concat(sigs))
