from __future__ import annotations

import importlib.metadata

import signature_mahalanobis_knn as m


def test_version():
    assert importlib.metadata.version("signature_mahalanobis_knn") == m.__version__
