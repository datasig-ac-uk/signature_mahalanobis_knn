# SigMahaKNN - Signature Mahalanobis KNN method

## Anamoly detection on multivariate streams with Variance Norm and Path Signature

[![Actions Status][actions-badge]][actions-link]
[![Documentation Status][rtd-badge]][rtd-link]
[![PyPI version][pypi-version]][pypi-link]
[![PyPI platforms][pypi-platforms]][pypi-link]
[![GitHub Discussion][github-discussions-badge]][github-discussions-link]

<!-- SPHINX-START -->

<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/datasig-ac-uk/signature_mahalanobis_knn/workflows/CI/badge.svg
[actions-link]:             https://github.com/datasig-ac-uk/signature_mahalanobis_knn/actions
[conda-badge]:              https://img.shields.io/conda/vn/conda-forge/signature_mahalanobis_knn
[conda-link]:               https://github.com/conda-forge/signature_mahalanobis_knn-feedstock
[github-discussions-badge]: https://img.shields.io/static/v1?label=Discussions&message=Ask&color=blue&logo=github
[github-discussions-link]:  https://github.com/datasig-ac-uk/signature_mahalanobis_knn/discussions
[pypi-link]:                https://pypi.org/project/signature_mahalanobis_knn/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/signature_mahalanobis_knn
[pypi-version]:             https://img.shields.io/pypi/v/signature_mahalanobis_knn
[rtd-badge]:                https://readthedocs.org/projects/signature_mahalanobis_knn/badge/?version=latest
[rtd-link]:                 https://signature_mahalanobis_knn.readthedocs.io/en/latest/?badge=latest

<!-- prettier-ignore-end -->

SigMahaKNN (`signature_mahalanobis_knn`) combines the variance norm (a
generalisation of the Mahalanobis distance) with path signatures for anomaly
detection for multivariate streams. The `signature_mahalanobis_knn` library is a
Python implementation of the SigMahaKNN method. The key contributions of this
library are:

- A simple and efficient implementation of the variance norm distance as
  provided by the `signature_mahalanobis_knn.Mahalanobis` class. The class has
  two main methods:
  - The `fit` method to fit the variance norm distance to a training datase
  - The `distance` method to compute the distance between two `numpy` arrays
    `x1` and `x2`
- A simple and efficient implementation of the SigMahaKNN method as provided by
  the `signature_mahalanobis_knn.SigMahaKNN` class. The class has two main
  methods:
  - The `fit` method to fit a model to a training dataset
    - The `fit` method can take in a corpus of streams as its input (where we
      will compute path signatures of using the `sktime` library with `esig` or
      `iisignature`) _or_ a corpus of path signatures as its input. This also
      opens up the possibility of using other feature represenations and
      applications of using the variance norm distance for anomaly detection
    - Currently, the library uses either `sklearn`'s `NearestNeighbors` class or
      `pynndescent`'s `NNDescent` class to efficiently compute the nearest
      neighbour distances of a new data point to the corpus training data
  - The `conformance` method to compute the conformance score for a set of new
    data points
    - Similarly to the `fit` method, the `conformance` method can take in a
      corpus of streams as its input (where we will compute path signatures of
      using the `sktime` library with `esig` or `iisignature`) _or_ a corpus of
      path signatures as its input

## Installation

The SigMahaKNN library is available on PyPI and can be installed with `pip`:

```bash
pip install signature_mahalanobis_knn
```

## Usage

As noted above, the `signature_mahalanobis_knn` library has two main classes:
`Mahalanobis`, a class for computing the variance norm distance, and
`SigMahaKNN`, a class for computing the conformance score for a set of new data
points.

### Computing the variance norm distance

### Using the SigMahaKNN method for anomaly detection

## Repo structure

The core implementation of the SigMahaKNN method is in the
`src/signature_mahalanobis_knn` folder:

- `mahal_distance.py` contains the implementation of the `Mahalanobis` class to
  compute the variance norm distance
- `sig_maha_knn.py` contains the implementation of the `SigMahaKNN` class to
  compute the conformance scores for a set of new data points against a corpus
  of training data
- `utils.py` contains some utility functions that are useful for the library
- `baselines/` is a folder containing some of the baseline methods we look at in
  the paper - see [paper-examples/README.md](paper-examples/README.md) for more
  details

## Examples

There are various examples in the `examples` and `paper-examples` folder:

- `examples` contains small examples using randomly generated data for
  illustration purposes
- `paper-examples` contains the examples used in the paper (link available
  soon!) where we compare the SigMahaKNN method to other baseline approaches
  (e.g. Isolation Forest and Local Outlier Factor) on real-world datasets
  - There are notebooks for downloading and preprocessing the datasets for the
    examples - see [paper-examples/README.md](paper-examples/README.md) for more
    details

## Contributing

To take advantage of `pre-commit`, which will automatically format your code and
run some basic checks before you commit:

```
pip install pre-commit  # or brew install pre-commit on macOS
pre-commit install  # will install a pre-commit hook into the git repo
```

After doing this, each time you commit, some linters will be applied to format
the codebase. You can also/alternatively run `pre-commit run --all-files` to run
the checks.

See [CONTRIBUTING.md](CONTRIBUTING.md) for more information on running the test
suite using `nox`.
