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
Python implementation of the SigMahaKNN method described in
[_Dimensionless Anomaly Detection on Multivariate Streams with Variance Norm and Path Signature_](https://arxiv.org/abs/2006.03487).

To find the examples from the paper, please see the
[paper-examples](paper-examples) folder which includes notebooks for downloading
and running the experiments.

The key contributions of this library are:

- A simple and efficient implementation of the variance norm distance as
  provided by the `signature_mahalanobis_knn.Mahalanobis` class. The class has
  two main methods:
  - The `fit` method to fit the variance norm distance to a training datase
  - The `distance` method to compute the distance between two `numpy` arrays
    `x1` and `x2`
- A simple and efficient implementation of the SigMahaKNN method as provided by
  the `signature_mahalanobis_knn.SignatureMahalanobisKNN` class. The class has two main
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
`SignatureMahalanobisKNN`, a class for computing the conformance score for a set of new data
points.

### Computing the variance norm distance

To compute the variance norm (a generalisation of the Mahalanobis distance) for a
pair of data points `x1` and `x2` given a corpus of training data `X` (a two-dimensional
`numpy` array), you can use the `Mahalanobis` class as follows:

```python
import numpy as np
from signature_mahalanobis_knn import Mahalanobis

# create a corpus of training data
X = np.random.rand(100, 10)

# initialise the Mahalanobis class
mahalanobis = Mahalanobis()
mahalanobis.fit(X)

# compute the variance norm distance between two data points
x1 = np.random.rand(10)
x2 = np.random.rand(10)
distance = mahalanobis.distance(x1, x2)
```

Here we provided an example with the default initialisation of the `Mahalanobis`
class. There are also a few parameters that can be set when initialising the class
(see details in [_Dimensionless Anomaly Detection on Multivariate Streams with Variance Norm and Path Signature_](https://arxiv.org/abs/2006.03487)):
- `subspace_thres`: (float) threshold for deciding whether or not a point is in the subspace, default is 1e-3
- `svd_thres`: (float) threshold for deciding the numerical rank of the data matrix, default is 1e-12
- `zero_thres`: (float) threshold for deciding whether the distance should be set to zero, default is 1e-12

### Using the SigMahaKNN method for anomaly detection

To use the SigMahaKNN method for anomaly detection of multivariate streams, you
can use the `SignatureMahalanobisKNN` class by first initialising the class and then using the
`fit` and `conformance` methods to fit a model to a training dataset of streams and compute
the conformance score for a set of new data streams, respectively:

```python
import numpy as np
from signature_mahalanobis_knn import SignatureMahalanobisKNN

# create a corpus of training data
# X is a three-dimensional numpy array with shape (n_samples, length, channels)
X = np.random.rand(100, 10, 3)

# initialise the SignatureMahalanobisKNN class
sig_maha_knn = SignatureMahalanobisKNN()
sig_maha_knn.fit(
  knn_library="sklearn",
  X_train=X,
  signature_kwargs={"depth": 3},
)

# create a set of test data streams
Y = np.random.rand(10, 10, 3)

# compute the conformance score for the test data streams
conformance_scores = sig_maha_knn.conformance(X_test=Y, n_neighbors=5)
```

Note here, we have provided an example whereby you pass in a corpus of streams to
fit and compute the conformance scores. We use the `sktime` library to compute
path signatures of the streams. 

However, if you already have computed signatures or you are using another feature representation method, you can pass in the corpus of
signatures to the `fit` and `conformance` methods instead of the streams. You do this by
passing in arguments `signatures_train` and `signatures_test` to the `fit` and `conformance`
methods, respectively.

```python
import numpy as np
from signature_mahalanobis_knn import SignatureMahalanobisKNN

# create a corpus of training data (signatures or other feature representations)
# X is a two-dimensional numpy array with shape (n_samples, n_features)
features = np.random.rand(100, 10)

# initialise the SignatureMahalanobisKNN class
sig_maha_knn = SignatureMahalanobisKNN()
sig_maha_knn.fit(
  knn_library="sklearn",
  signatures_train=features,
)

# create a set of test features
features_y = np.random.rand(10, 10)

# compute the conformance score for the test features
conformance_scores = sig_maha_knn.conformance(signatures_test=features_y, n_neighbors=5)
```

## Repo structure

The core implementation of the SigMahaKNN method is in the
`src/signature_mahalanobis_knn` folder:

- `mahal_distance.py` contains the implementation of the `Mahalanobis` class to
  compute the variance norm distance
- `sig_maha_knn.py` contains the implementation of the `SignatureMahalanobisKNN` class to
  compute the conformance scores for a set of new data points against a corpus
  of training data
- `utils.py` contains some utility functions that are useful for the library
- `baselines/` is a folder containing some of the baseline methods we look at in
  the paper - see [paper-examples/README.md](paper-examples/README.md) for more
  details

## Examples

There are various examples in  `paper-examples` folder:

- `paper-examples` contains the examples used our paper
  [_Dimensionless Anomaly Detection on Multivariate Streams with Variance Norm and Path Signature_](https://arxiv.org/abs/2006.03487)
  where we compare the SigMahaKNN method to other baseline approaches (e.g.
  Isolation Forest and Local Outlier Factor) on real-world datasets
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
