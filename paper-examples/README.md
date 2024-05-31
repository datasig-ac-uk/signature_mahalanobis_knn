# Anomaly Detection on Streamed Data - Code for Numerical Experiments

This code implements the numerical experiments described in the paper manuscript
**fill final paper title here**.

Experiments are implemented in Python as a set of Jupyter notebooks, with each
notebook corresponding to a section in the paper and responsible for generating
the experimental results reported in that section.

Before executing the notebooks, please ensure that the requirements listed in
the section below are met. Executing the notebooks should then generate the
following results. Note that for each experiment, you will need to run the
corresponding notebook for downloading and pre-processing the data sources
before running the notebook for the experiment itself. They are denoted by the
`_data` suffix. More information in each of the following sections.

The [Data.py](Data.py) file contains some helper functions for loading the data
used in the experiments. Note this is only used for the experiments in the
[paper-examples](paper-examples) folder and can only be ran once the data has
been downloaded and pre-processed for the experiments.

## [Handwritten digits: pen_digit_anomalies.ipynb](pen_digit_anomalies.ipynb)

Prior to running this experiment notebook, you will need to run the
[pen_digit_anomalies_data.ipynb](pen_digit_anomalies_data.ipynb) notebook to
download and pre-process the data.

## [Marine vessel traffic data: ship_movement_anomalies.ipynb](ship_movement_anomalies.ipynb)

Prior to running this experiment notebook, you will need to run the
[ship_movement_anomalies_data.ipynb](ship_movement_anomalies_data.ipynb)
notebook to download and pre-process the data.

## [Univariate time series: ucr_anomalies.ipynb](ucr_anomalies.ipynb)

Prior to running this experiment notebook, you will need to run the
[ucr_anomalies_data.ipynb](ucr_anomalies_data.ipynb) notebook
to download and pre-process the data.

## [Language dataset: language_dataset_anomalies.ipynb](language_dataset_anomalies.ipynb)

Prior to running this experiment notebook, you will need to run the
[language_dataset_anomalies_data.ipynb](language_dataset_anomalies_data.ipynb)
notebook to download and pre-process the data.

There is some data provided for this in the `data` folder, but the notebook will
provide instructions for processing the data.

# Requirements

The notebooks were implemented using Python 3.9. The list of Python package
dependencies is defined in [requirements.txt](requirements.txt). A typical
process for installing the package dependencies involves creating a new Python
virtual environment and then inside the environment executing:

```bash
pip install -r requirements.txt
```

# Experimental notes

The notebooks were developed and ran on a 2021 Macbook Pro with an M1 Pro chip
and 32GB of RAM and ran with `signature_mahalanobis_knn` version 0.1.0 with
Python 3.9.

Although `signature_mahalanobis_knn` is available for a range of Python
versions, we used 3.9 for the experiments since at the time of development,
`esig` was only available for up to Python 3.9 on PyPI. We use
[`esig`](https://github.com/datasig-ac-uk/esig) and `iisignature` to compute
path signatures in the notebooks, but note that one could also pass in feature
representations (e.g. path signatures) rather than streams to the `fit` and
`conformance` methods of the `SigMahaKNN` class.
