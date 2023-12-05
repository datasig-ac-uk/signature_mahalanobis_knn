from __future__ import annotations

import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.io import arff
from sklearn.model_selection import train_test_split
from tqdm import tqdm

DATA_DIR = "data/"


def download(source_url, target_filename, chunk_size=1024):
    response = requests.get(source_url, stream=True)

    with Path(target_filename).open("wb") as handle:
        for data in response.iter_content(chunk_size=chunk_size):
            handle.write(data)


def get_corpus_and_outlier_paths(df, desired_class):
    corpus_paths = []
    outlier_paths = []

    for i in range(df.shape[0]):
        if df.iloc[i]["target"] == desired_class:
            corpus_paths.append(
                np.column_stack([list(range(df.shape[1] - 1)), df.iloc[i][:-1]])
            )
        else:
            outlier_paths.append(
                np.column_stack([list(range(df.shape[1] - 1)), df.iloc[i][:-1]])
            )

    return corpus_paths, outlier_paths


def normalise(streams, minimum=None, maximum=None):
    if minimum is None and maximum is None:
        streams_stacked = np.vstack(streams)
        minimum = np.min(streams_stacked, axis=0)
        maximum = np.max(streams_stacked, axis=0)

    streams = [(stream - minimum) / (maximum - minimum) for stream in streams]

    return streams, minimum, maximum


def normalise_data(corpus, inliers, outliers):
    corpus, minimum, maximum = normalise(corpus)
    inliers, *_ = normalise(inliers, minimum, maximum)
    outliers, *_ = normalise(outliers, minimum, maximum)

    return corpus, inliers, outliers


class Data:
    """
    Hold time-series data and allow augmentations
    """

    def __init__(self, if_sample=False, n_samples=(800, 10, 10), random_seed=1):
        self.corpus = (
            None  # unlabelled corpus consists of streams, numpy.array(numpy.array)
        )
        self.test_inlier = None  # test set consists of numpy.array of inliers,
        self.test_outlier = None  # test set consists of numpy.array of outliers
        self.if_sample = if_sample
        self.n_samples = n_samples
        self.random_seed = random_seed

    def sample(self):
        random.seed(self.random_seed)
        self.corpus = random.choices(list(self.corpus), k=self.n_samples[0])
        self.test_inlier = random.choices(list(self.test_inlier), k=self.n_samples[1])
        self.test_outlier = random.choices(list(self.test_outlier), k=self.n_samples[2])

    def load_pen_digit(self, digit: int = 1):
        """
        Load pen digit dataset with a specific digit as training set
        :param digit: 0-9, use as "normality" training corpus
        :return: None
        """
        train_df = pd.read_pickle(DATA_DIR + "pen_digit_train.pkl")
        test_df = pd.read_pickle(DATA_DIR + "pen_digit_test.pkl")
        self.corpus = train_df[train_df["Digit"] == digit]["Stream"].to_numpy()
        self.test_inlier = test_df[test_df["Digit"] == digit]["Stream"].to_numpy()
        self.test_outlier = test_df[test_df["Digit"] != digit]["Stream"].to_numpy()

        if self.if_sample:
            self.sample()
        self.corpus, self.test_inlier, self.test_outlier = normalise_data(
            corpus=self.corpus,
            inliers=self.test_inlier,
            outliers=self.test_outlier,
        )

    def load_language_data(self, cumsum, with_prefix):
        """
        Load language data set with English and non-English words.
        :return: None
        """

        def construct_path(char_seq: str | list[str], alpha_len: int = 26) -> np.array:
            # construct path via one-hot encoding of characters
            n = len(char_seq)
            its = np.zeros(n, np.int64)
            for i in range(n):
                its[i] = ord(char_seq[i]) - 97
            A = np.zeros((n, alpha_len))
            j = 0
            for i in its:
                A[j, i] += 1
                j += 1

            return A

        def get_one_hot_paths_from_words(
            words: list[str],
            cumsum_transform: bool = True,
            alpha_len: int = 26,
        ) -> list[np.array]:
            # construct paths from words and optionally apply cumsum transform
            if cumsum_transform:
                return [
                    np.cumsum(
                        construct_path(char_seq=word, alpha_len=alpha_len), axis=0
                    )
                    for word in tqdm(words)
                ]

            return [
                construct_path(char_seq=word, alpha_len=alpha_len)
                for word in tqdm(words)
            ]

        # load in data frames
        english_train_pickle_file = DATA_DIR + "english_train.pkl"
        corpus_sample_pickle_file = DATA_DIR + "corpus_sample.pkl"
        english_train = pd.read_pickle(english_train_pickle_file)
        corpus_sample_df = pd.read_pickle(corpus_sample_pickle_file)

        if with_prefix:
            # concatenate a prefix to each word
            prefix_str = "concatenatedprefix"
            english_words = [f"{prefix_str}{word}" for word in english_train["word"]]
            corpus_words = [f"{prefix_str}{word}" for word in corpus_sample_df["word"]]
        else:
            english_words = english_train["word"]
            corpus_words = corpus_sample_df["word"]

        # obtain one-hot paths from words
        self.corpus = get_one_hot_paths_from_words(
            words=english_words,
            cumsum_transform=cumsum,
        )
        test_paths = get_one_hot_paths_from_words(
            words=corpus_words,
            cumsum_transform=cumsum,
        )

        # obtain indices of english and non-english words
        english_word_indices = corpus_sample_df[
            corpus_sample_df["language"] == "en"
        ].index.tolist()
        non_english_word_indices = corpus_sample_df[
            corpus_sample_df["language"] != "en"
        ].index.tolist()

        # split test paths into inliers and outliers
        self.test_inlier = [test_paths[i] for i in english_word_indices]
        self.test_outlier = [test_paths[i] for i in non_english_word_indices]

        if self.if_sample:
            self.sample()
        self.corpus, self.test_inlier, self.test_outlier = normalise_data(
            corpus=self.corpus,
            inliers=self.test_inlier,
            outliers=self.test_outlier,
        )

    def load_ship_movements(
        self,
        include_time_diffs,
        lead_lag_transform,
        invisibility_transform,
        thres_distance=32000,
        n_samples=5000,
    ):
        """

        :param thres_distance: Must be one of [4000, 8000, 16000, 32000]
        :param n_samples: samples taken for each of the train, test_in and test_out
        :return:
        """

        def get_stream(
            vessel, include_time_diffs, lead_lag_transform, invisibility_transform
        ):
            stream = np.column_stack((vessel["LAT"], vessel["LON"]))

            if include_time_diffs:
                stream = np.column_stack(
                    (stream, np.append(0, np.diff(vessel["BaseDateTime"])))
                )

            if lead_lag_transform:
                stream = np.repeat(stream, 2, axis=0)
                stream = np.column_stack((stream[1:, :], stream[:-1, :]))

            if invisibility_transform:
                stream = np.vstack((stream, stream[-1], np.zeros_like(stream[-1])))
                stream = np.column_stack(
                    (stream, np.append(np.ones(stream.shape[0] - 2), [0, 0]))
                )

            return stream

        # process data in a format where it could be indexed.
        def process_data(
            data_frame,
            include_time_diffs,
            lead_lag_transform,
            invisibility_transform,
        ):
            return [
                get_stream(
                    vessel=vessel,
                    include_time_diffs=include_time_diffs,
                    lead_lag_transform=lead_lag_transform,
                    invisibility_transform=invisibility_transform,
                )
                for _, vessel in data_frame.iterrows()
            ]

        # subsample data
        def sample_data(ais_by_vessel_split, random_state):
            return ais_by_vessel_split.sample(
                n=n_samples,
                weights="SUBSTREAM_WEIGHT",
                replace=True,
                random_state=random_state,
            )

        with Path(DATA_DIR + "inlier_mmsis_train.pkl").open("rb") as f:
            inlier_mmsis_train = pickle.load(f)
        with Path(DATA_DIR + "inlier_mmsis_test.pkl").open("rb") as f:
            inlier_mmsis_test = pickle.load(f)
        with Path(DATA_DIR + "outlier_mmsis.pkl").open("rb") as f:
            outlier_mmsis = pickle.load(f)

        if thres_distance not in [4000, 8000, 16000, 32000]:
            msg = "thres_distance needs to be in [4000, 8000, 16000, 32000]"
            raise ValueError(msg)
        ais_by_vessel_split_local = pd.read_pickle(
            DATA_DIR + "substreams_" + str(thres_distance) + ".pkl"
        )

        self.corpus = process_data(
            data_frame=sample_data(
                ais_by_vessel_split_local.loc[inlier_mmsis_train],
                random_state=1,
            ),
            include_time_diffs=include_time_diffs,
            lead_lag_transform=lead_lag_transform,
            invisibility_transform=invisibility_transform,
        )
        self.test_inlier = process_data(
            data_frame=sample_data(
                ais_by_vessel_split_local.loc[inlier_mmsis_test],
                random_state=2,
            ),
            include_time_diffs=include_time_diffs,
            lead_lag_transform=lead_lag_transform,
            invisibility_transform=invisibility_transform,
        )
        self.test_outlier = process_data(
            data_frame=sample_data(
                ais_by_vessel_split_local.loc[outlier_mmsis],
                random_state=3,
            ),
            include_time_diffs=include_time_diffs,
            lead_lag_transform=lead_lag_transform,
            invisibility_transform=invisibility_transform,
        )
        if self.if_sample:
            self.sample()
        self.corpus, self.test_inlier, self.test_outlier = normalise_data(
            corpus=self.corpus,
            inliers=self.test_inlier,
            outliers=self.test_outlier,
        )

    def load_ucr_dataset(
        self, data_set_name="Adiac", anomaly_level=0.001, random_state=1
    ):
        """

         :param data_set_name: Must be one of ['Adiac', 'ArrowHead', 'Beef', 'BeetleFly', 'BirdChicken', 'CBF',
        'ChlorineConcentration', 'Coffee', 'ECG200', 'ECGFiveDays', 'FaceFour',
        'GunPoint', 'Ham', 'Herring', 'Lightning2', 'Lightning7', 'Meat',
        'MedicalImages', 'MoteStrain', 'Plane', 'Strawberry', 'Symbols',
        'ToeSegmentation1', 'ToeSegmentation2', 'Trace', 'TwoLeadECG', 'Wafer',
        'Wine']
         :param anomaly_level: Must be one of [0.001, 0.05]
         :param random_state:
         :return:
        """
        Level_To_COLUMN = {0.001: "Atra", 0.05: "A5tra"}
        comparisons = pd.read_csv(
            DATA_DIR + "results_beggel_et_al_2019_tables_2_and_4.csv"
        )
        comparisons = comparisons.set_index("Dataset")
        if data_set_name not in comparisons.index:
            msg = "data_set_name must be in "
            raise ValueError(msg, comparisons.index)
        DATASET_PATH = DATA_DIR + "Univariate_arff"
        datatrain = arff.loadarff(
            Path(DATASET_PATH) / data_set_name / (data_set_name + "_TRAIN.arff")
        )
        datatest = arff.loadarff(
            Path(DATASET_PATH) / data_set_name / (data_set_name + "_TEST.arff")
        )
        alldata = pd.concat(
            [pd.DataFrame(datatrain[0]), pd.DataFrame(datatest[0])], ignore_index=True
        )
        alldata["target"] = pd.to_numeric(alldata["target"])
        corpus_paths, outlier_paths = get_corpus_and_outlier_paths(
            alldata, comparisons.loc[data_set_name].normal
        )
        corpus_train, corpus_test = train_test_split(
            corpus_paths,
            test_size=comparisons.loc[data_set_name].Ntes.astype("int"),
            random_state=random_state,
        )
        outliers_injection = comparisons.loc[data_set_name][
            Level_To_COLUMN[anomaly_level]
        ].astype("int")
        if outliers_injection != 0:
            outlier_paths, outlier_paths_to_train = train_test_split(
                outlier_paths, test_size=outliers_injection, random_state=random_state
            )
            corpus_train = (
                corpus_train + outlier_paths_to_train
            )  # injecting anomaly into the corpus

        self.corpus = corpus_train
        self.test_inlier = corpus_test
        self.test_outlier = outlier_paths
        if self.if_sample:
            self.sample()
        self.corpus, self.test_inlier, self.test_outlier = normalise_data(
            corpus=self.corpus,
            inliers=self.test_inlier,
            outliers=self.test_outlier,
        )
