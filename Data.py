import pickle
from scipy.io import arff
import os
import random

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn

DATA_DIR = './data'


def get_corpus_and_outlier_paths(df, desired_class):
    corpus_paths = []
    outlier_paths = []

    for i in range(df.shape[0]):
        if df.iloc[i]['target'] == desired_class:
            corpus_paths.append(np.column_stack([[i for i in range(df.shape[1] - 1)], df.iloc[i][:-1]]))
        else:
            outlier_paths.append(np.column_stack([[i for i in range(df.shape[1] - 1)], df.iloc[i][:-1]]))

    return corpus_paths, outlier_paths


def normalise(streams):
    return [sklearn.preprocessing.MinMaxScaler().fit_transform(stream) for stream in streams]


class Data:
    """
    Hold time-series data and allow augmentations
    """

    def __init__(self, if_sample=True, n_samples=(800, 10, 10), random_seed=1):
        self.corpus = None  # unlabelled corpus consists of streams, numpy.array(numpy.array)
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
        self.corpus = train_df[train_df["Digit"] == digit]["Stream"].values
        self.test_inlier = test_df[test_df["Digit"] == digit]["Stream"].values
        self.test_outlier = test_df[test_df["Digit"] != digit]["Stream"].values

        if self.if_sample:
            self.sample()
        self.corpus, self.test_inlier, self.test_outlier = \
            map(normalise, (self.corpus, self.test_inlier, self.test_outlier))

    def load_language_data(self):
        """
        Load language data set with English and German words.
        :return: None
        """
        paths = np.load(DATA_DIR + 'paths_en_de.npy')
        labels = np.load(DATA_DIR + 'labels_en_de.npy')
        (paths_train, paths_test,
         labels_train, labels_test) = train_test_split(paths, labels, random_state=1, test_size=0.2)
        paths_train = paths_train[labels_train == 0]

        self.corpus = paths_train
        self.test_inlier = paths_test[labels_test == 0]
        self.test_outlier = paths_test[labels_test == 1]
        if self.if_sample:
            self.sample()
        self.corpus, self.test_inlier, self.test_outlier = \
            map(normalise, (self.corpus, self.test_inlier, self.test_outlier))

    def load_ship_movements(self, thres_distance=32000, n_samples=5000):
        """

        :param thres_distance: Must be one of [4000, 8000, 16000, 32000]
        :param n_samples: samples taken for each of the train, test_in and test_out
        :return:
        """

        # process data in a format where it could be indexed.
        def process_data(data_frame):
            data_frame['TimeDiff'] = data_frame['BaseDateTime'].apply(lambda x: np.append(0, np.diff(x) ) )
            data_frame = data_frame[['LAT', 'LON', 'TimeDiff']]
            res = []
            for i in range(len(data_frame)):
                res.append(
                    np.array(
                        list(
                            zip(
                                data_frame.iloc[i].values[0],
                                data_frame.iloc[i].values[1],
                                data_frame.iloc[i].values[2],
                            )
                        )
                    )
                )
            return res

        # subsample data
        def sample_data(ais_by_vessel_split, random_state):
            return ais_by_vessel_split.sample(n=n_samples, weights='SUBSTREAM_WEIGHT', replace=True,
                                              random_state=random_state)

        with open(DATA_DIR + 'inlier_mmsis_train.pkl', 'rb') as f:
            inlier_mmsis_train = pickle.load(f)
        with open(DATA_DIR + 'inlier_mmsis_test.pkl', 'rb') as f:
            inlier_mmsis_test = pickle.load(f)
        with open(DATA_DIR + 'outlier_mmsis.pkl', 'rb') as f:
            outlier_mmsis = pickle.load(f)

        if thres_distance not in [4000, 8000, 16000, 32000]:
            raise ValueError("thres_distance needs to be in [4000, 8000, 16000, 32000]")
        ais_by_vessel_split_local = pd.read_pickle(DATA_DIR + 'substreams_' + str(thres_distance) + '.pkl')

        self.corpus = process_data(sample_data(ais_by_vessel_split_local.loc[inlier_mmsis_train], random_state=1))
        self.test_inlier = process_data(sample_data(ais_by_vessel_split_local.loc[inlier_mmsis_test], random_state=2))
        self.test_outlier = process_data(sample_data(ais_by_vessel_split_local.loc[outlier_mmsis], random_state=3))
        if self.if_sample:
            self.sample()
        self.corpus, self.test_inlier, self.test_outlier = \
            map(normalise, (self.corpus, self.test_inlier, self.test_outlier))

    def load_ucr_dataset(self, data_set_name='Adiac', anomaly_level=0.001, random_state=1):
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
        Level_To_COLUMN = {0.001: 'Atra', 0.05: 'A5tra'}
        comparisons = pd.read_csv(DATA_DIR + 'results_beggel_et_al_2019_tables_2_and_4.csv')
        comparisons = comparisons.set_index('Dataset')
        if data_set_name not in comparisons.index:
            raise ValueError("data_set_name must be in ", comparisons.index)
        DATASET_PATH = DATA_DIR + 'Univariate_arff'
        datatrain = arff.loadarff(
            os.path.join(DATASET_PATH,
                         data_set_name,
                         data_set_name + "_TRAIN.arff")
        )
        datatest = arff.loadarff(
            os.path.join(DATASET_PATH,
                         data_set_name,
                         data_set_name + "_TEST.arff")
        )
        alldata = pd.concat(
            [pd.DataFrame(datatrain[0]), pd.DataFrame(datatest[0])],
            ignore_index=True
        )
        alldata['target'] = pd.to_numeric(alldata['target'])
        corpus_paths, outlier_paths = get_corpus_and_outlier_paths(
            alldata,
            comparisons.loc[data_set_name].normal
        )
        corpus_train, corpus_test = train_test_split(
            corpus_paths,
            test_size=comparisons.loc[data_set_name].Ntes.astype('int'),
            random_state=random_state
        )
        outliers_injection = comparisons.loc[data_set_name][Level_To_COLUMN[anomaly_level]].astype('int')
        if outliers_injection != 0:
            outlier_paths, outlier_paths_to_train = train_test_split(
                outlier_paths,
                test_size=outliers_injection,
                random_state=random_state
            )
            corpus_train = corpus_train + outlier_paths_to_train  # injecting anomaly into the corpus

        self.corpus = corpus_train
        self.test_inlier = corpus_test
        self.test_outlier = outlier_paths
        if self.if_sample:
            self.sample()
        self.corpus, self.test_inlier, self.test_outlier = \
            map(normalise, (self.corpus, self.test_inlier, self.test_outlier))
