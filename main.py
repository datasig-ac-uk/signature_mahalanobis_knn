import time

import numpy as np

import Data
import SigMahaKNN

DATA_DIR = '/Users/zoos/PycharmProjects/Anomaly_detection/data/'

data = Data.Data(n_samples=(50, 5, 5), if_sample=True)
data.load_pen_digit()

print(data.corpus[0])

depth_to_auc = {}
for depth in [1, 2, 3, 4, 5]:
    digit_to_inlier_dists = {}
    digit_to_outlier_dists = {}

    for digit in range(10):
        data.load_pen_digit(digit=digit)
        print("doing digit", digit, "doing signature level ", depth)
        signature_maha_knn = SigMahaKNN.SignatureMahalanobisKNN(
            augmentation_list=None,
            depth=depth,
        )
        # Measure the time spent on fit
        start_time = time.time()
        signature_maha_knn.fit(data.corpus)
        fit_time = time.time() - start_time
        print("fit_time: ", fit_time)

        # Measure the time spent on conformance
        start_time = time.time()
        inlier_dists = signature_maha_knn.conformance(data.test_inlier)
        outlier_dists = signature_maha_knn.conformance(data.test_outlier)
        compute_auc_time = time.time() - start_time
        print("compute_auc_time: ", compute_auc_time)

        digit_to_inlier_dists[digit] = inlier_dists
        digit_to_outlier_dists[digit] = outlier_dists

        auc = signature_maha_knn.compute_auc_given_dists(inlier_dists, outlier_dists)
        print("depth: ", depth, ", Auc of", " digit: ", digit, " is ", auc)

    all_inlier_dists = np.concatenate(list(digit_to_inlier_dists.values()))
    all_outlier_dists = np.concatenate(list(digit_to_outlier_dists.values()))
    auc = signature_maha_knn.compute_auc_given_dists(all_inlier_dists, all_outlier_dists)
    print("Overall, ", "depth: ", depth, "AUC: ", auc)
    depth_to_auc[depth] = auc