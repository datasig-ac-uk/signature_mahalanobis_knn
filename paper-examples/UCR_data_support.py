from __future__ import annotations

import matplotlib.lines as mlines
import numpy as np
import pandas as pd

DATA_DIR = "data/"


def compute_best_and_std(
    data_set,
    iter,
    data,
    signature_maha_knn,
    depth,
    n_neighbours,
    anomaly_level=0.001,
):
    tnfn = {}
    for d in data_set:
        tnfn[d] = []
        for random_state in range(iter):
            # data loading
            data.load_ucr_dataset(
                data_set_name=d, random_state=random_state, anomaly_level=anomaly_level
            )
            # anomaly detection
            # Measure the time spent on fit
            signature_maha_knn.fit(
                knn_library="sklearn",
                X_train=data.corpus,
                signature_kwargs={
                    "augmentation_list": ("addtime",),
                    "depth": depth,
                },
            )

            # Measure the time spent on conformance
            inlier_dists = signature_maha_knn.conformance(
                data.test_inlier, n_neighbors=n_neighbours
            )
            outlier_dists = signature_maha_knn.conformance(
                data.test_outlier, n_neighbors=n_neighbours
            )

            tnfn[d].append(compute_tn_fn_for_all_thres(inlier_dists, outlier_dists))

    roughpathsbest = []
    roughpathserror = []
    for d in data_set:
        balanced_accuracies = []
        for tn, fn in tnfn[d]:
            balanced_accuracies.append(compute_balanced_accuracy(tn, fn))
        roughpathsbest.append(max(np.median(balanced_accuracies, axis=0)))
        bestthreshold = np.argmax(np.median(balanced_accuracies, axis=0))
        roughpathserror.append(
            np.std(balanced_accuracies, axis=0, ddof=1)[bestthreshold]
        )

    return roughpathsbest, roughpathserror


def plot_ucr_result(roughpathsbest, roughpathserror, anomaly_level, ax_num, axs):
    # result from beggels 2019
    comparisons = pd.read_csv(DATA_DIR + "results_beggel_et_al_2019_tables_2_and_4.csv")
    adslbest = np.maximum(comparisons.ADSL, comparisons.ADSLbest)
    adslerror = comparisons["ADSL sd"] * (
        comparisons.ADSLbest < comparisons.ADSL
    ) + comparisons["ADSLbest sd"] * (comparisons.ADSLbest >= comparisons.ADSL)

    # table
    comparison_table = pd.DataFrame(
        np.column_stack(
            [comparisons.Dataset, roughpathsbest, roughpathserror, adslbest, adslerror]
        )
    )

    # plotting
    fontsize = 15
    ax = axs.flatten()[ax_num]

    ax.scatter(adslbest, roughpathsbest)
    ax.set_xlim(0.45, 1.01)
    ax.set_ylim(0.45, 1.01)

    line = mlines.Line2D([0, 1.01], [0, 1.01], color="black", ls=":")
    ax.add_line(line)
    ax.set_xlabel("ADSL: median balanced accuracy", fontsize=fontsize)
    ax.set_ylabel("Signature variance: best achievable median BA", fontsize=fontsize)
    ax.set_title(str(anomaly_level) + " training anomaly rate", fontsize=fontsize)
    ax.tick_params(axis="both", which="major", labelsize=fontsize)
    ax.tick_params(axis="both", which="minor", labelsize=fontsize)
    for i in range(len(adslbest)):
        if comparisons.Dataset[i] in [
            "Wafer",
            "ChlorineConcentration",
            "BeetleFly",
            "Wine",
        ]:
            ax.text(
                adslbest[i] + 0.01,
                roughpathsbest[i],
                comparisons.Dataset[i],
                fontsize=fontsize,
            )
        if comparisons.Dataset[i] in ["FaceFour"]:
            ax.text(
                adslbest[i],
                roughpathsbest[i] + 0.01,
                comparisons.Dataset[i],
                fontsize=fontsize,
            )
        if comparisons.Dataset[i] in ["ECG200", "BirdChicken"]:
            ax.text(
                adslbest[i],
                roughpathsbest[i] - 0.025,
                comparisons.Dataset[i],
                fontsize=fontsize,
            )
        if comparisons.Dataset[i] in ["ToeSegmentation1", "ToeSegmentation2"]:
            ax.text(
                adslbest[i] + 0.01,
                roughpathsbest[i] - 0.005,
                comparisons.Dataset[i][:3] + comparisons.Dataset[i][-1],
                fontsize=fontsize,
            )

    return comparison_table


def compute_tn_fn_for_all_thres(inlier_dists, outlier_dists):
    # The below two lines compute the true negatives and false negatives for all possible threshold
    corpus_variances = list(inlier_dists)
    outlier_variances = list(outlier_dists)
    tn = np.concatenate(
        [
            [0],
            np.cumsum(
                np.array(
                    [
                        y
                        for _, y in sorted(
                            zip(
                                corpus_variances + outlier_variances,
                                [0] * len(corpus_variances)
                                + [1] * len(outlier_variances),
                            )
                        )
                    ]
                )
                == 0
            ),
        ]
    )
    fn = np.concatenate(
        [
            [0],
            np.cumsum(
                np.array(
                    [
                        y
                        for _, y in sorted(
                            zip(
                                corpus_variances + outlier_variances,
                                [0] * len(corpus_variances)
                                + [1] * len(outlier_variances),
                            )
                        )
                    ]
                )
                == 1
            ),
        ]
    )
    return (tn, fn)


def compute_balanced_accuracy(tn, fn):
    # compute balanced accuracy as the average between sensitivity and specificity
    false_pos = np.flip(max(tn) - tn)
    true_pos = np.flip(max(fn) - fn)
    true_neg = max(false_pos) - false_pos
    false_neg = max(true_pos) - true_pos
    sens = true_pos / (true_pos + false_neg)
    spec = true_neg / (true_neg + false_pos)
    return (sens + spec) / 2
