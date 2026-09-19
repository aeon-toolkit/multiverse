"""Turn raw tsml-eval prediction files into the result files used here.

The experiment scripts write one file per classifier, dataset and resample

    <predictions_path>/<classifier>/Predictions/<dataset>/testResample<id>.csv

holding the labels, predictions and probability estimates. The leaderboard in
multiverse.experiments.tables reads something smaller: one file per classifier
and metric, with a row per dataset

    results/multiverse/<classifier>/<classifier>_<metric>.csv

This module is the step between the two. Metrics are computed by tsml-eval, so
they match the numbers in the existing result files exactly.

Edit the settings in main and run it with

    python -m multiverse.experiments.ingest
"""

from pathlib import Path

import numpy as np
import pandas as pd
from aeon.datasets.tsc_datasets import multiverse_core
from tsml_eval.evaluation.storage import load_classifier_results

# metric name used in the file name -> attribute on a tsml-eval results object
metrics = {
    "accuracy": "accuracy",
    "balacc": "balanced_accuracy",
    "auroc": "auroc_score",
    "f1": "f1_score",
    "logloss": "log_loss",
    "sensitivity": "sensitivity",
    "specificity": "specificity",
}

results_path = Path(__file__).resolve().parents[2] / "results" / "multiverse"


def _load_results(path):
    """Load one prediction file with its statistics calculated by tsml-eval.

    tsml-eval cannot score a test split that is missing a class the model was
    trained on: log loss and the one-vs-rest AUROC both expect a probability
    column per class present, so they raise. Several archive problems split this
    way, the KERAAL multiclass datasets among them. For those the two metrics are
    computed here with the full label set given explicitly: log loss over every
    class, and AUROC as the prevalence-weighted one-vs-rest average over the
    classes present, which is what tsml-eval computes when every class is present.
    Every other metric is tsml-eval's own.
    """
    from sklearn.metrics import log_loss, roc_auc_score

    try:
        return load_classifier_results(str(path))
    except ValueError:
        results = load_classifier_results(str(path), calculate_stats=False)
        # stored as floats, but they are the column indices of the probabilities
        labels = np.asarray(results.class_labels).astype(int)
        probabilities = np.clip(np.asarray(results.probabilities), 0.0, 1.0)
        n_classes = probabilities.shape[1]
        results.log_loss = log_loss(labels, probabilities, labels=list(range(n_classes)))
        present = np.unique(labels)
        results.auroc_score = float(
            np.average(
                [roc_auc_score(labels == c, probabilities[:, c]) for c in present],
                weights=[np.sum(labels == c) for c in present],
            )
        )
        results.calculate_statistics()
        return results


def ingest(classifier, predictions_path, datasets=None, resample=0):
    """Write one file per metric for a classifier, and return the datasets used.

    Datasets with no prediction file are left out rather than filled with a
    placeholder, so a classifier that failed on a problem is missing it, which
    is what the leaderboard reports.
    """
    predictions = Path(predictions_path) / classifier / "Predictions"
    datasets = sorted(multiverse_core) if datasets is None else sorted(datasets)
    found = [d for d in datasets if (predictions / d / f"testResample{resample}.csv").is_file()]

    scores = {name: [] for name in metrics}
    for dataset in found:
        results = _load_results(predictions / dataset / f"testResample{resample}.csv")
        for name, attribute in metrics.items():
            scores[name].append(getattr(results, attribute))

    out = results_path / classifier
    out.mkdir(parents=True, exist_ok=True)
    for name in metrics:
        series = pd.Series(scores[name], index=found)
        series.index.name = "Resamples:"
        series.name = str(resample)
        series.to_csv(out / f"{classifier}_{name}.csv")
    return found


def main():
    predictions_path = "D:/Results/Multiverse/ConvolutionBased"
    classifiers = ["ROCKET"]

    for classifier in classifiers:
        found = ingest(classifier, predictions_path)
        missing = sorted(set(multiverse_core) - set(found))
        print(f"  {classifier}: {len(found)} datasets, {len(metrics)} metrics")
        if missing:
            print(f"    missing: {', '.join(missing)}")


if __name__ == "__main__":
    main()
