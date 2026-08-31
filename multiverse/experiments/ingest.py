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
        results = load_classifier_results(
            str(predictions / dataset / f"testResample{resample}.csv")
        )
        results.calculate_statistics()
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
