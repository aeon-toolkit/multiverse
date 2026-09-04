# Classifier results

This page describes the prediction and summary results stored in this repository. For
the archive collections, dataset splits and dataset selection, see
[`datasets.md`](datasets.md). For running experiments and converting raw predictions
into these tables, see [`evaluation.md`](evaluation.md).

## Published and generated results

Results from earlier bake-offs are available from
[timeseriesclassification.com](https://timeseriesclassification.com) and can be
loaded through aeon's [results loaders](https://github.com/aeon-toolkit/aeon/blob/main/aeon/benchmarking/results_loaders.py):

```python
from aeon.benchmarking.results_loaders import (
    get_available_estimators,
    get_estimator_results,
)

estimators = get_available_estimators("Classification")
results = get_estimator_results(
    estimators=["HC2"],
    datasets=["Chinatown", "Adiac"],
)
```

The regenerated multiverse results distributed with this repository are stored under
[`published_results/`](../published_results/). Raw prediction files and the scripts
that produce summary tables are described in [`evaluation.md`](evaluation.md).

## Repository summary tables

The checked-in benchmark tables are arranged by collection, estimator and metric. For
example:

```text
results/multiverse/<estimator>/<estimator>_<metric>.csv
```

Each table has one row per dataset and one column per resample. The index is labelled
`Resamples:`. A table named `accuracy_mean.csv` may also be present at the collection
level for a compact estimator-by-dataset view.

Load a metric for one estimator with the repository helpers:

```python
from multiverse.experiments.tables import available_estimators, load_metric

print(available_estimators())
accuracy = load_metric("RIST", "accuracy")
print(accuracy.shape)
```

`load_metric` averages across resample columns when more than one resample is present.
Datasets without a result for an estimator are left missing rather than assigned a
placeholder score. The leaderboard uses the common set of completed datasets when
comparing estimators, so the comparison is made on the same problems.

## Building summaries

After raw tsml-eval files have been ingested, build the HTML leaderboard with:

```bash
python -m multiverse.experiments.tables
```

Or generate one explicitly:

```python
from multiverse.experiments.tables import leaderboard

leaderboard(
    datasets=["BasicMotions", "ItalyPowerDemand", "Trace"],
    estimators=["ROCKET", "DrCIF", "ConvTran"],
    metrics=["accuracy", "balacc", "logloss"],
    output_path="./results/multiverse/leaderboard.html",
)
```

See [`leaderboard.md`](leaderboard.md) for the available views and ranking options.
