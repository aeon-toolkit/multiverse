# Experimental protocols

This repository uses [`tsml-eval`](https://github.com/time-series-machine-learning/tsml-eval)
to run classification experiments and store predictions. `tsml-eval` is the evaluation
toolkit used by the time-series machine-learning projects around aeon. It loads the
train and test files for a dataset, fits an aeon-compatible classifier, measures the
run, writes predictions and probabilities, and provides utilities for comparing
classifiers over datasets and resamples.

The basic protocol here is deliberately simple: fit on the supplied training split,
then evaluate once on the supplied test split. Resample `0` means the archive's default
train/test split. Additional resample IDs can be run when repeated evaluation is
required.

## Installation

The experiment drivers are an optional dependency because running experiments is not
needed to import the classifiers or read the result tables:

```bash
pip install -e ".[experiments]"
```

At present, the released `tsml-eval` package pins an older aeon range than this project
uses. If pip reports an aeon dependency conflict, install a checkout of the `main`
branch of `tsml-eval` until a compatible release is available:

```bash
git clone https://github.com/time-series-machine-learning/tsml-eval.git
pip install -e ./tsml-eval
```

Deep-learning classifiers also need the project's deep-learning extra:

```bash
pip install -e ".[deep-learning]"
```

## Run one experiment

The smallest complete example is
[`run_single_dataset.py`](../multiverse/experiments/run_single_dataset.py). Set the
dataset path and output path, then choose an archive dataset and a classifier name:

```python
from tsml_eval.experiments import (
    get_classifier_by_name,
    load_and_run_classification_experiment,
)

classifier_name = "ROCKET"
classifier = get_classifier_by_name(classifier_name, random_state=0)

load_and_run_classification_experiment(
    problem_path="C:/Data/Multiverse",
    results_path="./results-raw",
    dataset="BasicMotions",
    classifier=classifier,
    classifier_name=classifier_name,
    resample_id=0,
    overwrite=False,
)
```

`problem_path` must contain the standard archive layout:

```text
<problem_path>/<dataset>/<dataset>_TRAIN.ts
<problem_path>/<dataset>/<dataset>_TEST.ts
```

The function loads those files, fits only on `_TRAIN.ts`, predicts `_TEST.ts`, and
writes:

```text
<results_path>/<classifier>/Predictions/<dataset>/testResample<id>.csv
```

With `overwrite=False`, an existing result is retained, which makes interrupted batch
runs safe to restart. The same call can be put inside loops over classifiers, datasets,
and resample IDs; see `run_benchmark.py` for that pattern.

## Result-file format

The output is a tsml-format classification CSV, rather than a normal rectangular table.
It is intended to be read with `load_classifier_results`, not parsed with
`pandas.read_csv`.

The file contains:

1. A metadata line containing the dataset, classifier, split (`TEST`), resample ID,
   time unit, and a description.
2. A parameter-information line containing the estimator configuration.
3. A summary line containing accuracy, fit time, predict time, benchmark time, memory
   usage, number of classes, and optional train-error-estimation fields.
4. One line per test case containing the true class, predicted class, one probability
   for each class, prediction time, and an optional case description.

For example, a result for `ROCKET` on `BasicMotions` at resample `0` is found at:

```text
results-raw/ROCKET/Predictions/BasicMotions/testResample0.csv
```

Load and score one file as follows:

```python
from tsml_eval.evaluation.storage import load_classifier_results

result = load_classifier_results(
    "results-raw/ROCKET/Predictions/BasicMotions/testResample0.csv"
)
result.calculate_statistics()
print(result.accuracy)
print(result.balanced_accuracy)
print(result.fit_time, result.predict_time)
```

The stored probabilities allow metrics such as AUROC and log loss
to be calculated later. 

## Collate and compare results with tsml-eval

Once one result file exists for every requested classifier, dataset, and resample, use
`evaluate_classifiers_by_problem` to collate and compare them:

```python
from tsml_eval.evaluation import evaluate_classifiers_by_problem

evaluate_classifiers_by_problem(
    load_path="./results-raw",
    classifier_names=["ROCKET", "DrCIF", "ConvTran"],
    dataset_names=["BasicMotions", "ItalyPowerDemand", "Trace"],
    save_path="./evaluations",
    resamples=1,              # evaluates resample 0
    eval_name="example",
    continue_on_missing=False,
)
```

The evaluator finds files using the standard directory layout and writes an evaluation
directory containing per-metric CSV files, summary CSV files with mean scores and mean
ranks, and comparison figures. Set `resamples=30` to evaluate IDs `0` through `29`, or
pass an explicit list such as `[0, 1, 2]`. By default a missing file is an error; use
`continue_on_missing=True` when deliberately allowing incomplete comparisons. The
default behaviour removes incomplete datasets from summary comparisons, which keeps
all classifiers on the same set of completed problems.

## Collate into this repository's result tables

The repository's `ingest.py` converts the raw tsml-eval files into the smaller tables
used by the leaderboard. It loads each file with `load_classifier_results`, calculates
the standard metrics, and writes one file per classifier and metric:

```text
results/multiverse/<classifier>/<classifier>_<metric>.csv
```

For example:

```python
from multiverse.experiments.ingest import ingest

ingest(
    classifier="ROCKET",
    predictions_path="./results-raw",
    datasets=["BasicMotions", "ItalyPowerDemand", "Trace"],
    resample=0,
)
```

The generated metric files have one row per dataset and one column per resample. Their
index is labelled `Resamples:`; when there is one resample, the single column is usually
`0`. Missing prediction files are left out rather than filled with a score, so failures
remain visible and can be reported separately.

To ingest the default Multiverse-core dataset list and the configured classifiers, edit
the settings at the top of `multiverse/experiments/ingest.py` and run:

```bash
python -m multiverse.experiments.ingest
```

Finally, build the HTML leaderboard from the collated tables:

```bash
python -m multiverse.experiments.tables
```

The lower-level table API is also available for inspection:

```python
from multiverse.experiments.tables import load_metric, leaderboard

accuracy = load_metric("ROCKET", "accuracy")
leaderboard(
    datasets=["BasicMotions", "ItalyPowerDemand", "Trace"],
    estimators=["ROCKET", "DrCIF", "ConvTran"],
    metrics=["accuracy", "balacc", "logloss"],
    output_path="./results/multiverse/leaderboard.html",
)
```

The ingested tables are a convenient
summary for this repository's leaderboards and should be regenerated if raw results are
changed.
