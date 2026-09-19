This is a work in progress, we are adding results as we generate them.



<table>
  <tr>
    <td width="25%" align="center"><img src="img/multiverse3.png" width="100%"></td>
    <td width="50%" align="center"><h1>Welcome to the Multiverse</h1></td>
    <td width="25%" align="center"><img src="img/multiverse4.png" width="100%"></td>
  </tr>
</table>

<p align="center">
  <strong>The Multiverse archive for multivariate time series classification.</strong>

The **Multiverse** is an expanded archive for multivariate time series classification 
(MTSC), together with supporting code, metadata, and benchmark results. It consolidates 
datasets from the original UEA MTSC archive, newer MTSC collections, donated 
standalone datasets, and associated benchmark results into a single open repository.

**Leaderboards:** [Multiverse-core](#multiverse-core-leaderboard) (66 datasets) ·
[Full archive](docs/leaderboard_full.md) (the paper's 100 datasets) ·
[EEG](docs/leaderboard_eeg.md) (26 datasets)

The archive is described in
[The Multiverse of Time Series Machine Learning: an Archive for Multivariate Time Series
Classification](https://arxiv.org/abs/2603.20352) (arXiv:2603.20352). If you use the
archive, please cite it; `CITATION.cff` has the full entry.

The wider literature the archive is measured against is reviewed in
[Benchmark-Driven Multivariate Time Series Classification: The Role of the UEA MTSC
Archive](https://eprints.soton.ac.uk/512258/), a survey of 142 papers using the UEA
archive. The accuracies it collected are in
[`published_results/`](published_results/), and are what our runs are compared against.

The current paper version describes:

- 133 unique MTSC problems
- 147 released datasets when preprocessing variants are included
- a curated 66 dataset subset, **Multiverse-core (MV-core)**, for algorithm benchmarking

### Multiverse-core leaderboard

<!-- LEADERBOARD:START -->
| # | Estimator | Accuracy rank | Accuracy | Balanced accuracy | AUROC | F1 | Log loss &darr; | Sensitivity | Specificity |
|---|---|---|---|---|---|---|---|---|---|
| 1 | HC2 | **7.52** | **0.7936** | **0.7528** | **0.8949** | **0.7337** | **0.5307** | 0.7499 | **0.8018** |
| 2 | MRHydra | 9.02 | 0.7806 | 0.7503 | 0.8039 | 0.7293 | 7.9077 | **0.7584** | 0.7794 |
| 3 | RDST | 9.34 | 0.7746 | 0.7371 | 0.7933 | 0.7113 | 8.1235 | 0.7199 | 0.7924 |
| 4 | RIST | 9.84 | 0.7750 | 0.7428 | 0.8685 | 0.7196 | 0.6087 | 0.7424 | 0.7758 |
| 5 | CIF | 10.19 | 0.7782 | 0.7459 | 0.8847 | 0.7273 | 0.6376 | 0.7473 | 0.7799 |
| 6 | DrCIF | 10.36 | 0.7731 | 0.7401 | 0.8743 | 0.7202 | 0.6370 | 0.7396 | 0.7752 |
| 7 | QUANT | 10.87 | 0.7667 | 0.7370 | 0.8693 | 0.7177 | 0.7198 | 0.7476 | 0.7594 |
| 8 | Arsenal | 10.99 | 0.7683 | 0.7309 | 0.8425 | 0.7096 | 3.5868 | 0.7293 | 0.7777 |
| 9 | ROCKET | 11.14 | 0.7698 | 0.7328 | 0.7899 | 0.7088 | 8.2968 | 0.7245 | 0.7811 |
| 10 | LITETime-MV | 11.16 | 0.7505 | 0.7274 | 0.8511 | 0.6864 | 1.3223 | 0.7113 | 0.7707 |
| 11 | STSF | 11.23 | 0.7735 | 0.7486 | 0.8722 | 0.7188 | 0.6639 | 0.7417 | 0.7862 |
| 12 | H-InceptionTime | 11.76 | 0.7418 | 0.7179 | 0.8460 | 0.6881 | 1.3451 | 0.7233 | 0.7428 |
| 13 | ConvTran | 12.95 | 0.7485 | 0.7152 | 0.8545 | 0.6892 | 0.8930 | 0.7251 | 0.7401 |
| 14 | Catch22 | 13.09 | 0.7495 | 0.7189 | 0.8622 | 0.6977 | 0.6971 | 0.7263 | 0.7459 |
| 15 | PatchMTSC | 13.16 | 0.7478 | 0.6996 | 0.8276 | 0.6723 | 0.7796 | 0.7018 | 0.7425 |
| 16 | DisjointCNN | 13.44 | 0.7267 | 0.7021 | 0.8267 | 0.6662 | 2.1025 | 0.6857 | 0.7344 |
| 17 | STC | 13.84 | 0.7527 | 0.7109 | 0.8633 | 0.6843 | 0.6397 | 0.7072 | 0.7629 |
| 18 | TSF | 14.06 | 0.7434 | 0.7142 | 0.8570 | 0.6918 | 0.9812 | 0.7105 | 0.7536 |
| 19 | TDE | 15.01 | 0.7266 | 0.6811 | 0.8344 | 0.6488 | 0.8411 | 0.6800 | 0.7351 |
| 20 | TS2Vec | 15.41 | 0.7203 | 0.6798 | 0.8005 | 0.6556 | 0.7806 | 0.6886 | 0.7154 |
| 21 | Summary | 16.66 | 0.6886 | 0.6585 | 0.8135 | 0.6357 | 0.9475 | 0.6661 | 0.6901 |
| 22 | XCM | 16.78 | 0.6707 | 0.6357 | 0.7914 | 0.5827 | 2.1902 | 0.6233 | 0.6824 |
| 23 | TimesNet | 17.22 | 0.7039 | 0.6685 | 0.8236 | 0.6437 | 1.2796 | 0.6824 | 0.6968 |
| 24 | TimesURL | 17.28 | 0.6973 | 0.6538 | 0.7828 | 0.6099 | 0.9919 | 0.6345 | 0.7050 |
| 25 | Dummy | 22.66 | 0.3802 | 0.3105 | 0.5000 | 0.1804 | 1.3681 | 0.3192 | 0.3882 |

Average over the 58 Multiverse-core datasets with results for every estimator on every metric, ordered by average accuracy rank. Best in each column in bold.
<!-- LEADERBOARD:END -->

Rebuilt with `python -m multiverse.experiments.tables`, which also writes a sortable
version with per-metric ranks to
[`results/multiverse/leaderboard.html`](results/multiverse/leaderboard.html)
([preview](https://raw.githack.com/aeon-toolkit/multiverse/main/results/multiverse/leaderboard.html),
since GitHub shows HTML as source). Missing results, and why, are listed on that page.

The same command writes a per-dataset view to
[`results/multiverse/datasets.html`](results/multiverse/datasets.html)
([preview](https://raw.githack.com/aeon-toolkit/multiverse/main/results/multiverse/datasets.html)),
which turns the question around: for each dataset it gives the Dummy floor, the median
and best over the other estimators, which estimator was best, how much the best gained
over Dummy, and how far apart the estimators were. It is sorted by that gain, so the
problems where nothing yet beats the baseline come first.

This repository aims to make it easier to:

- load Multiverse datasets through `aeon`
- inspect archive metadata and dataset variants
- reproduce baseline benchmark results
- compare against published and recreated results
- contribute new results, metadata, and documentation as the archive evolves

</p>

<p align="center">
  <a href="docs/datasets.md">Datasets</a>
  ·
  <a href="docs/results.md">Results</a>
  ·
  <a href="docs/leaderboard.md">Leaderboard</a>
  ·
  <a href="docs/runtime.md">Runtime</a>
  ·
  <a href="docs/memory.md">Memory</a>
  ·
  <a href="docs/evaluation.md">Evaluation</a>
  ·
  <a href="docs/classifiers.md">Classifiers</a>
  ·
  <a href="docs/contributing.md">Contributing</a>
</p>

## Installation

Install the release package from PyPI:

```bash
pip install aeon-multiverse
```

or the development version from GitHub:

```bash
pip install git+https://github.com/aeon-toolkit/multiverse.git
```

At present the safest route is to install from source, since the package is changing
rapidly:

```bash
git clone https://github.com/aeon-toolkit/multiverse.git
cd multiverse
pip install -e .
```

This repository depends on `aeon` and uses the `aeon` dataset loading interface as the
main public API for archive access.

## Quick start

### Load a dataset

The archive datasets are published on Zenodo, in the
[tsml community](https://zenodo.org/communities/tsml/records?q=&f=subject%3Auea%20archive&l=list&p=1&s=20&sort=newest).
You do not need to download them by hand: use ``aeon`` to fetch a dataset from Zenodo and
load it into memory.

```python
from aeon.datasets import load_classification

X, y = load_classification("BasicMotions")
train_X, train_y = load_classification("BasicMotions", split="train")
test_X, test_y = load_classification("BasicMotions", split="test")

print(X.shape)
```

More info and links to code - [`docs/datasets.md`](docs/datasets.md)

### Train and test a classifier

```python
from aeon.classification.deep_learning import InceptionTimeClassifier
from multiverse.classification import (
    ConvTranClassifier,
    PatchMTSCClassifier,
    TimesNetClassifier,
)

clf = InceptionTimeClassifier()
clf.fit(X, y)
preds = clf.predict(X)
```
More info and links to aeon classifiers - [`docs/classifiers.md`](docs/classifiers.md)
Multiverse ported classifiers - [`multiverse/classification`](multiverse/classification)

### Compare your results to published results
Load results directly in code
```python
from aeon.classification.deep_learning import InceptionTimeClassifier

```
Or explore published results explored in this repo - [`docs/results.md`](docs/results.md)

### Run an experiment

Results are generated with [`tsml_eval`](https://github.com/time-series-machine-learning/tsml-eval),
which writes one file per classifier, dataset and resample in the format the tooling in
this repository reads:

```text
<results_path>/<classifier>/Predictions/<dataset>/testResample<id>.csv
```

It is the `experiments` extra, `pip install aeon-multiverse[experiments]`, since only
these scripts need it. Its current release pins `aeon<1.2.0` and so cannot yet be
installed alongside this package; until the next release, use a checkout of `tsml_eval`
`main`.

Set the data path, results path, classifiers and datasets at the top of `main` in one of:

- [`multiverse/experiments/run_single_dataset.py`](multiverse/experiments/run_single_dataset.py)
  — one classifier on one dataset, the smallest complete example
- [`multiverse/experiments/run_benchmark.py`](multiverse/experiments/run_benchmark.py)
  — a set of classifiers over Multiverse-core
- [`multiverse/experiments/run_eeg_bakeoff.py`](multiverse/experiments/run_eeg_bakeoff.py)
  — the same, over the EEG archive

then run it:

```bash
python -m multiverse.experiments.run_benchmark
```

Combinations that already have results are skipped, so an interrupted run can be started
again, and a failure is reported without ending the run. Running the full benchmark on
one machine takes a very long time; the published results were distributed over a
cluster.

### Donate your code and published results

Coming soon

---

## Repository layout

```text
multiverse/
├── docs/                    # Documentation
├── img/                     # Images used in the documentation
├── results/                 # Benchmark results, one directory per classifier
├── published_results/       # Accuracies reported in the MTSC literature
├── survey/                  # Signpost only: the directory moved to published_results/
└── multiverse/              # Python package
    ├── classification/      # Classifiers not available in aeon
    ├── examples/            # Short runnable examples
    └── experiments/         # Result generation and leaderboard tables
