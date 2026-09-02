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
| 1 | HC2 | **7.86** | **0.7909** | 0.7518 | **0.8990** | 0.7273 | **0.5383** | 0.7459 | **0.7943** |
| 2 | MRHydra | 8.31 | 0.7837 | **0.7564** | 0.8105 | **0.7316** | 7.7974 | **0.7642** | 0.7757 |
| 3 | RDST | 9.16 | 0.7734 | 0.7333 | 0.7912 | 0.6991 | 8.1667 | 0.7109 | 0.7874 |
| 4 | RIST | 9.76 | 0.7720 | 0.7397 | 0.8748 | 0.7147 | 0.6218 | 0.7408 | 0.7655 |
| 5 | DrCIF | 10.07 | 0.7747 | 0.7429 | 0.8813 | 0.7173 | 0.6484 | 0.7397 | 0.7708 |
| 6 | FreshPRINCE | 10.10 | 0.7743 | 0.7487 | 0.8745 | 0.7211 | 0.6007 | 0.7414 | 0.7770 |
| 7 | CIF | 10.13 | 0.7781 | 0.7471 | 0.8908 | 0.7212 | 0.6430 | 0.7441 | 0.7753 |
| 8 | QUANT | 10.55 | 0.7720 | 0.7462 | 0.8831 | 0.7189 | 0.6175 | 0.7521 | 0.7581 |
| 9 | Arsenal | 10.70 | 0.7680 | 0.7321 | 0.8457 | 0.7024 | 3.8631 | 0.7257 | 0.7732 |
| 10 | ROCKET | 10.82 | 0.7690 | 0.7326 | 0.7925 | 0.7019 | 8.3249 | 0.7200 | 0.7764 |
| 11 | LITETime-MV | 11.14 | 0.7506 | 0.7299 | 0.8513 | 0.6820 | 1.3206 | 0.7132 | 0.7637 |
| 12 | STSF | 11.55 | 0.7724 | 0.7477 | 0.8804 | 0.7080 | 0.6432 | 0.7345 | 0.7826 |
| 13 | H-InceptionTime | 11.63 | 0.7408 | 0.7190 | 0.8496 | 0.6838 | 1.3227 | 0.7223 | 0.7378 |
| 14 | LiteTIME | 12.34 | 0.7341 | 0.7104 | 0.8394 | 0.6680 | 1.4776 | 0.7113 | 0.7336 |
| 15 | PatchMTSC | 13.12 | 0.7428 | 0.6897 | 0.8261 | 0.6533 | 0.7655 | 0.6852 | 0.7352 |
| 16 | ConvTran | 13.14 | 0.7462 | 0.7102 | 0.8592 | 0.6767 | 0.8190 | 0.7159 | 0.7345 |
| 17 | Catch22 | 13.15 | 0.7475 | 0.7181 | 0.8697 | 0.6922 | 0.7147 | 0.7240 | 0.7374 |
| 18 | STC | 13.95 | 0.7545 | 0.7172 | 0.8744 | 0.6940 | 0.6391 | 0.7185 | 0.7537 |
| 19 | TSF | 14.00 | 0.7515 | 0.7236 | 0.8740 | 0.6883 | 0.7252 | 0.7093 | 0.7606 |
| 20 | TDE | 14.63 | 0.7262 | 0.6813 | 0.8374 | 0.6382 | 0.8869 | 0.6714 | 0.7344 |
| 21 | Summary | 16.66 | 0.6858 | 0.6574 | 0.8268 | 0.6230 | 0.9123 | 0.6574 | 0.6844 |
| 22 | TimesURL | 16.95 | 0.6958 | 0.6533 | 0.7906 | 0.5967 | 1.0055 | 0.6257 | 0.6973 |
| 23 | 1NN-DTW | 18.45 | 0.6712 | 0.6454 | 0.7197 | 0.6136 | 11.8506 | 0.6521 | 0.6636 |
| 24 | Dummy | 21.81 | 0.3645 | 0.3029 | 0.5000 | 0.1507 | 1.4067 | 0.2855 | 0.3816 |

Average over the 52 Multiverse-core datasets with results for every estimator on every metric, ordered by average accuracy rank. Best in each column in bold.
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
