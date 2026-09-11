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
| 1 | HC2 | **7.63** | **0.7927** | **0.7512** | **0.8944** | **0.7318** | **0.5321** | 0.7487 | **0.8007** |
| 2 | MRHydra | 8.98 | 0.7811 | 0.7502 | 0.8047 | 0.7292 | 7.8914 | **0.7598** | 0.7785 |
| 3 | RDST | 9.32 | 0.7747 | 0.7366 | 0.7937 | 0.7105 | 8.1197 | 0.7200 | 0.7918 |
| 4 | RIST | 9.74 | 0.7761 | 0.7434 | 0.8693 | 0.7201 | 0.6103 | 0.7439 | 0.7761 |
| 5 | DrCIF | 10.23 | 0.7744 | 0.7408 | 0.8755 | 0.7210 | 0.6386 | 0.7418 | 0.7749 |
| 6 | CIF | 10.32 | 0.7778 | 0.7449 | 0.8848 | 0.7262 | 0.6402 | 0.7472 | 0.7788 |
| 7 | QUANT | 10.73 | 0.7680 | 0.7377 | 0.8707 | 0.7188 | 0.7224 | 0.7506 | 0.7584 |
| 8 | LITETime-MV | 10.94 | 0.7526 | 0.7292 | 0.8518 | 0.6908 | 1.2944 | 0.7188 | 0.7672 |
| 9 | Arsenal | 11.15 | 0.7671 | 0.7290 | 0.8413 | 0.7075 | 3.6366 | 0.7281 | 0.7759 |
| 10 | ROCKET | 11.21 | 0.7696 | 0.7319 | 0.7900 | 0.7076 | 8.3060 | 0.7242 | 0.7802 |
| 11 | STSF | 11.22 | 0.7740 | 0.7487 | 0.8728 | 0.7187 | 0.6667 | 0.7431 | 0.7856 |
| 12 | H-InceptionTime | 11.81 | 0.7411 | 0.7169 | 0.8455 | 0.6864 | 1.3447 | 0.7220 | 0.7425 |
| 13 | ConvTran | 13.07 | 0.7478 | 0.7140 | 0.8542 | 0.6872 | 0.8894 | 0.7226 | 0.7408 |
| 14 | Catch22 | 13.25 | 0.7488 | 0.7177 | 0.8620 | 0.6962 | 0.7009 | 0.7258 | 0.7445 |
| 15 | PatchMTSC | 13.25 | 0.7472 | 0.6982 | 0.8270 | 0.6704 | 0.7772 | 0.7007 | 0.7416 |
| 16 | DisjointCNN | 13.29 | 0.7278 | 0.7027 | 0.8258 | 0.6648 | 2.1048 | 0.6806 | 0.7411 |
| 17 | STC | 14.00 | 0.7521 | 0.7095 | 0.8632 | 0.6825 | 0.6427 | 0.7059 | 0.7623 |
| 18 | TSF | 14.04 | 0.7439 | 0.7142 | 0.8581 | 0.6917 | 0.9887 | 0.7114 | 0.7533 |
| 19 | TDE | 14.97 | 0.7269 | 0.6807 | 0.8355 | 0.6475 | 0.8456 | 0.6786 | 0.7365 |
| 20 | TS2Vec | 15.31 | 0.7209 | 0.6796 | 0.8015 | 0.6552 | 0.7822 | 0.6892 | 0.7153 |
| 21 | XCM | 16.68 | 0.6710 | 0.6353 | 0.7908 | 0.5801 | 2.1677 | 0.6180 | 0.6876 |
| 22 | Summary | 16.71 | 0.6878 | 0.6572 | 0.8137 | 0.6342 | 0.9545 | 0.6657 | 0.6885 |
| 23 | TimesURL | 17.24 | 0.6974 | 0.6532 | 0.7841 | 0.6089 | 0.9807 | 0.6348 | 0.7040 |
| 24 | TimesNet | 17.29 | 0.7033 | 0.6673 | 0.8237 | 0.6424 | 1.2793 | 0.6827 | 0.6949 |
| 25 | Dummy | 22.62 | 0.3781 | 0.3072 | 0.5000 | 0.1719 | 1.3799 | 0.3072 | 0.3950 |

Average over the 57 Multiverse-core datasets with results for every estimator on every metric, ordered by average accuracy rank. Best in each column in bold.
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
