this is a work in progress
---



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

The current paper version describes:

- 133 unique MTSC problems
- 147 released datasets when preprocessing variants are included
- a curated 66 dataset subset, **Multiverse-core (MV-core)**, for algorithm benchmarking

### Multiverse-core leaderboard

<!-- LEADERBOARD:START -->
| # | Estimator | Accuracy | Balanced accuracy | AUROC | F1 | Log loss &darr; | Sensitivity | Specificity | Accuracy rank |
|---|---|---|---|---|---|---|---|---|---|
| 1 | HC2 | **0.7909** | 0.7518 | **0.8990** | 0.7273 | **0.5383** | 0.7459 | **0.7943** | **7.64** |
| 2 | MRHydra | 0.7837 | **0.7564** | 0.8105 | **0.7316** | 7.7974 | **0.7642** | 0.7757 | 8.14 |
| 3 | RDST | 0.7734 | 0.7333 | 0.7912 | 0.6991 | 8.1667 | 0.7109 | 0.7874 | 9.02 |
| 4 | RIST | 0.7720 | 0.7397 | 0.8748 | 0.7147 | 0.6218 | 0.7408 | 0.7655 | 9.60 |
| 5 | DrCIF | 0.7747 | 0.7429 | 0.8813 | 0.7173 | 0.6484 | 0.7397 | 0.7708 | 9.86 |
| 6 | FreshPRINCE | 0.7743 | 0.7487 | 0.8745 | 0.7211 | 0.6007 | 0.7414 | 0.7770 | 9.87 |
| 7 | CIF | 0.7781 | 0.7471 | 0.8908 | 0.7212 | 0.6430 | 0.7441 | 0.7753 | 9.94 |
| 8 | QUANT | 0.7720 | 0.7462 | 0.8831 | 0.7189 | 0.6175 | 0.7521 | 0.7581 | 10.29 |
| 9 | Arsenal | 0.7680 | 0.7321 | 0.8457 | 0.7024 | 3.8631 | 0.7257 | 0.7732 | 10.41 |
| 10 | ROCKET | 0.7690 | 0.7326 | 0.7925 | 0.7019 | 8.3249 | 0.7200 | 0.7764 | 10.58 |
| 11 | LITETime-MV | 0.7506 | 0.7299 | 0.8513 | 0.6820 | 1.3206 | 0.7132 | 0.7637 | 10.92 |
| 12 | STSF | 0.7724 | 0.7477 | 0.8804 | 0.7080 | 0.6432 | 0.7345 | 0.7826 | 11.29 |
| 13 | H-InceptionTime | 0.7408 | 0.7190 | 0.8496 | 0.6838 | 1.3227 | 0.7223 | 0.7378 | 11.39 |
| 14 | LiteTIME | 0.7341 | 0.7104 | 0.8394 | 0.6680 | 1.4776 | 0.7113 | 0.7336 | 12.08 |
| 15 | PatchMTSC | 0.7428 | 0.6897 | 0.8261 | 0.6533 | 0.7655 | 0.6852 | 0.7352 | 12.77 |
| 16 | ConvTran | 0.7462 | 0.7102 | 0.8592 | 0.6767 | 0.8190 | 0.7159 | 0.7345 | 12.89 |
| 17 | Catch22 | 0.7475 | 0.7181 | 0.8697 | 0.6922 | 0.7147 | 0.7240 | 0.7374 | 12.93 |
| 18 | STC | 0.7545 | 0.7172 | 0.8744 | 0.6940 | 0.6391 | 0.7185 | 0.7537 | 13.63 |
| 19 | TSF | 0.7515 | 0.7236 | 0.8740 | 0.6883 | 0.7252 | 0.7093 | 0.7606 | 13.63 |
| 20 | TDE | 0.7262 | 0.6813 | 0.8374 | 0.6382 | 0.8869 | 0.6714 | 0.7344 | 14.21 |
| 21 | Summary | 0.6858 | 0.6574 | 0.8268 | 0.6230 | 0.9123 | 0.6574 | 0.6844 | 16.12 |
| 22 | 1NN-DTW | 0.6712 | 0.6454 | 0.7197 | 0.6136 | 11.8506 | 0.6521 | 0.6636 | 17.82 |
| 23 | Dummy | 0.3645 | 0.3029 | 0.5000 | 0.1507 | 1.4067 | 0.2855 | 0.3816 | 20.95 |

Average over the 52 Multiverse-core datasets with results for every estimator on every metric, ordered by average accuracy rank. Best in each column in bold.
<!-- LEADERBOARD:END -->

Rebuilt with `python -m multiverse.experiments.tables`, which also writes a sortable
version with per-metric ranks to
[`results/multiverse/leaderboard.html`](results/multiverse/leaderboard.html)
([preview](https://raw.githack.com/aeon-toolkit/multiverse/main/results/multiverse/leaderboard.html),
since GitHub shows HTML as source). Missing results, and why, are listed on that page.

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
To reproduce a benchmark run or evaluate a new classifier, start from:


- [`experiments/run_single_dataset.py`](experiments/run_single_dataset.py)
- [`experiments/run_benchmark.py`](experiments/run_benchmark.py)

### Donate your code and published results

Coming soon

---

## Repository layout

```text
multiverse/
├── docs/                  # Documentation
├── experiments/           # Benchmark and reproduction scripts
├── results/               # Submitted results and schema
└── multiverse/            # Python package source for classifiers
