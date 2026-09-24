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

**Leaderboards:** [Multiverse-core](#multiverse-core-leaderboard) (65 datasets) ·
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
- a curated 65 dataset subset, **Multiverse-core (MV-core)**, for algorithm benchmarking

### Multiverse-core leaderboard

<!-- LEADERBOARD:START -->
| # | Estimator | Accuracy rank | Accuracy | Balanced accuracy | AUROC | F1 | Log loss &darr; | Sensitivity | Specificity |
|---|---|---|---|---|---|---|---|---|---|
| 1 | HC2 | **7.40** | **0.7938** | **0.7543** | **0.8951** | **0.7378** | **0.5316** | 0.7570 | **0.7987** |
| 2 | MRHydra | 8.77 | 0.7838 | 0.7529 | 0.8074 | 0.7336 | 7.7909 | **0.7609** | 0.7828 |
| 3 | RDST | 9.18 | 0.7752 | 0.7396 | 0.7968 | 0.7179 | 8.1012 | 0.7289 | 0.7891 |
| 4 | RIST | 9.82 | 0.7742 | 0.7430 | 0.8677 | 0.7213 | 0.6123 | 0.7469 | 0.7724 |
| 5 | CIF | 10.16 | 0.7775 | 0.7463 | 0.8843 | 0.7293 | 0.6414 | 0.7521 | 0.7767 |
| 6 | DrCIF | 10.37 | 0.7722 | 0.7402 | 0.8734 | 0.7216 | 0.6413 | 0.7437 | 0.7718 |
| 7 | Arsenal | 10.80 | 0.7709 | 0.7336 | 0.8469 | 0.7145 | 3.6382 | 0.7340 | 0.7792 |
| 8 | QUANT | 10.92 | 0.7651 | 0.7362 | 0.8678 | 0.7177 | 0.7263 | 0.7506 | 0.7554 |
| 9 | ROCKET | 11.04 | 0.7701 | 0.7345 | 0.7927 | 0.7134 | 8.2862 | 0.7315 | 0.7781 |
| 10 | LITETime-MV | 11.24 | 0.7483 | 0.7260 | 0.8493 | 0.6851 | 1.3395 | 0.7129 | 0.7668 |
| 11 | STSF | 11.41 | 0.7699 | 0.7447 | 0.8700 | 0.7145 | 0.6742 | 0.7379 | 0.7827 |
| 12 | H-InceptionTime | 11.89 | 0.7379 | 0.7137 | 0.8434 | 0.6837 | 1.3662 | 0.7192 | 0.7389 |
| 13 | Catch22 | 12.90 | 0.7539 | 0.7224 | 0.8680 | 0.7027 | 0.6972 | 0.7290 | 0.7507 |
| 14 | ConvTran | 13.12 | 0.7446 | 0.7110 | 0.8520 | 0.6846 | 0.9070 | 0.7219 | 0.7356 |
| 15 | PatchMTSC | 13.31 | 0.7443 | 0.6959 | 0.8247 | 0.6682 | 0.7905 | 0.6996 | 0.7381 |
| 16 | DisjointCNN | 13.64 | 0.7224 | 0.6975 | 0.8239 | 0.6611 | 2.1344 | 0.6814 | 0.7299 |
| 17 | STC | 13.86 | 0.7518 | 0.7106 | 0.8637 | 0.6853 | 0.6428 | 0.7104 | 0.7599 |
| 18 | TSF | 14.15 | 0.7419 | 0.7138 | 0.8558 | 0.6926 | 0.9914 | 0.7141 | 0.7498 |
| 19 | TDE | 14.94 | 0.7272 | 0.6843 | 0.8379 | 0.6601 | 0.8450 | 0.6919 | 0.7304 |
| 20 | TS2Vec | 15.41 | 0.7192 | 0.6790 | 0.7995 | 0.6562 | 0.7869 | 0.6907 | 0.7125 |
| 21 | Summary | 16.52 | 0.6936 | 0.6614 | 0.8194 | 0.6393 | 0.9505 | 0.6648 | 0.6979 |
| 22 | XCM | 16.80 | 0.6691 | 0.6359 | 0.7915 | 0.5859 | 2.1699 | 0.6299 | 0.6769 |
| 23 | TimesURL | 17.23 | 0.6975 | 0.6565 | 0.7833 | 0.6206 | 0.9962 | 0.6456 | 0.6998 |
| 24 | TimesNet | 17.41 | 0.7001 | 0.6647 | 0.8209 | 0.6399 | 1.2957 | 0.6804 | 0.6919 |
| 25 | Dummy | 22.70 | 0.3748 | 0.3072 | 0.5000 | 0.1836 | 1.3810 | 0.3248 | 0.3774 |

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
