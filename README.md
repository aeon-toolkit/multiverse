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
| 1 | HC2 | **8.40** | **0.7887** | 0.7541 | **0.9000** | 0.7346 | **0.5440** | 0.7547 | **0.7910** |
| 2 | MRHydra | 9.21 | 0.7810 | **0.7579** | 0.8130 | **0.7368** | 7.8942 | **0.7715** | 0.7718 |
| 3 | RDST | 10.10 | 0.7707 | 0.7372 | 0.7963 | 0.7105 | 8.2660 | 0.7236 | 0.7833 |
| 4 | RIST | 10.75 | 0.7693 | 0.7422 | 0.8755 | 0.7221 | 0.6294 | 0.7504 | 0.7613 |
| 5 | DrCIF | 10.87 | 0.7721 | 0.7454 | 0.8821 | 0.7248 | 0.6558 | 0.7490 | 0.7669 |
| 6 | CIF | 11.07 | 0.7756 | 0.7497 | 0.8920 | 0.7288 | 0.6497 | 0.7536 | 0.7714 |
| 7 | FreshPRINCE | 11.08 | 0.7717 | 0.7516 | 0.8752 | 0.7293 | 0.6075 | 0.7515 | 0.7731 |
| 8 | Arsenal | 11.42 | 0.7654 | 0.7340 | 0.8471 | 0.7092 | 3.9265 | 0.7337 | 0.7696 |
| 9 | QUANT | 11.65 | 0.7693 | 0.7486 | 0.8839 | 0.7262 | 0.6238 | 0.7616 | 0.7539 |
| 10 | LITETime-MV | 11.97 | 0.7476 | 0.7312 | 0.8518 | 0.6875 | 1.3300 | 0.7200 | 0.7600 |
| 11 | ROCKET | 12.01 | 0.7661 | 0.7345 | 0.7955 | 0.7080 | 8.4299 | 0.7282 | 0.7724 |
| 12 | STSF | 12.60 | 0.7698 | 0.7503 | 0.8813 | 0.7155 | 0.6493 | 0.7439 | 0.7790 |
| 13 | H-InceptionTime | 12.71 | 0.7375 | 0.7205 | 0.8506 | 0.6897 | 1.3334 | 0.7303 | 0.7333 |
| 14 | LiteTIME | 13.22 | 0.7308 | 0.7122 | 0.8402 | 0.6746 | 1.4921 | 0.7199 | 0.7291 |
| 15 | DisjointCNN | 13.63 | 0.7286 | 0.7061 | 0.8354 | 0.6688 | 1.9705 | 0.6889 | 0.7368 |
| 16 | ConvTran | 14.37 | 0.7430 | 0.7139 | 0.8606 | 0.6882 | 0.8300 | 0.7289 | 0.7295 |
| 17 | Catch22 | 14.50 | 0.7442 | 0.7203 | 0.8703 | 0.6996 | 0.7238 | 0.7337 | 0.7326 |
| 18 | PatchMTSC | 14.51 | 0.7395 | 0.6934 | 0.8288 | 0.6660 | 0.7748 | 0.6985 | 0.7300 |
| 19 | STC | 15.19 | 0.7516 | 0.7188 | 0.8748 | 0.7004 | 0.6447 | 0.7264 | 0.7496 |
| 20 | TSF | 15.36 | 0.7484 | 0.7257 | 0.8747 | 0.6952 | 0.7335 | 0.7179 | 0.7565 |
| 21 | TS2Vec | 15.87 | 0.7212 | 0.6849 | 0.8082 | 0.6588 | 0.7326 | 0.6980 | 0.7100 |
| 22 | TDE | 15.93 | 0.7230 | 0.6823 | 0.8383 | 0.6441 | 0.8859 | 0.6786 | 0.7301 |
| 23 | Summary | 18.54 | 0.6814 | 0.6586 | 0.8263 | 0.6294 | 0.9251 | 0.6661 | 0.6787 |
| 24 | TimesNet | 18.86 | 0.6971 | 0.6688 | 0.8280 | 0.6390 | 1.1785 | 0.6850 | 0.6838 |
| 25 | TimesURL | 18.95 | 0.6916 | 0.6563 | 0.7931 | 0.6084 | 1.0193 | 0.6379 | 0.6914 |
| 26 | 1NN-DTW | 20.47 | 0.6672 | 0.6457 | 0.7214 | 0.6193 | 11.9949 | 0.6584 | 0.6584 |
| 27 | Dummy | 24.77 | 0.3538 | 0.2991 | 0.5000 | 0.1537 | 1.4284 | 0.2911 | 0.3695 |

Average over the 51 Multiverse-core datasets with results for every estimator on every metric, ordered by average accuracy rank. Best in each column in bold.
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
