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
| 1 | HC2 | **7.45** | **0.7917** | **0.7557** | **0.8935** | **0.7302** | **0.5350** | 0.7469 | **0.7998** |
| 2 | MRHydra | 8.89 | 0.7794 | 0.7520 | 0.8040 | 0.7266 | 7.9526 | **0.7577** | 0.7768 |
| 3 | RDST | 9.29 | 0.7729 | 0.7386 | 0.7928 | 0.7075 | 8.1867 | 0.7172 | 0.7902 |
| 4 | RIST | 9.60 | 0.7744 | 0.7451 | 0.8679 | 0.7174 | 0.6150 | 0.7416 | 0.7743 |
| 5 | CIF | 9.97 | 0.7770 | 0.7487 | 0.8842 | 0.7246 | 0.6442 | 0.7459 | 0.7780 |
| 6 | DrCIF | 9.97 | 0.7731 | 0.7433 | 0.8745 | 0.7189 | 0.6430 | 0.7400 | 0.7736 |
| 7 | QUANT | 10.43 | 0.7668 | 0.7404 | 0.8694 | 0.7166 | 0.7285 | 0.7491 | 0.7570 |
| 8 | LITETime-MV | 10.70 | 0.7511 | 0.7320 | 0.8503 | 0.6878 | 1.3004 | 0.7167 | 0.7660 |
| 9 | Arsenal | 10.75 | 0.7663 | 0.7335 | 0.8419 | 0.7061 | 3.6444 | 0.7266 | 0.7752 |
| 10 | ROCKET | 10.83 | 0.7690 | 0.7362 | 0.7925 | 0.7065 | 8.3274 | 0.7228 | 0.7798 |
| 11 | STSF | 11.01 | 0.7727 | 0.7508 | 0.8723 | 0.7164 | 0.6685 | 0.7412 | 0.7845 |
| 12 | H-InceptionTime | 11.38 | 0.7421 | 0.7208 | 0.8447 | 0.6853 | 1.3448 | 0.7227 | 0.7436 |
| 13 | ConvTran | 12.54 | 0.7490 | 0.7177 | 0.8529 | 0.6862 | 0.8826 | 0.7234 | 0.7419 |
| 14 | DisjointCNN | 12.69 | 0.7296 | 0.7057 | 0.8246 | 0.6641 | 2.1100 | 0.6815 | 0.7431 |
| 15 | PatchMTSC | 13.01 | 0.7454 | 0.6990 | 0.8250 | 0.6671 | 0.7818 | 0.6981 | 0.7397 |
| 16 | Catch22 | 13.07 | 0.7463 | 0.7177 | 0.8605 | 0.6929 | 0.7068 | 0.7229 | 0.7420 |
| 17 | TSF | 13.67 | 0.7426 | 0.7175 | 0.8571 | 0.6896 | 0.9987 | 0.7095 | 0.7521 |
| 18 | STC | 13.68 | 0.7507 | 0.7137 | 0.8624 | 0.6803 | 0.6468 | 0.7036 | 0.7611 |
| 19 | TDE | 14.60 | 0.7251 | 0.6834 | 0.8339 | 0.6446 | 0.8524 | 0.6759 | 0.7349 |
| 20 | TS2Vec | 14.79 | 0.7201 | 0.6809 | 0.7994 | 0.6527 | 0.7835 | 0.6879 | 0.7144 |
| 21 | Summary | 16.37 | 0.6845 | 0.6570 | 0.8113 | 0.6299 | 0.9645 | 0.6621 | 0.6852 |
| 22 | TimesNet | 16.67 | 0.7020 | 0.6709 | 0.8218 | 0.6396 | 1.2889 | 0.6810 | 0.6934 |
| 23 | TimesURL | 16.79 | 0.6950 | 0.6562 | 0.7831 | 0.6052 | 0.9868 | 0.6312 | 0.7016 |
| 24 | Dummy | 21.87 | 0.3709 | 0.3067 | 0.5000 | 0.1626 | 1.3928 | 0.2987 | 0.3880 |

Average over the 56 Multiverse-core datasets with results for every estimator on every metric, ordered by average accuracy rank. Best in each column in bold.
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
