<table>
  <tr>
    <td width="25%" align="center"><img src="img/multiverse3.png" width="100%"></td>
    <td width="50%" align="center"><h1>Welcome to the Multiverse</h1></td>
    <td width="25%" align="center"><img src="img/multiverse4.png" width="100%"></td>
  </tr>
</table>

<p align="center">
  <strong>The archive and benchmark repository for multivariate time series classification.</strong>
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

**The Multiverse** is a new archive of multivariate time series classification datasets.
This repository is for accessing, benchmarking, and extending this new archive.

It brings together datasets, published results, reproducible evaluation workflows, and leaderboard infrastructure in one place. The aim is to make it easier to:

- access the <a href="docs/datasets.md"> multiverse </a>, a collection of benchmark 
  datasets for  multivariate time series 
  classification,
- explore and compare against <a href="docs/results.md">published results</a> of 
  classification algorithms,
- reproduce baseline <a href="docs/experiments.md"> experiments</a>,
- evaluate <a href="docs/classifiers.md">new classifiers consistently</a>,
- and <a href="docs/contributing.md">contribute</a> new algorithms and results back to 
  the archive.

This repository is intended as both a practical resource for researchers and a 
public record of benchmark results.

---
### Top of the league

Places 1 to 5 by ranks

Further information and more extensive leaderboard views linked here:

- [`docs/leaderboard.md`](leaderboards/leaderboard.md)

## Install package

Install the release package from PyPI:

```bash
pip install aeon-multiverse
```
or install the development version from GitHub:

```bash
pip install git+https://github.com/aeon-toolkit/multiverse.git
```

### Load a dataset

Use ``aeon`` to download data from zenodo and load into memory.

```python
from aeon.datasets import load_classification
X, y = load_classification("BasicMotions")
print(X.shape)
print(y[:10])
trainX, trainy = load_classification("BasicMotions", split="train")
testX, testy = load_classification("BasicMotions", split="test")

```

More info and links to code - [`docs/leaderboard.md`](docs/leaderboard.md)

### Train and test a classifier
Train and test any aeon classifier that can 
```python
from aeon.classification.deep_learning import InceptionTimeClassifier
from multiverse.classification import TimesNet

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
Or explore published results explored in this repo - [`docs/results.md`]
(docs/results.md)

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