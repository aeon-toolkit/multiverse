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
  <a href="docs/experiments.md">Experiments</a>
  ·
  <a href="docs/classifiers.md">Classifiers</a>
  ·
  <a href="docs/contributing.md">Contributing</a>
</p>

**The Multiverse** is a new archive of multivariate time series classification datasets.
This repository is for accessing, benchmarking, and extending this new archive.

It brings together datasets, published results, reproducible evaluation workflows, and leaderboard infrastructure in one place. The aim is to make it easier to:

- access benchmark datasets for <a href="docs/datasets.md"> multivariate time series 
  classification</a>,
- load data into Python workflows,
- explore and compare against <a href="docs/results.md">published results</a>,
- reproduce baseline <a href="docs/experiments.md"> experiments</a>,
- evaluate <a href="docs/classifiers.md">new classifiers consistently</a>,
- and <a href="docs/contributing.md">contribute</a> new algorithms and results back to 
  the archive.

This repository is intended as both a practical resource for researchers and a public record of benchmark results.

---
### Top of the league

Places 1 to 5 by ranks

## Quick start

### Load a dataset
Use the examples in [`examples/aeon_quickstart.py`](examples/aeon_quickstart.py) as a starting point for downloading and loading Multiverse datasets.

### Run an experiment
To reproduce a benchmark run or evaluate a new classifier, start from:

- [`experiments/run_single_dataset.py`](experiments/run_single_dataset.py)
- [`experiments/run_benchmark.py`](experiments/run_benchmark.py)

### Explore the results
Browse the generated leaderboard views in:

- [`leaderboards/leaderboard.md`](leaderboards/leaderboard.md)

### Donate your code and published results

For submission layout and result schema, see:

- [`results/README.md`](results/README.md)
- [`results/schema.md`](results/schema.md)

---

## Repository layout

```text
multiverse/
├── docs/                  # Documentation
├── experiments/           # Benchmark and reproduction scripts
├── results/               # Submitted results and schema
└── src/                   # Python package source for classifiers