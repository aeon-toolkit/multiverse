<table>
  <tr>
    <td width="25%" align="center" valign="middle">
      <img src="img/multiverse3.png" alt="Multiverse" width="100%">
    </td>

    <td width="50%" align="center" valign="middle">
      <h1>Welcome to the Multiverse</h1>
    </td>

    <td width="25%" align="center" valign="middle">
      <img src="img/multiverse4.png" alt="Multiverse" width="100%">
    </td>
  </tr>
</table>

<p align="center">
  <strong>The archive and benchmark repository for multivariate time series classification.</strong>
</p>

<p align="center">
  <a href="leaderboards/leaderboard.md">Leaderboard</a>
  ·
  <a href="docs/index.md">Documentation</a>
  ·
  <a href="multiverse_registry/README.md">Dataset registry</a>
  ·
  <a href="results/README.md">Results format</a>
  ·
  <a href="CONTRIBUTING.md">Contributing</a>
</p>

---

## Welcome to the Multiverse

**Multiverse** is a repository for accessing, benchmarking, and extending the new archive for **multivariate time series classification**.

It brings together datasets, published results, reproducible evaluation workflows, and leaderboard infrastructure in one place. The aim is to make it easier to:

- access benchmark datasets,
- load data into Python workflows,
- compare against published results,
- reproduce baseline experiments,
- evaluate new classifiers consistently,
- and contribute new methods back to the archive.

This repository is intended as both a practical resource for researchers and a public record of benchmark results.

---

## What is in this repository?

### 1. Leaderboards
Static leaderboard views for Multiverse sub-archives, with links to richer interactive views where available.

Current entry points include:

- **Main leaderboard**: `leaderboards/leaderboard.md`
- **Multiverse Mini**: *placeholder link*
- **Multivariate EEG leaderboard**: *placeholder link*

### 2. Code examples
Examples showing how to:

- download datasets and load them into memory,
- download published results and compare classifiers,
- run benchmark experiments,
- assess a new classifier on a selected archive,
- prepare and submit results to the archive.

Useful starting points:

- `examples/aeon_quickstart.py`
- `experiments/run_single_dataset.py`
- `experiments/run_benchmark.py`

### 3. Documentation
Guidance on data access, evaluation, leaderboards, and repository structure.

See:

- `docs/index.md`
- `docs/datasets.md`
- `docs/loading.md`
- `docs/evaluation.md`
- `docs/leaderboard.md`

### 4. Dataset registry
The authoritative registry of datasets and metadata used by the archive.

See:

- `multiverse_registry/mtsc_registry.csv`
- `multiverse_registry/README.md`

### 5. Submitted results
A standard structure for storing submitted benchmark results and metadata.

See:

- `results/README.md`
- `results/schema.md`

---

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

For submission layout and result schema, see:

- [`results/README.md`](results/README.md)
- [`results/schema.md`](results/schema.md)

---

## Repository layout

```text
multiverse/
├── docs/                  # Documentation
├── examples/              # Minimal usage examples
├── experiments/           # Benchmark and reproduction scripts
├── leaderboards/          # Leaderboard generation and published views
├── multiverse_registry/   # Dataset registry and metadata
├── results/               # Submitted results and schema
└── src/                   # Python package source