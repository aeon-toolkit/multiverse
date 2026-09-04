# Leaderboards

The leaderboards can be interactively generated on the WEBSITE COMING SOON. These are some 
illustrative static leaderboards ranked on classification accuracy. We will embed 
the interactive version when its ready. In the interim, we present some generative tools. 

## Generating a leaderboard

`multiverse.experiments.tables` builds a self-contained HTML leaderboard from the
results stored in this repository. Pass a list of datasets, a list of estimators, and
the metric to rank on.

```python
from aeon.datasets.tsc_datasets import multiverse_core
from multiverse.experiments.tables import available_estimators, leaderboard

path = leaderboard(
    datasets=sorted(multiverse_core),
    estimators=available_estimators(exclude=("Dummy",)),
    sort_by="accuracy",
    title="Multiverse-core leaderboard",
)
print(path)  # results/multiverse/leaderboard.html
```

The page holds one table, with an estimator per row and an average score and average
rank for each metric. Only datasets with a result for every estimator on every metric
are used, so each column describes the same problems; anything left out is listed on
the page with the reason taken from `results/multiverse/missing_results.csv`.

A critical difference diagram can be added with `critical_difference=True`. It is off
by default because we generate the front page table for all estimators.

Building the same page from the command line:

```bash
python -m multiverse.experiments.tables
```

Useful arguments:

```python
leaderboard(
    datasets=sorted(multiverse_core),
    estimators=["RIST", "QUANT", "CIF"],   # or available_estimators()
    metrics=["accuracy", "auroc"],         # default: every metric available
    sort_by="auroc",                       # ranks rows, and the CD diagram metric
    critical_difference=True,              # add a CD diagram; off by default
    max_cd_estimators=6,                   # cap the diagram; see the warning below
    output_path="docs/leaderboard_auroc.html",
)
```

Every column in the generated table is sortable: click a heading to sort by it, and
click again to reverse. The first click puts the best value on top, so ascending for
ranks and for log loss, descending for the rest.

