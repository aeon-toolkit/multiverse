# Leaderboards

The leaderboards can be interactively generated on the WEBSITE. These are some 
illustrative static leaderboards ranked on classification accuracy. We will embed 
the interactive version and update this dynamic in time. 

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
by default because on the current results the omnibus Friedman test does not reject
over the leading estimators, so the diagram is a single clique and shows nothing the
table does not.

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

**A warning on `max_cd_estimators`.** Truncating the critical difference diagram to the
best `n` estimators changes the statistics rather than just hiding rows. Ranks, the
omnibus test and the corrected alpha are all computed over the subset shown. The
diagram starts with an omnibus Friedman test, and dropping the weakest estimators
compresses the spread of average ranks, which can take that test from rejecting to not
rejecting. When Friedman does not reject, aeon places every estimator in a single clique
and runs no pairwise tests at all, so no differences appear.

On the current Multiverse-core results this is not hypothetical:

| Estimators in the diagram | Friedman p | Outcome |
|---|---|---|
| top 6 | 0.44 | one clique, no pairwise tests |
| top 8 | 0.14 | one clique, no pairwise tests |
| all 10 | 0.0003 | 7 significant pairs at alpha/(k-1) = 0.011 |

Treat a truncated diagram as a statement about that subset only.

`available_estimators()` lists the estimators that have results, and `load_metric()`
returns one estimator's scores for one metric as a `pandas.Series` if you would rather
build your own table.

## Multiverse

## Multiverse-core


## EEG archive

The EEG archive is a collection of EEG classification problems, described in [1]. On 
release, it contains 30 datasets. Two of these are univariate and two are not 
available on zenodo. The resulting list is contained in the multiverse


## UEA archive

People will still use the UEA archive, so it is worth maintaining a list for sanity 
checks. The archive contains 30 datasets, but 