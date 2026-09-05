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

## UEA

The UEA archive is the older 30-dataset collection that almost every published
multivariate result is quoted on, so a table restricted to it is what a reader
comparing against the literature needs. This is a subset view of the same runs that
produce the Multiverse-core leaderboard, not a separate experiment, built by passing
`UEA` as the dataset list:

```python
from aeon.datasets.tsc_datasets import UEA
from multiverse.experiments.tables import available_estimators, leaderboard

leaderboard(
    datasets=sorted(UEA),
    estimators=available_estimators(exclude=("DisjointCNN-Aeon",)),
    sort_by="accuracy",
    title="UEA leaderboard",
    output_path="results/multiverse/leaderboard_uea.html",
)
```

**Read the coverage before the ranking.** Four of the thirty are not in
Multiverse-core and have no results here at all: BasicMotions, FingerMovements,
InsectWingbeat and SelfRegulationSCP2. Of the twenty-six that remain, the table uses
only those with a result for every estimator, and the page lists what it dropped and
why. A partial-coverage table is exactly what the 2026 MTSC survey criticises in the
literature, so the number of datasets is stated on the page rather than left to be
inferred from the ranking.

<!-- UEA_LEADERBOARD:START -->
| # | Estimator | Accuracy rank | Accuracy | Balanced accuracy | AUROC | F1 | Log loss &darr; | Sensitivity | Specificity |
|---|---|---|---|---|---|---|---|---|---|
| 1 | HC2 | **7.37** | **0.7665** | **0.7452** | 0.8823 | 0.7411 | **0.6692** | 0.7470 | **0.7703** |
| 2 | RDST | 8.80 | 0.7459 | 0.7294 | 0.8179 | 0.7243 | 9.1587 | 0.7263 | 0.7560 |
| 3 | MRHydra | 9.20 | 0.7523 | 0.7388 | 0.8236 | **0.7432** | 8.9285 | 0.7599 | 0.7360 |
| 4 | Arsenal | 10.26 | 0.7321 | 0.7134 | 0.8439 | 0.7116 | 5.5624 | 0.7119 | 0.7417 |
| 5 | ROCKET | 10.28 | 0.7317 | 0.7146 | 0.8089 | 0.7130 | 9.6705 | 0.7133 | 0.7393 |
| 6 | RIST | 10.57 | 0.7433 | 0.7278 | 0.8755 | 0.7325 | 0.7983 | 0.7454 | 0.7322 |
| 7 | H-InceptionTime | 10.67 | 0.7223 | 0.7230 | 0.8653 | 0.6967 | 1.5030 | 0.7053 | 0.7345 |
| 8 | CIF | 11.33 | 0.7525 | 0.7378 | 0.8825 | 0.7400 | 0.8488 | **0.7604** | 0.7349 |
| 9 | FreshPRINCE | 11.74 | 0.7422 | 0.7281 | 0.8796 | 0.7239 | 0.7764 | 0.7313 | 0.7457 |
| 10 | DrCIF | 11.83 | 0.7386 | 0.7252 | 0.8734 | 0.7246 | 0.8458 | 0.7384 | 0.7303 |
| 11 | LITETime-MV | 12.00 | 0.7073 | 0.7064 | 0.8568 | 0.6779 | 1.4779 | 0.6905 | 0.7218 |
| 12 | LiteTIME | 12.50 | 0.7087 | 0.7019 | 0.8576 | 0.6751 | 1.6854 | 0.6985 | 0.7204 |
| 13 | DisjointCNN | 13.20 | 0.7011 | 0.7030 | 0.8510 | 0.6704 | 1.7943 | 0.6938 | 0.7070 |
| 14 | QUANT | 13.78 | 0.7285 | 0.7171 | **0.8888** | 0.7195 | 0.8041 | 0.7421 | 0.7074 |
| 15 | STSF | 14.24 | 0.7345 | 0.7223 | 0.8774 | 0.6934 | 0.8338 | 0.7007 | 0.7600 |
| 16 | TS2Vec | 14.78 | 0.7070 | 0.6913 | 0.8470 | 0.6917 | 0.8902 | 0.7150 | 0.6877 |
| 17 | TDE | 15.15 | 0.7079 | 0.6862 | 0.8484 | 0.6775 | 1.1475 | 0.6897 | 0.7095 |
| 18 | PatchMTSC | 15.37 | 0.7110 | 0.6986 | 0.8601 | 0.6899 | 0.7670 | 0.7192 | 0.6928 |
| 19 | ConvTran | 16.11 | 0.6931 | 0.6801 | 0.8552 | 0.6793 | 0.8155 | 0.7049 | 0.6736 |
| 20 | STC | 16.15 | 0.7265 | 0.7036 | 0.8803 | 0.7035 | 0.8124 | 0.7186 | 0.7184 |
| 21 | TSF | 16.17 | 0.7214 | 0.7076 | 0.8671 | 0.6917 | 0.9127 | 0.6977 | 0.7365 |
| 22 | Catch22 | 16.30 | 0.7006 | 0.6854 | 0.8557 | 0.6897 | 0.9814 | 0.7096 | 0.6802 |
| 23 | 1NN-DTW | 17.61 | 0.6848 | 0.6759 | 0.7785 | 0.6702 | 11.3600 | 0.6720 | 0.6879 |
| 24 | TimesURL | 18.24 | 0.6809 | 0.6658 | 0.8290 | 0.6539 | 1.3557 | 0.6698 | 0.6738 |
| 25 | Summary | 19.48 | 0.6477 | 0.6355 | 0.8295 | 0.6206 | 1.3000 | 0.6291 | 0.6589 |
| 26 | TimesNet | 20.09 | 0.6584 | 0.6504 | 0.8332 | 0.6386 | 1.1641 | 0.6628 | 0.6504 |
| 27 | Dummy | 24.78 | 0.2168 | 0.1980 | 0.5000 | 0.0800 | 1.9123 | 0.1853 | 0.2288 |

Average over the 23 UEA datasets with results for every estimator on every metric, ordered by average accuracy rank. Best in each column in bold.
<!-- UEA_LEADERBOARD:END -->

Sortable version with per-metric ranks:
[`results/multiverse/leaderboard_uea.html`](../results/multiverse/leaderboard_uea.html)
([preview](https://raw.githack.com/aeon-toolkit/multiverse/main/results/multiverse/leaderboard_uea.html)).

