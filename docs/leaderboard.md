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
| 1 | HC2 | **6.48** | **0.7617** | **0.7412** | 0.8752 | **0.7372** | **0.6681** | 0.7429 | **0.7655** |
| 2 | RDST | 8.08 | 0.7407 | 0.7250 | 0.8098 | 0.7197 | 9.3448 | 0.7212 | 0.7512 |
| 3 | MRHydra | 8.50 | 0.7462 | 0.7332 | 0.8145 | 0.7371 | 9.1480 | 0.7526 | 0.7315 |
| 4 | Arsenal | 9.08 | 0.7282 | 0.7103 | 0.8375 | 0.7084 | 5.3575 | 0.7084 | 0.7380 |
| 5 | ROCKET | 9.38 | 0.7265 | 0.7101 | 0.8004 | 0.7084 | 9.8595 | 0.7084 | 0.7341 |
| 6 | H-InceptionTime | 9.42 | 0.7208 | 0.7214 | 0.8605 | 0.6962 | 1.5037 | 0.7046 | 0.7323 |
| 7 | RIST | 9.73 | 0.7374 | 0.7226 | 0.8660 | 0.7261 | 0.7933 | 0.7370 | 0.7292 |
| 8 | CIF | 10.15 | 0.7475 | 0.7334 | 0.8742 | 0.7356 | 0.8410 | **0.7550** | 0.7307 |
| 9 | LITETime-MV | 10.67 | 0.7048 | 0.7040 | 0.8505 | 0.6769 | 1.4815 | 0.6894 | 0.7181 |
| 10 | DrCIF | 10.73 | 0.7328 | 0.7199 | 0.8639 | 0.7193 | 0.8386 | 0.7324 | 0.7250 |
| 11 | QUANT | 12.17 | 0.7245 | 0.7136 | **0.8803** | 0.7161 | 0.7978 | 0.7383 | 0.7035 |
| 12 | DisjointCNN | 12.27 | 0.6943 | 0.6961 | 0.8387 | 0.6627 | 1.8839 | 0.6830 | 0.7044 |
| 13 | STSF | 12.33 | 0.7309 | 0.7191 | 0.8703 | 0.6914 | 0.8255 | 0.6982 | 0.7556 |
| 14 | PatchMTSC | 13.31 | 0.7096 | 0.6977 | 0.8549 | 0.6906 | 0.7619 | 0.7220 | 0.6874 |
| 15 | TS2Vec | 13.58 | 0.6990 | 0.6839 | 0.8334 | 0.6853 | 0.8820 | 0.7088 | 0.6783 |
| 16 | TDE | 13.81 | 0.7026 | 0.6818 | 0.8386 | 0.6745 | 1.1277 | 0.6877 | 0.7016 |
| 17 | ConvTran | 13.85 | 0.6915 | 0.6791 | 0.8493 | 0.6784 | 0.8090 | 0.7032 | 0.6725 |
| 18 | TSF | 14.12 | 0.7183 | 0.7052 | 0.8600 | 0.6901 | 0.9018 | 0.6962 | 0.7324 |
| 19 | STC | 14.29 | 0.7224 | 0.7004 | 0.8722 | 0.7000 | 0.8052 | 0.7138 | 0.7157 |
| 20 | Catch22 | 14.50 | 0.6945 | 0.6800 | 0.8443 | 0.6836 | 0.9690 | 0.7021 | 0.6762 |
| 21 | TimesURL | 16.48 | 0.6745 | 0.6600 | 0.8170 | 0.6473 | 1.3326 | 0.6612 | 0.6703 |
| 22 | TimesNet | 17.35 | 0.6582 | 0.6505 | 0.8275 | 0.6400 | 1.1485 | 0.6648 | 0.6481 |
| 23 | Summary | 17.48 | 0.6431 | 0.6314 | 0.8181 | 0.6179 | 1.2745 | 0.6269 | 0.6521 |
| 24 | Dummy | 22.23 | 0.2286 | 0.2106 | 0.5000 | 0.1044 | 1.8615 | 0.2193 | 0.2193 |

Average over the 24 UEA datasets with results for every estimator on every metric, ordered by average accuracy rank. Best in each column in bold.
<!-- UEA_LEADERBOARD:END -->

Sortable version with per-metric ranks:
[`results/multiverse/leaderboard_uea.html`](../results/multiverse/leaderboard_uea.html)
([preview](https://raw.githack.com/aeon-toolkit/multiverse/main/results/multiverse/leaderboard_uea.html)).

