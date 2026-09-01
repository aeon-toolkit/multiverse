# Published results

Accuracies reported in the multivariate time series classification literature,
extracted from the papers surveyed in [Benchmark-Driven Multivariate Time Series
Classification: The Role of the UEA MTSC Archive](https://eprints.soton.ac.uk/512258/),
by Bagnall, Ismail-Fawaz and Middlehurst. It surveys 142 peer-reviewed papers published
between 2020 and early 2026 that use the UEA multivariate archive. One file per paper,
grouped by the family the survey assigns it to.

They are here so our runs can be compared against what was published. That comparison
has already found things worth knowing: an aeon implementation 20 accuracy points below
its paper across 23 datasets, and the per-dataset hyperparameters behind several
published figures.

| directory | papers |
|---|---|
| `Convolutional/` | 28 |
| `Deep Other/` | 34 |
| `Transformer/` | 29 |
| `Foundation/` | 19 |
| `MachineLearning/` | 8 |

Alongside them: `Supplementary.pdf`, `final_selection.xlsx` listing the surveyed papers,
and `mtsc.bib`. Several directories carry a `README.txt` or `readme_batch*.txt` recording
which table in each paper a file was taken from, and any caveat about it. Read those
before using a file; they note, for instance, where a paper reports archive-level summary
metrics rather than per-dataset accuracies.

## Format

Most files are wide: a `Dataset` column, then one column per method, with the paper's own
method usually first.

```
Dataset,XC,XC_Seq,MC,MF,WM,ED,DWI,DWD,ED_n,DWI_n,DWD_n,Batch,Win_pct
Articulary Word Recognition,98.44,99.1,98.26,97.14,95.7,99.56,...
```

Four things to watch, all of which have caught us out.

**Value scale is not consistent.** Some files use percentages, some fractions. Roughly
7,300 values are above 1.5 and 17,400 are at or below it. Normalise per file rather than
globally, and be careful with genuinely low accuracies: 0.13 could be either.

**Not every file is wide.** 105 of 118 are; the rest lead with `Table`, `Scope`,
`Method`, `Setting`, `Type`, `Code`, `Abbreviation` or `Task`, because the paper's table
did not fit the wide shape. Check the header before parsing.

**Not every row is a dataset.** Files carry summary rows such as `Average Rank`, `Best`,
`Wins`, `Losses`, `Draws` and `p-val`. Filter them out or they become datasets with
implausible scores.

**Repeated values across papers are copies, not independent runs.** A baseline column is
usually transcribed from the original paper rather than re-run. Where twelve papers all
report 0.321 for TimesNet on Handwriting, that is one measurement quoted twelve times.
Treat the modal value as the source paper's own number and single-source values as much
weaker evidence.

Two files are not extracted from papers: `Convolutional/uea_30_datasets_accuracy_table_conv_*.csv`
are summary tables built across the convolutional papers, and
`MachineLearning/*_accuracy.csv` carry a `Resamples:` header because they are our own
results in the format used under `results/`, kept here for comparison.

## Using them

```python
import csv

with open("published_results/Convolutional/Fauvel2021XCM.csv", newline="") as f:
    for row in csv.DictReader(f):
        dataset = row["Dataset"].replace(" ", "")   # names carry spaces
        accuracy = float(row["XC"]) / 100           # this file is percentages
```

Dataset names vary between papers: spaces (`Articulary Word Recognition`), the archive
name (`ArticularyWordRecognition`), abbreviations (`AWR`), and occasional misspellings
(`Ering` for `ERing`, `Atrial Fibrilation` for `AtrialFibrillation`). Match on a
normalised form and check what did not match, rather than assuming a lookup covered
everything.

## Comparability

Published numbers are not always produced under the protocol used here. Some come from
pipelines that select the retained epoch on the test set, or tune hyperparameters against
it, which inflates them relative to a held-out protocol. The survey's inclusion criteria
exist for this reason. Treat a gap between our result and a published one as a question
about protocol first and implementation second.
