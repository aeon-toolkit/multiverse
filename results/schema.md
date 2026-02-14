# Result submission schema

## Required files

Each submission folder must contain:

- `metrics.csv`
- `run-metadata.json`

Folder structure:

`results/submitted/<algorithm>/<multiverse-version>/`

Example:

`results/submitted/example_algo/multiverse-v1.0.0/metrics.csv`

## metrics.csv

Required columns:
- `dataset`
- `metric` (for example `accuracy`)
- `score` (numeric)
- `split` (`test` or `resamples`)

Optional columns (recommended):
- `fit_seconds`
- `score_seconds`
- `notes`

## run-metadata.json

Must include:
- `algorithm`
- `archive_version`
- `evaluation_regime` (for example `default_split` or `resamples`)
- `software` (a dict of package versions)
- `hardware` (free text is acceptable)
- `command` (how you ran the experiment)
- `git_commit` (commit hash of the code used to produce the results)

## Reproducibility

- Do not tune on the test split.
- If you use per-dataset hyperparameters, state the policy in `notes`.
- If you use resamples, include mean and standard deviation in `metrics.csv` or provide an additional summary file.
