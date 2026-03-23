# Classifier Results

Results used in past bake offs are available on [tsc.com]
(https://timeseriesclassification.com) and 
obtainable in code with [aeon](https://github.com/aeon-toolkit/aeon/blob/main/aeon/benchmarking/results_loaders.py).

```python

from aeon.benchmarking.results_loaders import get_available_estimators
cls = get_available_estimators("Classification")  # doctest: +SKIP
from aeon.benchmarking.results_loaders import get_estimator_results
cls = ["HC2"]  # doctest: +SKIP
data = ["Chinatown", "Adiac"]  # doctest: +SKIP
get_estimator_results(estimators=cls, datasets=data) # doctest: +SKIP
```

We currently store the multiverse results in the results directory. Currently 
only have accuracy for the default splits for subsets of the multiverse. This is 
still a work in progress. You will soone be able to explore and download these results 
interactively on the[multiverse website](COMING SOON).

The dataset lists are 

```python

from aeon.datasets.tsc_datasets import multiverse_core, multiverse2026, eeg2026
print(len(multiverse_core)) # 66
print(len(multiverse2026)) # 133
print(len(eeg2026))  # 28

```

### The Full Multiverse, 2026

The full multiverse has 133 datasets in it. We have results for 17 classifiers on 
some subset of these problems. 

```python
from pathlib import Path
import pandas as pd

# Run this from the repository root
df = pd.read_csv(Path("results") / "multiverse" / "accuracy_mean.csv")
print(df.head())
```
## The Multiverse-core (M-core)

We specify a subset of 66 datasets for evaluation. These are more balanced in 
application, remove overly similar, too simple or zero information datasets and 
have a good distribution in size and length.

```python
df = pd.read_csv(Path("results") / "multiverse_core" / "accuracy_mean.csv")
print(df.shape)
```


## The EEG Classification archive, 2026

The EEG archive is a sub-project meant to benchmark EEG classification algorithms. 
The project is based around [aeon-neuro](https://github.com/aeon-toolkit/aeon-neuro)


```python
df = pd.read_csv(Path("results") / "eeg" / "accuracy_mean.csv")
print(df.shape)
```

