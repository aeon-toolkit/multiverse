# Datasets

## Downloading and loading

You can retrieve a dataset directly from zenodo with the following aeon code. It will by default 
store the data in your home directory, or you can specify the path as an argument

```python
from aeon.datasets import load_classification
X,y = load_classification("BasicMotions")
X,y = load_classification("BasicMotions", extract_path="C:\\Temp\\")

```

## Packaging convention

If you download the data using aeon, it is stored in the `extract_path` in a 
directory with the name of the problems and files

- `<NAME>_TRAIN.ts`
- `<NAME>_TEST.ts`
and possibly
- `<NAME>_TRAIN_eq.ts`
or
- `<NAME>_TEST_nmv.ts`

The extra files are for cases where time series in the original dataset has unequal 
length or missing values. Versions with lengths equalised or missing imputed are
stored in files with `_eq` or `_nmv` suffix. 


- When you first call ``load_classification`` again, it looks first in the `extract_path` 
or in your home directory to see if the file exists. If it does, it does not 
  download it again. You can load the combined train/test splits or the train/test 
  separately with the argument `split`:

```python
X,y = load_classification("BasicMotions") # Load combined train/test
trainX,trainy = load_classification("BasicMotions", split="train")
testX,testy = load_classification("BasicMotions", split="test")
```
Equal length datasets are stored in 3D numpy arrays of shape ``(n_cases, n_channels, n_timepoints)``.
Note this is different to some other packages such as tensorflow which assume a single 
time series is shape ``(n_timepoints, n_channels)`` so if you are not using ``aeon`` 
you may need to reshape it. 

To use ``sklearn`` classifiers directly on multivariate equal length datasets, one option is to flatten
the data so that the 3D array `(n_cases, n_channels, n_timepoints)` becomes a 2D array
of shape `(n_cases, n_channels*n_timepoints)`.

```python
flatTrainX = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
flatTestX = X.reshape(X.shape[0], X.shape[1] * X.shape[2])

```
Unequal length datasets are stored in a list of 2D numpy arrays. You can control 
whether to load the equal length version with the parameter ``load_equal_length``.

```python
X,y = load_classification("JapaneseVowels", load_equal_length = False) # Unequal length example
```
Imputed missing value versions can be loaded with the argument ``load_no_missing``. 
You can download whole archives from zenodo or in code 
```python
from aeon.datasets import download_archive

download_archive(archive="UEA", extract_path="C:\\Temp\\")


```
Currently should be one of "EEG","UCR","UEA","Imbalanced","TSR", "Unequal". See 
``aeon`` documentation for more details. 
There are lists of datasets in aeon and a dictionary of all zenodo keys.

```python

from aeon.datasets.tsc_datasets import multiverse_core, multiverse2026, eeg2026
print(len(multiverse_core)) # 66
print(len(multiverse2026)) # 133
print(len(eeg2026))  # 28

```
