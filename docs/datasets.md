# Datasets

## Downloading and loading

You can retrieve a dataset directly from zenodo with the following aeon code. It will by default 
store the data in your home directory, or you can specify the 

```python
from aeon.classification import load_classification
X,y = load_classification("BasicMotions")
X,y = load_classification("BasicMotions", extract_path="C:\\Temp\\")

```

## Packaging convention

For each dataset, data is store one zip file on Zenodo containing:

- `<NAME>_TRAIN.ts`
- `<NAME>_TEST.ts`

If the original dataset has unequal length or missing values, we also store a file

- `<NAME>_TRAIN_eq.ts`
or
- `<NAME>_TEST_nmv.ts`

If you download the data using aeon, it is stored in the `extract_path` in a directory

- `<NAME>_TRAIN.ts`
- `<NAME>_TEST.ts`
and possibly
- `<NAME>_TRAIN_eq.ts`
or
- `<NAME>_TEST_nmv.ts`
When you first call ``load_classification`` again, it looks first in the `extract_path` 
or your.

```python
X,y = load_classification("BasicMotions") # Load combined train/test
trainX,trainy = load_classification("BasicMotions", split="train")
testX,testy = load_classification("BasicMotions", split="test")
```
Equal length datasets are stored in 3D numpy arrays of shape ``(n_cases, n_channels, n_timepoints)``.
Note this is different to some other packages which assume a single time series is shape
``(n_timepoints, n_channels)`` so if you are not using aeon you may need to reshape it. 

To use ``sklearn`` classifiers directly on multivariate equal length datasets, one option is to flatten
the data so that the 3D array `(n_cases, n_channels, n_timepoints)` becomes a 2D array
of shape `(n_cases, n_channels*n_timepoints)`.

```python
flatTrainX = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
flatTestX = X.reshape(X.shape[0], X.shape[1] * X.shape[2])

```
Unequal length datasets are stored in 
a list of 2D numpy arrays. You can control whether to load the equal length version with the 
parameter ``equal_length``.

```python
X,y = load_classification("JapaneseVowels", load_equal = False) # Unequal length example
```

You can download whole archives from zenodo or in code 
```python
from aeon.classification import load_classification
```
There are lists of datasets in aeon and a dictionary of all zenodo keys.


