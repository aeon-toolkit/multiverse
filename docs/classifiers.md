# Multivariate time series classification algorithms

A wide range of classifiers are available in ``aeon``. You can list all those capable of 
learning from multiverse data using the 

```python
from aeon.utils.discovery import all_estimators
all = all_estimators("classifier", tag_filter={"capability:multivariate": True})
```
you can also filter on the capability to handle unequal length multivariate like so
```python
from aeon.utils.discovery import all_estimators
all = all_estimators("classifier", tag_filter={"capability:multivariate": True, "unequal_length":True})
```
there is extensive documentation with references about these classifiers in ``aeon``.

# Wrapped classifiers in this package

Some classifiers, particularly deep learning, are not implemented in aeon and do not
have a scikit learn compatible interface. One reason for this is that they tend to separate training and validation
datasets external to the fit function. We believe this increases the danger of leakage between train and test. Hence,
we have wrapped some of the classifiers and encapsulated validation as an option in ``fit``.
These are stored in the multiverse package ``multiverse.classification``.

## TimesNet

TimesNet is frequently used as a benchmark.

Some info herre.