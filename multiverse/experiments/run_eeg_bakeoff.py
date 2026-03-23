"""Code to reproduce the results in [1].

Note this is for guidance only, showing you how to access the classifiers and
datasets. Running it like this on a single machine will take a
very long time.


This creates a single result file for each combination of classifier and datasets
which stores all predictions and probability estimates. There is an example file in

Of course, you dont have to run it like this
[1]
"""
from aeon.datasets import load_classification
from aeon.datasets.tsc_datasets import eeg
from aeon.classification.hybrid import HIVECOTEV2
from aeon_neuro import classifiers
from tsml_eval import

classifiers = []
datasets = eeg


def simple_experiment():




if __name__ == "__main__":
    for d in datasets:
        for cls in classifiers:
            single_experiment(d, cls)


