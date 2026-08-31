"""A minimal end-to-end example: load a Multiverse dataset and classify it.

``aeon`` is a scikit-learn compatible toolkit for time series machine learning.
It provides the dataset loaders used to fetch Multiverse problems from Zenodo,
and the classifiers benchmarked in this repository.

Run it with::

    python -m multiverse.examples.aeon_quickstart
"""

from __future__ import annotations

from aeon.classification.interval_based import TimeSeriesForestClassifier
from aeon.datasets import load_classification
from aeon.datasets.tsc_datasets import multiverse_core


def main() -> None:
    """Fit and score a classifier on one Multiverse-core dataset."""
    print(f"Multiverse-core has {len(multiverse_core)} datasets")
    print("first few:", sorted(multiverse_core)[:5])

    dataset = "BasicMotions"
    X_train, y_train = load_classification(dataset, split="train")
    X_test, y_test = load_classification(dataset, split="test")
    print(f"{dataset}: train {X_train.shape}, test {X_test.shape}")

    classifier = TimeSeriesForestClassifier(n_estimators=200, random_state=0)
    classifier.fit(X_train, y_train)

    predictions = classifier.predict(X_test)
    probabilities = classifier.predict_proba(X_test)
    print(f"predictions {predictions.shape}, probabilities {probabilities.shape}")
    print(f"{dataset} accuracy={classifier.score(X_test, y_test):.4f}")


if __name__ == "__main__":
    main()
