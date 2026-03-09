"""
aeon is a scikit learn compatible toolkit for time series machine learning.
It has a

"""
from __future__ import annotations

from aeon.classification.interval_based import TimeSeriesForestClassifier
from aeon.datasets import multiverse_mini


def main():
    print("First datasets:", list_datasets()[:10])

    X_train, y_train = load_classification("BasicMotions", split="train")
    X_test, y_test = load_classification("BasicMotions", split="test")

    clf = (n_estimators=200, random_state=0)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)

    acc = clf.score(X_test, y_test)
    print(f" accuracy={acc:.4f}")


if __name__ == "__main__":
    main()
