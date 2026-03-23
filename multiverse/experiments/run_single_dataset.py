from __future__ import annotations

import argparse
from aeon.classification.interval_based import TimeSeriesForestClassifier
from aeon.datasets import load_classificiation


def single_experiment():
    dataset = "ChinaTown"

    X_train, y_train = load_classificiation(dataset, split="train")
    X_test, y_test = load_classificiation(dataset, split= "test")

    clf = TimeSeriesForestClassifier(n_estimators=20, random_state=0)
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    print(f"{args.dataset} accuracy={acc:.4f}")


if __name__ == "__main__":
    single_experiment()
