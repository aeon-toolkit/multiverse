from __future__ import annotations

import argparse
from aeon.classification.interval_based import TimeSeriesForestClassifier
from multiverse.datasets import load_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()

    X_train, y_train = load_dataset(args.dataset, "train")
    X_test, y_test = load_dataset(args.dataset, "test")

    clf = TimeSeriesForestClassifier(n_estimators=200, random_state=0)
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    print(f"{args.dataset} accuracy={acc:.4f}")


if __name__ == "__main__":
    main()
