from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import pandas as pd
from aeon.classification.interval_based import TimeSeriesForestClassifier

from multiverse.datasets import list_datasets, load_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="all", help="Comma-separated list, or 'all'")
    ap.add_argument("--out", default="benchmark_results.csv")
    args = ap.parse_args()

    if args.datasets.strip().lower() == "all":
        datasets = list_datasets()
    else:
        datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    rows = []
    for name in datasets:
        X_train, y_train = load_dataset(name, "train")
        X_test, y_test = load_dataset(name, "test")

        clf = TimeSeriesForestClassifier(n_estimators=200, random_state=0)

        t0 = time.time()
        clf.fit(X_train, y_train)
        fit_s = time.time() - t0

        t1 = time.time()
        acc = clf.score(X_test, y_test)
        score_s = time.time() - t1

        rows.append({
            "dataset": name,
            "metric": "accuracy",
            "score": float(acc),
            "split": "test",
            "fit_seconds": float(fit_s),
            "score_seconds": float(score_s),
        })

    df = pd.DataFrame(rows).sort_values(["dataset"])
    Path(args.out).write_text(df.to_csv(index=False), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
