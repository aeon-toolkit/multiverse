from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import pandas as pd
from aeon.classification.convolution_based import RocketClassifier

from aeon.datasets import load_classification
from aeon.datasets.tsc_datasets import multivariate_equal_length

def experiment_example():
    datasets = ["BasicMotions"]
    rows = []
    for name in datasets:
        X_train, y_train = load_classification(name, "train")
        X_test, y_test = load_classification(name, "test")

        clf = RocketClassifier(n_kernels=500, random_state=0)


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
    experiment_example()
