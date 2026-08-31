"""Run a set of classifiers over a set of Multiverse datasets with tsml-eval.

This is the loop that produced the results in results/, reduced to its
essentials. Note this is for guidance only, showing how to access the
classifiers and datasets. Running it like this on a single machine will take a
very long time; the published results were distributed over a cluster with per
job memory and time limits.

Every combination writes one file

    <results_path>/<classifier>/Predictions/<dataset>/testResample<id>.csv

Combinations that already have a result file are skipped, so an interrupted run
can be started again. A combination that fails is reported and the run
continues, because on a benchmark this size some will always run out of memory
or time.

Edit the settings in main and run it with

    python -m multiverse.experiments.run_benchmark
"""

from aeon.datasets.tsc_datasets import multiverse_core

from multiverse.experiments.run_single_dataset import single_experiment


def main():
    classifiers = ["ROCKET", "DrCIF", "ConvTran"]
    datasets = sorted(multiverse_core)

    print(f"{len(classifiers)} classifiers x {len(datasets)} datasets")
    for classifier in classifiers:
        for dataset in datasets:
            try:
                single_experiment(dataset, classifier)
                print(f"  done   {classifier} {dataset}")
            except Exception as e:
                print(f"  failed {classifier} {dataset}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
