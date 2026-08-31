"""Run a set of classifiers over a set of Multiverse datasets with ``tsml_eval``.

This is the loop that produced the results in ``results/``, reduced to its
essentials. It is sequential and single machine, so running the full
Multiverse-core benchmark this way will take a very long time: the published
runs were distributed over a cluster, with per-job memory and time limits. Treat
this as the definition of the experiment rather than as the way to execute it.

Every combination writes one file::

    <results_path>/<classifier>/Predictions/<dataset>/testResample<id>.csv

Existing files are skipped unless ``--overwrite`` is given, so an interrupted
run can simply be started again.

Run it with::

    python -m multiverse.experiments.run_benchmark \\
        --data-path /path/to/Multiverse \\
        --results-path ./results-raw \\
        --classifiers ROCKET DrCIF ConvTran

By default it runs over Multiverse-core, the 66 dataset benchmarking subset.
Use ``--datasets`` to name datasets explicitly, or ``--dataset-list`` to pick
another aeon collection such as ``multiverse2026`` or ``eeg2026``.
"""

from __future__ import annotations

import argparse
import time
import traceback
from pathlib import Path

from multiverse.experiments.run_single_dataset import single_experiment


def _resolve_datasets(name: str) -> list[str]:
    """Return a named aeon dataset collection, sorted."""
    from aeon.datasets import tsc_datasets

    collection = getattr(tsc_datasets, name, None)
    if collection is None:
        available = [
            n
            for n in dir(tsc_datasets)
            if not n.startswith("_") and isinstance(getattr(tsc_datasets, n), (list, set, tuple))
        ]
        raise ValueError(
            f"unknown dataset list {name!r}; available: {', '.join(sorted(available))}"
        )
    return sorted(collection)


def run_benchmark(
    data_path: str | Path,
    results_path: str | Path,
    classifiers,
    datasets,
    resample_id: int = 0,
    random_state: int | None = 0,
    overwrite: bool = False,
    verbose: bool = True,
) -> list[dict]:
    """Run every classifier on every dataset, reporting what happened.

    A failure on one combination is recorded and the run continues, because on a
    benchmark of this size some combinations will always fail on memory or time,
    and losing the rest of the run to one exception is not useful.

    Returns
    -------
    list of dict
        One entry per combination, with ``classifier``, ``dataset``, ``status``
        (``"done"``, ``"skipped"`` or ``"failed"``), ``seconds`` and, for
        failures, ``error``.
    """
    data_path, results_path = Path(data_path), Path(results_path)
    log = []

    for classifier_name in classifiers:
        for dataset in datasets:
            expected = (
                results_path
                / classifier_name
                / "Predictions"
                / dataset
                / f"testResample{resample_id}.csv"
            )
            if expected.is_file() and not overwrite:
                log.append(
                    {
                        "classifier": classifier_name,
                        "dataset": dataset,
                        "status": "skipped",
                        "seconds": 0.0,
                    }
                )
                continue

            start = time.time()
            try:
                single_experiment(
                    data_path=data_path,
                    results_path=results_path,
                    classifier_name=classifier_name,
                    dataset=dataset,
                    resample_id=resample_id,
                    random_state=random_state,
                    overwrite=overwrite,
                )
                entry = {
                    "classifier": classifier_name,
                    "dataset": dataset,
                    "status": "done",
                    "seconds": time.time() - start,
                }
            except Exception as error:  # noqa: BLE001 - one failure must not end the run
                entry = {
                    "classifier": classifier_name,
                    "dataset": dataset,
                    "status": "failed",
                    "seconds": time.time() - start,
                    "error": f"{type(error).__name__}: {error}",
                }
                if verbose:
                    traceback.print_exc()
            log.append(entry)

            if verbose:
                print(
                    f"  {entry['status']:8s} {classifier_name:16s} {dataset:28s} "
                    f"{entry['seconds']:8.1f}s"
                    + (f"  {entry.get('error', '')}" if entry["status"] == "failed" else "")
                )
    return log


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-path", required=True, help="archive directory")
    parser.add_argument("--results-path", required=True, help="where to write results")
    parser.add_argument(
        "--classifiers", nargs="+", required=True, help="classifier names to run"
    )
    parser.add_argument(
        "--datasets", nargs="+", default=None, help="datasets, overriding --dataset-list"
    )
    parser.add_argument(
        "--dataset-list",
        default="multiverse_core",
        help="named aeon collection to run over (default: multiverse_core)",
    )
    parser.add_argument("--resample", type=int, default=0, help="resample id")
    parser.add_argument("--random-state", type=int, default=0, help="classifier seed")
    parser.add_argument(
        "--overwrite", action="store_true", help="rerun combinations that have results"
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    """Run the benchmark from the command line and print a summary."""
    args = _parse_args(argv)
    datasets = args.datasets or _resolve_datasets(args.dataset_list)
    print(
        f"{len(args.classifiers)} classifier(s) x {len(datasets)} dataset(s) "
        f"= {len(args.classifiers) * len(datasets)} runs"
    )

    log = run_benchmark(
        data_path=args.data_path,
        results_path=args.results_path,
        classifiers=args.classifiers,
        datasets=datasets,
        resample_id=args.resample,
        random_state=args.random_state,
        overwrite=args.overwrite,
    )

    counts = {}
    for entry in log:
        counts[entry["status"]] = counts.get(entry["status"], 0) + 1
    print("\n" + ", ".join(f"{status}: {n}" for status, n in sorted(counts.items())))
    for entry in log:
        if entry["status"] == "failed":
            print(f"  failed  {entry['classifier']} / {entry['dataset']}: {entry['error']}")


if __name__ == "__main__":
    main()
