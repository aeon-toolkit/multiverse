"""Run the EEG classification archive with ``tsml_eval``.

The EEG archive is a sub-collection of the Multiverse archive, described in [1]_
and available as ``eeg2026`` in aeon. It is a separate benchmark because EEG
problems have their own conventions, and because ``aeon-neuro`` provides
classifiers built for them.

This is the same loop as :mod:`multiverse.experiments.run_benchmark`, fixed to
the EEG dataset list. It is for guidance: running every classifier over the
archive on one machine takes a very long time, and the published results were
produced on a cluster.

Run it with::

    python -m multiverse.experiments.run_eeg_bakeoff \\
        --data-path /path/to/Multiverse \\
        --results-path ./results-eeg \\
        --classifiers HC2 ROCKET

Results are written per classifier and dataset, the same layout the rest of the
repository reads::

    <results_path>/<classifier>/Predictions/<dataset>/testResample<id>.csv

.. [1] The EEG archive is documented at
   https://github.com/aeon-toolkit/aeon-neuro
"""

from __future__ import annotations

import argparse

from multiverse.experiments.run_benchmark import run_benchmark


def eeg_datasets() -> list[str]:
    """Return the EEG archive dataset names.

    The collection is ``eeg2026`` in aeon. It was named ``eeg`` in earlier
    versions, which is why an older form of this script failed to import.
    """
    from aeon.datasets.tsc_datasets import eeg2026

    return sorted(eeg2026)


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-path", required=True, help="archive directory")
    parser.add_argument("--results-path", required=True, help="where to write results")
    parser.add_argument(
        "--classifiers",
        nargs="+",
        default=["HC2"],
        help="classifier names to run (default: HC2)",
    )
    parser.add_argument("--resample", type=int, default=0, help="resample id")
    parser.add_argument("--random-state", type=int, default=0, help="classifier seed")
    parser.add_argument(
        "--overwrite", action="store_true", help="rerun combinations that have results"
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    """Run the EEG bake off from the command line."""
    args = _parse_args(argv)
    datasets = eeg_datasets()
    print(f"EEG archive: {len(datasets)} datasets, {len(args.classifiers)} classifier(s)")

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


if __name__ == "__main__":
    main()
