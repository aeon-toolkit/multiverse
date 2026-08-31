"""Run one classifier on one Multiverse dataset with ``tsml_eval``.

This is the smallest complete example of generating results in the format the
rest of this repository reads. ``tsml_eval`` writes one file per
classifier/dataset/resample::

    <results_path>/<classifier>/Predictions/<dataset>/testResample<id>.csv

holding the true labels, the predictions, the probability estimates, and the
fit and predict times. Those files are the input to
:mod:`multiverse.experiments.tables`, which turns them into a leaderboard.

Run it with::

    python -m multiverse.experiments.run_single_dataset \\
        --data-path /path/to/Multiverse \\
        --results-path ./results-raw \\
        --classifier ConvTran \\
        --dataset BasicMotions

``--classifier`` accepts anything ``tsml_eval`` can resolve by name, which
includes the aeon classifiers and the ones ported in this package.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tsml_eval.experiments import (
    get_classifier_by_name,
    load_and_run_classification_experiment,
)


def single_experiment(
    data_path: str | Path,
    results_path: str | Path,
    classifier_name: str,
    dataset: str,
    resample_id: int = 0,
    random_state: int | None = 0,
    overwrite: bool = False,
) -> Path:
    """Run one classifier on one dataset and write the result file.

    Parameters
    ----------
    data_path : str or Path
        Directory holding the archive, one sub-directory per dataset, each with
        ``<dataset>_TRAIN.ts`` and ``<dataset>_TEST.ts``.
    results_path : str or Path
        Directory to write results into.
    classifier_name : str
        Name ``tsml_eval`` can resolve, for example ``"ConvTran"`` or ``"HC2"``.
    dataset : str
        Dataset name, matching its directory under ``data_path``.
    resample_id : int, default=0
        Resample to run. 0 is the default train/test split.
    random_state : int or None, default=0
        Seed passed to the classifier.
    overwrite : bool, default=False
        If False, an existing result file for this combination is left alone.
        Leaving this False is what makes a large run restartable.

    Returns
    -------
    Path
        The file that was written, or would have been.
    """
    classifier = get_classifier_by_name(classifier_name, random_state=random_state)
    load_and_run_classification_experiment(
        problem_path=str(data_path),
        results_path=str(results_path),
        dataset=dataset,
        classifier=classifier,
        classifier_name=classifier_name,
        resample_id=resample_id,
        overwrite=overwrite,
    )
    return (
        Path(results_path)
        / classifier_name
        / "Predictions"
        / dataset
        / f"testResample{resample_id}.csv"
    )


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-path", required=True, help="archive directory")
    parser.add_argument("--results-path", required=True, help="where to write results")
    parser.add_argument("--classifier", required=True, help="classifier name")
    parser.add_argument("--dataset", required=True, help="dataset name")
    parser.add_argument("--resample", type=int, default=0, help="resample id")
    parser.add_argument("--random-state", type=int, default=0, help="classifier seed")
    parser.add_argument(
        "--overwrite", action="store_true", help="rerun even if a result exists"
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    """Run one experiment from the command line."""
    args = _parse_args(argv)
    written = single_experiment(
        data_path=args.data_path,
        results_path=args.results_path,
        classifier_name=args.classifier,
        dataset=args.dataset,
        resample_id=args.resample,
        random_state=args.random_state,
        overwrite=args.overwrite,
    )
    print(f"wrote {written}")


if __name__ == "__main__":
    main()
