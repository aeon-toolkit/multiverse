"""Run the EEG classification archive with tsml-eval.

The EEG archive is a sub-collection of the Multiverse archive, 28 datasets,
available as eeg2026 in aeon. It is kept separate because EEG problems have
their own conventions and because aeon-neuro provides classifiers built for
them.

This is the same loop as run_benchmark, fixed to the EEG dataset list. Note it
is for guidance only. Running every classifier over the archive on one machine
will take a very long time.

Results are written per classifier and dataset, the same layout the rest of the
repository reads

    <results_path>/<classifier>/Predictions/<dataset>/testResample<id>.csv

Edit the settings in main and run it with

    python -m multiverse.experiments.run_eeg_bakeoff
"""

from aeon.datasets.tsc_datasets import eeg2026

from multiverse.experiments.run_single_dataset import single_experiment


def main():
    classifiers = ["HC2"]
    datasets = sorted(eeg2026)

    print(f"EEG archive: {len(datasets)} datasets, {len(classifiers)} classifiers")
    for classifier in classifiers:
        for dataset in datasets:
            try:
                single_experiment(dataset, classifier)
                print(f"  done   {classifier} {dataset}")
            except Exception as e:
                print(f"  failed {classifier} {dataset}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
