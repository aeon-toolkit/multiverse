"""Run one classifier on one Multiverse dataset with tsml-eval.

This is the smallest complete example of generating results in the format the
rest of this repository reads. tsml-eval writes one file per classifier,
dataset and resample

    <results_path>/<classifier>/Predictions/<dataset>/testResample<id>.csv

containing the class labels, the predictions, the probability estimates and the
fit and predict times. Those files are the input to
multiverse.experiments.tables, which turns them into a leaderboard.

Edit the settings in main and run it with

    python -m multiverse.experiments.run_single_dataset

The classifier name is anything tsml-eval can resolve, which includes the aeon
classifiers and the ones ported into this package.
"""

from tsml_eval.experiments import (
    get_classifier_by_name,
    load_and_run_classification_experiment,
)

data_path = "C:/Data/Multiverse"
results_path = "./results-raw"


def single_experiment(dataset, classifier_name, resample=0, random_state=0):
    """Run one classifier on one dataset and write the result file."""
    classifier = get_classifier_by_name(classifier_name, random_state=random_state)
    load_and_run_classification_experiment(
        problem_path=data_path,
        results_path=results_path,
        dataset=dataset,
        classifier=classifier,
        classifier_name=classifier_name,
        resample_id=resample,
        overwrite=False,
    )


def main():
    dataset = "BasicMotions"
    classifier = "ROCKET"

    single_experiment(dataset, classifier)
    print(f"{classifier} on {dataset} written to {results_path}")


if __name__ == "__main__":
    main()
