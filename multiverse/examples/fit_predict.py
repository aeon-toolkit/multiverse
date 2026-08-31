"""Fit and predict with the classifiers ported into this package.

The classifiers in multiverse.classification follow the aeon interface, so they
are used exactly like any aeon classifier: fit on a 3D numpy array of shape
(n_cases, n_channels, n_timepoints), then predict or predict_proba.

The settings here are deliberately small so the example finishes quickly. The
defaults are the ones the authors of each method published, and are what the
benchmark runs use.

Run it with

    python -m multiverse.examples.fit_predict
"""

from aeon.classification.convolution_based import RocketClassifier
from aeon.datasets import load_classification

from multiverse.classification import TimesNetClassifier, TimesURLClassifier


def main():
    dataset = "BasicMotions"

    X_train, y_train = load_classification(dataset, split="train")
    X_test, y_test = load_classification(dataset, split="test")
    print(f"{dataset}: train {X_train.shape}, test {X_test.shape}")
    print(f"classes: {sorted(set(y_train))}")

    classifiers = [
        RocketClassifier(n_kernels=500, random_state=0),
        TimesNetClassifier(n_epochs=10, device="cpu", random_state=0),
        TimesURLClassifier(n_iters=20, device="cpu", random_state=0),
    ]

    for classifier in classifiers:
        classifier.fit(X_train, y_train)
        predictions = classifier.predict(X_test)
        probabilities = classifier.predict_proba(X_test)
        accuracy = (predictions == y_test).mean()
        print(
            f"  {type(classifier).__name__:22s} accuracy={accuracy:.4f} "
            f"probabilities={probabilities.shape}"
        )


if __name__ == "__main__":
    main()
