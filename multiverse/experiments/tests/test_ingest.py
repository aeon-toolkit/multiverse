"""Tests for turning raw prediction files into result files.

A prediction file is built with tsml-eval's own writer, so the test exercises
the real file format rather than a guess at it.
"""

import numpy as np
import pandas as pd
import pytest

import multiverse.experiments.ingest as ingest_module
from multiverse.experiments.ingest import ingest, metrics

pytest.importorskip("tsml_eval")


def write_predictions(path, classifier, dataset, labels, predictions, resample=0):
    """Write one prediction file in the layout the experiment scripts produce."""
    from tsml_eval.evaluation.storage import ClassifierResults

    n_classes = len(set(labels))
    probabilities = np.zeros((len(predictions), n_classes))
    probabilities[np.arange(len(predictions)), predictions] = 1.0

    results = ClassifierResults(
        dataset_name=dataset,
        classifier_name=classifier,
        split="TEST",
        resample_id=resample,
        n_classes=n_classes,
        class_labels=np.asarray(labels),
        predictions=np.asarray(predictions),
        probabilities=probabilities,
        fit_time=1.0,
        predict_time=1.0,
    )
    directory = path / classifier / "Predictions" / dataset
    directory.mkdir(parents=True, exist_ok=True)
    results.save_to_file(str(directory) + "/")
    return directory / f"testResample{resample}.csv"


@pytest.fixture
def predictions(tmp_path):
    """Two datasets for Alice: one predicted perfectly, one entirely wrong."""
    write_predictions(tmp_path, "Alice", "d1", [0, 0, 1, 1], [0, 0, 1, 1])
    write_predictions(tmp_path, "Alice", "d2", [0, 0, 1, 1], [1, 1, 0, 0])
    return tmp_path


def test_ingest_writes_one_file_per_metric(predictions, tmp_path, monkeypatch):
    out = tmp_path / "out"
    monkeypatch.setattr(ingest_module, "results_path", out)

    found = ingest("Alice", predictions, datasets=["d1", "d2"])

    assert found == ["d1", "d2"]
    for metric in metrics:
        written = out / "Alice" / f"Alice_{metric}.csv"
        assert written.is_file()
        series = pd.read_csv(written, index_col=0).iloc[:, 0]
        assert list(series.index) == ["d1", "d2"]


def test_ingested_values_are_correct(predictions, tmp_path, monkeypatch):
    """A perfect run scores 1 and a fully wrong run scores 0."""
    out = tmp_path / "out"
    monkeypatch.setattr(ingest_module, "results_path", out)
    ingest("Alice", predictions, datasets=["d1", "d2"])

    accuracy = pd.read_csv(out / "Alice" / "Alice_accuracy.csv", index_col=0).iloc[:, 0]
    assert accuracy["d1"] == pytest.approx(1.0)
    assert accuracy["d2"] == pytest.approx(0.0)


def test_datasets_without_predictions_are_left_out(predictions, tmp_path, monkeypatch):
    """A dataset with no prediction file is missing, not filled in."""
    out = tmp_path / "out"
    monkeypatch.setattr(ingest_module, "results_path", out)

    found = ingest("Alice", predictions, datasets=["d1", "d2", "d3"])

    assert found == ["d1", "d2"]
    accuracy = pd.read_csv(out / "Alice" / "Alice_accuracy.csv", index_col=0).iloc[:, 0]
    assert "d3" not in accuracy.index


def test_ingest_output_is_readable_by_the_leaderboard(predictions, tmp_path, monkeypatch):
    """What ingest writes is what load_metric reads."""
    from multiverse.experiments.tables import load_metric

    out = tmp_path / "out"
    monkeypatch.setattr(ingest_module, "results_path", out)
    ingest("Alice", predictions, datasets=["d1", "d2"])

    series = load_metric("Alice", "accuracy", out)
    assert series.name == "Alice"
    assert list(series.index) == ["d1", "d2"]
    assert series["d1"] == pytest.approx(1.0)
