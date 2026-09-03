"""Tests for the leaderboard tables.

These build a small results directory from known numbers, so the expected
ranking can be worked out by hand and the tests need no archive data.
"""

import numpy as np
import pandas as pd
import pytest

from multiverse.experiments.tables import (
    LOWER_IS_BETTER,
    METRIC_LABELS,
    available_estimators,
    dataset_markdown,
    dataset_page,
    dataset_summary,
    leaderboard,
    leaderboard_markdown,
    load_metric,
    load_missing_reasons,
    write_markdown_table,
)

# Alice is best on every metric, Bob second, Carol worst, so the expected order
# is the same whichever metric is used to sort. Carol is missing "d3", which is
# what makes it the dataset the leaderboard has to drop.
SCORES = {
    "Alice": {"d1": 0.9, "d2": 0.8, "d3": 0.7},
    "Bob": {"d1": 0.6, "d2": 0.5, "d3": 0.4},
    "Carol": {"d1": 0.3, "d2": 0.2},
}


@pytest.fixture
def results_dir(tmp_path):
    """A results directory holding SCORES for every metric."""
    for estimator, values in SCORES.items():
        directory = tmp_path / estimator
        directory.mkdir()
        for metric in METRIC_LABELS:
            # log loss is lower-is-better, so invert it and expect the same order
            series = pd.Series(
                {
                    dataset: (1 - score if metric in LOWER_IS_BETTER else score)
                    for dataset, score in values.items()
                }
            )
            series.index.name = "Resamples:"
            series.name = "0"
            series.to_csv(directory / f"{estimator}_{metric}.csv")
    return tmp_path


def test_available_estimators(results_dir):
    """Estimators are found by directory, and can be excluded."""
    assert available_estimators(results_dir) == ["Alice", "Bob", "Carol"]
    assert available_estimators(results_dir, exclude=("Bob",)) == ["Alice", "Carol"]


def test_load_metric(results_dir):
    """One estimator's scores load as a named series."""
    series = load_metric("Alice", "accuracy", results_dir)
    assert series.name == "Alice"
    assert series["d1"] == pytest.approx(0.9)
    assert len(series) == 3


def test_load_metric_missing_file(results_dir):
    """A metric an estimator does not have is an error, not a silent gap."""
    with pytest.raises(FileNotFoundError):
        load_metric("Alice", "nosuchmetric", results_dir)


def test_only_complete_datasets_are_used(results_dir, tmp_path):
    """Carol lacks d3, so every estimator is scored on d1 and d2 only."""
    page = leaderboard(
        ["d1", "d2", "d3"],
        ["Alice", "Bob", "Carol"],
        results_dir=results_dir,
        output_path=tmp_path / "out.html",
    ).read_text(encoding="utf-8")

    assert "3 estimators on 2 datasets" in page
    # Alice's accuracy is the mean of d1 and d2, not of all three
    assert f"{np.mean([0.9, 0.8]):.4f}" in page
    assert f"{np.mean([0.9, 0.8, 0.7]):.4f}" not in page


def test_dropping_the_incomplete_estimator_restores_the_dataset(results_dir, tmp_path):
    """Without Carol, d3 comes back."""
    page = leaderboard(
        ["d1", "d2", "d3"],
        ["Alice", "Bob"],
        results_dir=results_dir,
        output_path=tmp_path / "out.html",
    ).read_text(encoding="utf-8")
    assert "2 estimators on 3 datasets" in page


def test_rows_are_ordered_by_rank(results_dir, tmp_path):
    """Alice beats Bob beats Carol, and the table says so in that order."""
    page = leaderboard(
        ["d1", "d2", "d3"],
        ["Carol", "Alice", "Bob"],
        results_dir=results_dir,
        output_path=tmp_path / "out.html",
    ).read_text(encoding="utf-8")
    assert page.index("Alice") < page.index("Bob") < page.index("Carol")


def test_lower_is_better_metric_ranks_ascending(results_dir, tmp_path):
    """Sorting on log loss gives the same order, because it is inverted."""
    page = leaderboard(
        ["d1", "d2", "d3"],
        ["Alice", "Bob", "Carol"],
        sort_by="logloss",
        results_dir=results_dir,
        output_path=tmp_path / "out.html",
    ).read_text(encoding="utf-8")
    assert "lower" in page.lower()
    assert page.index("Alice") < page.index("Bob") < page.index("Carol")


def test_missing_results_are_reported(results_dir, tmp_path):
    """The dropped dataset is named, against the estimator that lacks it."""
    page = leaderboard(
        ["d1", "d2", "d3"],
        ["Alice", "Bob", "Carol"],
        results_dir=results_dir,
        output_path=tmp_path / "out.html",
    ).read_text(encoding="utf-8")
    assert "Missing results" in page
    assert "Carol" in page and "d3" in page


def test_missing_reason_is_used_when_recorded(results_dir, tmp_path):
    """A reason in missing_results.csv appears instead of "not recorded"."""
    pd.DataFrame(
        [{"estimator": "Carol", "dataset": "d3", "reason": "ran out of memory",
          "detail": "", "source": ""}]
    ).to_csv(results_dir / "missing_results.csv", index=False)

    page = leaderboard(
        ["d1", "d2", "d3"],
        ["Alice", "Bob", "Carol"],
        results_dir=results_dir,
        output_path=tmp_path / "out.html",
    ).read_text(encoding="utf-8")
    assert "ran out of memory" in page
    assert "not recorded" not in page


def test_missing_reasons_absent_file(tmp_path):
    """No reasons file is not an error."""
    frame = load_missing_reasons(tmp_path)
    assert frame.empty
    assert list(frame.columns) == ["estimator", "dataset", "reason", "detail"]


def test_no_overlapping_datasets_raises(results_dir, tmp_path):
    """Asking for datasets nobody has is an error, not an empty page."""
    with pytest.raises(ValueError):
        leaderboard(
            ["nope"],
            ["Alice", "Bob"],
            results_dir=results_dir,
            output_path=tmp_path / "out.html",
        )


def test_unknown_sort_metric_raises(results_dir, tmp_path):
    with pytest.raises(ValueError):
        leaderboard(
            ["d1", "d2"],
            ["Alice", "Bob"],
            sort_by="nosuchmetric",
            results_dir=results_dir,
            output_path=tmp_path / "out.html",
        )


def test_page_is_self_contained(results_dir, tmp_path):
    """Nothing is fetched from outside, so the page renders anywhere."""
    page = leaderboard(
        ["d1", "d2"],
        ["Alice", "Bob"],
        results_dir=results_dir,
        output_path=tmp_path / "out.html",
    ).read_text(encoding="utf-8")
    assert "http://" not in page
    assert 'src="http' not in page and "@import" not in page


def test_markdown_table(results_dir):
    """The Markdown view has a row per estimator and orders them the same way."""
    table = leaderboard_markdown(
        ["d1", "d2", "d3"], ["Alice", "Bob", "Carol"], results_dir=results_dir
    )
    rows = [line for line in table.splitlines() if line.startswith("|")]
    assert len(rows) == 2 + 3  # header, separator, three estimators
    assert {len(row.split("|")) for row in rows} == {len(rows[0].split("|"))}
    assert table.index("Alice") < table.index("Bob") < table.index("Carol")
    assert "2 Multiverse-core datasets" in table


def test_write_markdown_table(tmp_path):
    """The marked block is replaced, and the rest of the file is untouched."""
    path = tmp_path / "README.md"
    path.write_text("before\n<!-- T:START -->\nold\n<!-- T:END -->\nafter\n", encoding="utf-8")

    assert write_markdown_table(path, "new", marker="T")
    text = path.read_text(encoding="utf-8")
    assert "new" in text and "old" not in text
    assert text.startswith("before") and text.rstrip().endswith("after")

    # replacing again is stable, so regenerating does not churn the file
    write_markdown_table(path, "new", marker="T")
    assert path.read_text(encoding="utf-8") == text


def test_write_markdown_table_without_markers(tmp_path):
    """A file with no markers is reported, not silently rewritten."""
    path = tmp_path / "README.md"
    path.write_text("no markers here\n", encoding="utf-8")
    assert not write_markdown_table(path, "new", marker="T")
    assert path.read_text(encoding="utf-8") == "no markers here\n"


def test_dataset_summary_excludes_the_baseline(results_dir):
    """Best, median and spread ignore the baseline; gain is measured against it."""
    summary = dataset_summary(
        ["d1", "d2", "d3"], ["Alice", "Bob", "Carol"], baseline="Carol",
        results_dir=results_dir,
    )
    # d1: Alice 0.9, Bob 0.6, baseline Carol 0.3
    assert summary.at["d1", "best"] == pytest.approx(0.9)
    assert summary.at["d1", "best_estimator"] == "Alice"
    assert summary.at["d1", "worst"] == pytest.approx(0.6)
    assert summary.at["d1", "baseline"] == pytest.approx(0.3)
    assert summary.at["d1", "gain"] == pytest.approx(0.6)
    assert summary.at["d1", "spread"] == pytest.approx(0.3)
    assert summary.at["d1", "estimators"] == 2


def test_dataset_summary_keeps_partly_covered_datasets(results_dir):
    """A dataset only some estimators finished is kept, with a lower count.

    This is where it differs from the leaderboard, which has to drop d3 to keep
    the averages comparable.
    """
    summary = dataset_summary(
        ["d1", "d2", "d3"], ["Alice", "Bob", "Carol"], results_dir=results_dir
    )
    assert list(summary.index) == ["d1", "d2", "d3"]
    # Carol has no d3, and is not the baseline here, so only two contribute
    assert summary.at["d3", "estimators"] == 2
    assert summary.at["d1", "estimators"] == 3


def test_dataset_summary_lower_is_better_flips_best_and_gain(results_dir):
    """For log loss the best score is the smallest, and gain stays positive."""
    summary = dataset_summary(
        ["d1"], ["Alice", "Bob", "Carol"], metric="logloss", baseline="Carol",
        results_dir=results_dir,
    )
    # stored as 1 - score, so Alice 0.1, Bob 0.4, Carol 0.7
    assert summary.at["d1", "best"] == pytest.approx(0.1)
    assert summary.at["d1", "best_estimator"] == "Alice"
    assert summary.at["d1", "gain"] == pytest.approx(0.6)
    assert summary.at["d1", "spread"] == pytest.approx(0.3)


def test_dataset_page_is_written_and_sorted_by_gain(results_dir, tmp_path):
    """The page lists every dataset, worst gain first."""
    path = dataset_page(
        ["d1", "d2", "d3"], ["Alice", "Bob", "Carol"], baseline="Carol",
        results_dir=results_dir, output_path=tmp_path / "datasets.html",
    )
    html = path.read_text(encoding="utf-8")
    for dataset in ["d1", "d2", "d3"]:
        assert f"<td>{dataset}</td>" in html
    # d3 has no baseline score, so its gain is NaN and it sorts last
    assert html.index("<td>d2</td>") < html.index("<td>d3</td>")
    assert "Gain over dummy" in html


def test_dataset_markdown_marks_the_best(results_dir):
    """The Markdown table bolds the best score and names the estimator."""
    table = dataset_markdown(
        ["d1"], ["Alice", "Bob", "Carol"], baseline="Carol", results_dir=results_dir
    )
    assert "| d1 |" in table
    assert "**0.9000**" in table
    assert "Alice" in table
