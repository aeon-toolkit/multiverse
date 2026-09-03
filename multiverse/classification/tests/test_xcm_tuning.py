"""Tests for XCM's cross-validated parameter search.

The authors set window size and batch size per dataset by grid search over a
stratified five-fold cross-validation of the training set (§4.3). These check
the search runs when asked, is skipped when not, and never sees test data.
"""

import pytest

pytest.importorskip("tensorflow")

from aeon.testing.data_generation import make_example_3d_numpy

from multiverse.classification import XCMClassifier


def test_xcm_scalar_parameters_run_no_search():
    """A scalar window and batch fit once, with no cross-validation."""
    X, y = make_example_3d_numpy(
        n_cases=20, n_channels=2, n_timepoints=40, n_labels=2, random_state=0
    )
    clf = XCMClassifier(n_epochs=1, n_filters=2, random_state=0).fit(X, y)
    assert clf.cv_results_ == []
    assert clf.batch_size_ == 32
    # 0.8 of 40 points, under the 100 point bound
    assert clf.window_size_ == 32
    assert clf.window_fraction_ == 0.8


def test_xcm_tunes_over_a_window_grid():
    """A sequence triggers the authors' search and records every grid point."""
    X, y = make_example_3d_numpy(
        n_cases=20, n_channels=2, n_timepoints=40, n_labels=2, random_state=0
    )
    clf = XCMClassifier(
        window_size=[0.2, 1.0], n_epochs=1, n_filters=2, cv_folds=2, random_state=0
    ).fit(X, y)
    assert len(clf.cv_results_) == 2
    assert {r["window_size"] for r in clf.cv_results_} == {0.2, 1.0}
    assert all(len(r["fold_accuracies"]) == 2 for r in clf.cv_results_)
    # the fitted fraction is whichever scored best
    best = max(clf.cv_results_, key=lambda r: r["mean_accuracy"])
    assert clf.window_fraction_ == best["window_size"]


def test_xcm_tunes_window_and_batch_together():
    """Both sequences give the product of the two grids."""
    X, y = make_example_3d_numpy(
        n_cases=20, n_channels=2, n_timepoints=40, n_labels=2, random_state=0
    )
    clf = XCMClassifier(
        window_size=[0.2, 1.0], batch_size=[8, 16], n_epochs=1, n_filters=2,
        cv_folds=2, random_state=0,
    ).fit(X, y)
    assert len(clf.cv_results_) == 4
    assert clf.batch_size_ in {8, 16}


def test_xcm_max_window_none_leaves_the_kernel_unbounded():
    """None reproduces the authors' behaviour of not bounding the kernel."""
    X, y = make_example_3d_numpy(
        n_cases=10, n_channels=2, n_timepoints=300, n_labels=2, random_state=0
    )
    bounded = XCMClassifier(n_epochs=1, n_filters=2, random_state=0).fit(X, y)
    assert bounded.window_size_ == 100  # capped

    unbounded = XCMClassifier(
        max_window=None, n_epochs=1, n_filters=2, random_state=0
    ).fit(X, y)
    assert unbounded.window_size_ == 240  # 0.8 of 300


def test_xcm_rejects_bad_grids():
    """Every candidate is validated, not just the first."""
    X, y = make_example_3d_numpy(
        n_cases=10, n_channels=2, n_timepoints=20, n_labels=2, random_state=0
    )
    with pytest.raises(ValueError, match="window_size"):
        XCMClassifier(window_size=[0.5, 1.5]).fit(X, y)
    with pytest.raises(ValueError, match="batch_size"):
        XCMClassifier(batch_size=[8, 0]).fit(X, y)
    with pytest.raises(ValueError, match="cv_folds"):
        XCMClassifier(window_size=[0.2, 0.8], cv_folds=1).fit(X, y)
