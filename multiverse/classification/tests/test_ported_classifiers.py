"""Tests for the classifiers ported into multiverse.

These check the aeon interface contract rather than the published accuracy of
the networks: the correct input layout, valid probabilities, and reproducibility
on CPU for a fixed seed.
"""

import numpy as np
import pytest

pytest.importorskip("torch")

from aeon.testing.data_generation import make_example_3d_numpy

from multiverse.classification import (
    ConvTranClassifier,
    PatchMTSCClassifier,
    TimesNetClassifier,
)
from multiverse.classification._convtran import _ConvTranNetwork
from multiverse.classification._patchmtsc import _PatchMTSCNetwork

SMALL_PARAMS = {
    ConvTranClassifier: {
        "emb_size": 8,
        "dim_ff": 16,
        "num_heads": 2,
        "n_epochs": 2,
        "batch_size": 4,
        "validation_size": 0.25,
        "device": "cpu",
        "random_state": 0,
    },
    PatchMTSCClassifier: {
        "emb_size": 8,
        "d_model_patch": 8,
        "dim_ff": 16,
        "num_heads": 2,
        "patch_len": 4,
        "stride": 2,
        "n_epochs": 2,
        "batch_size": 4,
        "validation_size": 0.25,
        "device": "cpu",
        "random_state": 0,
    },
    TimesNetClassifier: {
        "e_layers": 1,
        "d_model": 16,
        "d_ff": 16,
        "top_k": 2,
        "num_kernels": 2,
        "n_epochs": 2,
        "batch_size": 4,
        "device": "cpu",
        "random_state": 0,
    },
}


@pytest.mark.parametrize("classifier_class", list(SMALL_PARAMS))
def test_fit_predict_proba(classifier_class):
    """Fit with channels != timepoints and return valid probabilities."""
    X, y = make_example_3d_numpy(
        n_cases=20, n_channels=3, n_timepoints=12, n_labels=2, random_state=0
    )
    y = np.where(y == 0, "class_a", "class_b")

    classifier = classifier_class(**SMALL_PARAMS[classifier_class])
    classifier.fit(X, y)
    probabilities = classifier.predict_proba(X[:5])

    assert probabilities.shape == (5, 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    assert set(classifier.predict(X[:5])).issubset(set(y))


@pytest.mark.parametrize("classifier_class", list(SMALL_PARAMS))
def test_repeatable_on_cpu(classifier_class):
    """The same seed should reproduce CPU probabilities."""
    X, y = make_example_3d_numpy(
        n_cases=16, n_channels=2, n_timepoints=10, n_labels=2, random_state=1
    )
    params = dict(SMALL_PARAMS[classifier_class], n_epochs=1)

    first = classifier_class(**params).fit(X, y).predict_proba(X)
    second = classifier_class(**params).fit(X, y).predict_proba(X)

    np.testing.assert_allclose(first, second, atol=1e-7)


@pytest.mark.parametrize(
    "network_class,kwargs",
    [
        (
            _ConvTranNetwork,
            {"emb_size": 8, "num_heads": 2, "dim_ff": 16, "dropout": 0.01},
        ),
        (
            _PatchMTSCNetwork,
            {
                "emb_size": 8,
                "num_heads": 2,
                "dim_ff": 16,
                "d_model_patch": 8,
                "patch_len": 4,
                "stride": 2,
                "dropout": 0.01,
                "head_dropout": 0.0,
                "pap_dropout": 0.0,
                "graph_decay": 5e-4,
            },
        ),
    ],
)
def test_network_rejects_transposed_aeon_input(network_class, kwargs):
    """Guard against silently swapping the channel and time axes."""
    network = network_class(
        n_channels=3, n_timepoints=12, n_classes=2, **kwargs
    )

    with pytest.raises(ValueError, match="input shape changed"):
        network(np.zeros((4, 12, 3), dtype=np.float32))


def test_patchmtsc_clamps_patch_geometry_to_short_series():
    """A series shorter than patch_len should still fit."""
    X, y = make_example_3d_numpy(
        n_cases=12, n_channels=2, n_timepoints=6, n_labels=2, random_state=2
    )
    classifier = PatchMTSCClassifier(
        **dict(SMALL_PARAMS[PatchMTSCClassifier], patch_len=16, stride=8)
    )
    classifier.fit(X, y)

    assert classifier.patch_len_ == 6
    assert classifier.stride_ == 6
    assert classifier.predict_proba(X).shape == (12, 2)


@pytest.mark.parametrize("classifier_class", list(SMALL_PARAMS))
def test_get_test_params(classifier_class):
    """Test parameters should construct a usable estimator."""
    params = classifier_class._get_test_params()

    assert isinstance(params, dict)
    assert isinstance(classifier_class(**params), classifier_class)


def test_timesnet_learning_rate_schedule():
    """The TSLib ``type1`` schedule fires every five epochs and halves each time."""
    import torch

    classifier = TimesNetClassifier(learning_rate=1e-3, n_epochs=30)
    parameter = torch.nn.Parameter(torch.zeros(1))

    for epoch, expected in [
        (1, None),
        (4, None),
        (5, 1e-3 * 0.5**4),
        (6, None),
        (10, 1e-3 * 0.5**9),
        (15, 1e-3 * 0.5**14),
    ]:
        optimiser = torch.optim.SGD([parameter], lr=1e-3)
        assert classifier._adjust_learning_rate(optimiser, epoch) == expected
        if expected is not None:
            assert optimiser.param_groups[0]["lr"] == expected


def test_timesnet_learning_rate_schedule_can_be_disabled():
    """``lr_adjust=None`` leaves the rate untouched at every epoch."""
    import torch

    classifier = TimesNetClassifier(learning_rate=1e-3, n_epochs=30, lr_adjust=None)
    optimiser = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)

    for epoch in range(1, 31):
        assert classifier._adjust_learning_rate(optimiser, epoch) is None
    assert optimiser.param_groups[0]["lr"] == 1e-3


def test_timesnet_schedule_changes_training():
    """The schedule is not a no-op over a run long enough to reach epoch five."""
    X, y = make_example_3d_numpy(
        n_cases=40, n_channels=3, n_timepoints=20, n_labels=3, random_state=3
    )
    params = dict(SMALL_PARAMS[TimesNetClassifier], n_epochs=8)

    scheduled = TimesNetClassifier(**params, lr_adjust="type1").fit(X, y)
    constant = TimesNetClassifier(**params, lr_adjust=None).fit(X, y)

    assert not np.allclose(
        scheduled.predict_proba(X), constant.predict_proba(X), atol=1e-6
    )


@pytest.mark.parametrize("classifier_class", list(SMALL_PARAMS))
def test_no_fitted_attributes_before_fit(classifier_class):
    """Fitted attributes must not exist until ``fit`` has run."""
    classifier = classifier_class(**SMALL_PARAMS[classifier_class])

    for attribute in ["history_", "device_", "n_channels_"]:
        assert not hasattr(classifier, attribute), (
            f"{attribute} exists before fit, so is-fitted checks misreport"
        )


@pytest.mark.parametrize(
    "params,message",
    [
        ({"e_layers": 0}, "e_layers must be a positive integer"),
        ({"dropout": 1.0}, "dropout must be in"),
        ({"validation_size": 1.5}, "validation_size must be in"),
        ({"lr_adjust": "type9"}, "lr_adjust must be one of"),
    ],
)
def test_timesnet_rejects_invalid_parameters(params, message):
    """Invalid parameters are rejected at the start of fit."""
    X, y = make_example_3d_numpy(
        n_cases=12, n_channels=2, n_timepoints=12, n_labels=2, random_state=0
    )
    classifier = TimesNetClassifier(**dict(SMALL_PARAMS[TimesNetClassifier], **params))

    with pytest.raises(ValueError, match=message):
        classifier.fit(X, y)
