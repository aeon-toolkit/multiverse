"""Tests for the RankSCL port.

The ranking loss is checked against a hand computation rather than only for
running, since it is the paper's contribution and a transcription of it is
exactly the kind of thing that can be subtly wrong while still training.
"""

import math

import numpy as np
import pytest

pytest.importorskip("torch")

import torch  # noqa: E402
from aeon.testing.data_generation import make_example_3d_numpy  # noqa: E402

from multiverse.classification import RankSCLClassifier  # noqa: E402
from multiverse.classification._rankscl import (  # noqa: E402
    _augment,
    _build_encoder,
    _ranking_loss,
    _same_class_neighbour,
)

SMALL = {
    "n_epochs": 2,
    "batch_size": 4,
    "aug_positives": 1,
    "probe": "logistic",
    "device": "cpu",
    "random_state": 0,
}


def _data(n_cases=20, n_channels=2, n_timepoints=40, n_labels=2):
    return make_example_3d_numpy(
        n_cases=n_cases,
        n_channels=n_channels,
        n_timepoints=n_timepoints,
        n_labels=n_labels,
        random_state=0,
    )


def test_ranking_loss_matches_a_hand_computation():
    """Four points on a line, so every distance can be worked out by hand."""
    embeddings = torch.tensor([[0.0], [1.0], [0.5], [3.0]])
    labels = torch.tensor([0, 0, 1, 1])

    def distance(i, j):
        return abs(embeddings[i, 0].item() - embeddings[j, 0].item())

    terms = []
    for anchor in range(4):
        negatives = [distance(anchor, n) for n in range(4) if labels[n] != labels[anchor]]
        for positive in range(4):
            if positive == anchor or labels[positive] != labels[anchor]:
                continue
            gap = distance(anchor, positive)
            closer = [n for n in negatives if n <= gap]
            terms.append(sum(1 / (1 + math.exp(-(gap - c))) for c in closer))
    expected = sum(math.atan(t) for t in terms) / len(terms)

    assert float(_ranking_loss(embeddings, labels, "EU")) == pytest.approx(expected)


def _loop_ranking_loss(embeddings, labels, distance):
    """The authors' loop, kept as the reference the vectorised form must match."""
    if distance == "Cosine":
        matrix = -torch.cosine_similarity(
            embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2
        )
    else:
        matrix = torch.cdist(embeddings, embeddings, p=2)
    same = labels.reshape(1, -1) == labels.reshape(-1, 1)
    violations = []
    for anchor in range(matrix.shape[0]):
        negatives = matrix[anchor][~same[anchor]]
        if negatives.numel() == 0:
            continue
        positives = same[anchor].nonzero().flatten()
        for positive in positives[positives != anchor]:
            gap = matrix[anchor, positive]
            closer = negatives[negatives <= gap]
            violations.append(torch.sigmoid(gap - closer).sum())
    if not violations:
        return None
    return torch.atan(torch.stack(violations)).mean()


@pytest.mark.parametrize("distance", ["EU", "Cosine"])
@pytest.mark.parametrize("n_labels", [2, 5])
def test_ranking_loss_matches_the_authors_loop(distance, n_labels):
    """The dense form is a rewrite for speed, so it must equal the loop exactly.

    Shaped like a real batch: the archive settings give 4 * (2 * 5 + 1) = 44
    embeddings with every case repeated 11 times.
    """
    generator = torch.Generator().manual_seed(3)
    embeddings = torch.randn(44, 16, generator=generator)
    labels = torch.randint(0, n_labels, (4,), generator=generator).repeat(11)

    expected = _loop_ranking_loss(embeddings, labels, distance)
    actual = _ranking_loss(embeddings, labels, distance)
    if expected is None:
        assert actual is None
    else:
        assert float(actual) == pytest.approx(float(expected), rel=1e-6)


def test_ranking_loss_gradient_matches_the_authors_loop():
    """Equal values are not enough; the training signal must match too."""
    generator = torch.Generator().manual_seed(5)
    base = torch.randn(22, 8, generator=generator)
    labels = torch.tensor([0, 1]).repeat(11)

    grads = []
    for loss_fn in (_loop_ranking_loss, _ranking_loss):
        embeddings = base.clone().requires_grad_(True)
        loss_fn(embeddings, labels, "EU").backward()
        grads.append(embeddings.grad)
    assert torch.allclose(grads[0], grads[1], atol=1e-6)


def test_ranking_loss_is_none_without_negatives():
    """A single-class batch has nothing to rank, where the original raises."""
    embeddings = torch.tensor([[0.0], [1.0]])
    assert _ranking_loss(embeddings, torch.tensor([0, 0]), "EU") is None


def test_same_class_neighbour_draws_within_the_class():
    """Every replacement comes from the same class, and never from itself."""
    embeddings = torch.arange(6, dtype=torch.float32).reshape(6, 1)
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    generator = torch.Generator().manual_seed(0)
    out = _same_class_neighbour(embeddings, labels, generator)
    for position in range(6):
        drawn = int(out[position, 0])
        assert labels[drawn] == labels[position]
        assert drawn != position


def test_same_class_neighbour_keeps_a_singleton():
    """A class with one member in the batch becomes its own positive."""
    embeddings = torch.tensor([[0.0], [1.0], [2.0]])
    labels = torch.tensor([0, 0, 1])
    out = _same_class_neighbour(embeddings, labels, torch.Generator().manual_seed(0))
    assert float(out[2, 0]) == 2.0


def test_augment_shapes_follow_the_positive_count():
    """The batch grows to 2 * aug_positives + 1 blocks, labels with it."""
    embeddings = torch.randn(4, 8)
    labels = torch.tensor([0, 1, 0, 1])
    augmented, repeated = _augment(embeddings, labels, 3)
    assert augmented.shape == (4 * (2 * 3 + 1), 8)
    assert repeated.shape == (4 * (2 * 3 + 1),)
    # the first block is the normalised input, and the rest are jittered
    assert torch.allclose(augmented[:4], torch.nn.functional.normalize(embeddings, dim=1))
    assert not torch.allclose(augmented[4:8], embeddings)


def test_encoder_shapes():
    """The encoder gives a 320 wide representation and projection."""
    model = _build_encoder(3)
    projected, representation = model(torch.randn(5, 3, 60))
    assert projected.shape == (5, 320)
    assert representation.shape == (5, 320)


def test_fit_predict_proba():
    """Probabilities are well formed and predictions are known labels."""
    X, y = _data()
    clf = RankSCLClassifier(**SMALL).fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (len(y), clf.n_classes_)
    assert np.allclose(proba.sum(axis=1), 1)
    assert set(clf.predict(X)).issubset(set(clf.classes_))


def test_repeatable_on_cpu():
    """The same seed gives the same probabilities."""
    X, y = _data()
    first = RankSCLClassifier(**SMALL).fit(X, y).predict_proba(X)
    second = RankSCLClassifier(**SMALL).fit(X, y).predict_proba(X)
    assert np.allclose(first, second)


def test_batch_size_larger_than_the_collection_is_refused():
    """The original drops the last incomplete batch, so this trains on nothing."""
    X, y = _data(n_cases=3)
    with pytest.raises(ValueError, match="exceeds the 3 training cases"):
        RankSCLClassifier(**{**SMALL, "batch_size": 8}).fit(X, y)


@pytest.mark.parametrize(
    "parameters,message",
    [
        ({"distance": "manhattan"}, "distance"),
        ({"probe": "forest"}, "probe"),
        ({"aug_positives": -1}, "aug_positives"),
        ({"n_epochs": 0}, "n_epochs"),
    ],
)
def test_parameters_are_validated(parameters, message):
    """Bad parameters are rejected before any training happens."""
    X, y = _data()
    with pytest.raises(ValueError, match=message):
        RankSCLClassifier(**{**SMALL, **parameters}).fit(X, y)


def test_shape_is_checked_at_predict():
    """A collection of a different shape is refused rather than mispredicted."""
    X, y = _data()
    clf = RankSCLClassifier(**SMALL).fit(X, y)
    with pytest.raises(ValueError, match="channels"):
        clf.predict(np.random.random((4, 5, 40)))
    with pytest.raises(ValueError, match="length"):
        clf.predict(np.random.random((4, 2, 17)))


def test_svm_grid_does_not_carry_platt_scaling():
    """The searched estimator must match the authors' probability=False.

    libsvm fits Platt scaling by an internal five-fold cross-validation inside
    every ``fit``, so carrying probability=True into the grid turns a ten-value
    search over five folds from about 50 SVC trainings into 300. On the archive
    collections that alone ran the probe past a 60 hour job.
    """
    clf = RankSCLClassifier(probe="svm")
    clf.n_classes_ = 2
    assert clf._build_probe(n_cases=100, seed=0).estimator.probability is False
    assert clf._build_probe(n_cases=10, seed=0).probability is False


def test_svm_probe_ends_up_with_probabilities():
    """predict_proba is part of the aeon interface, so the fitted probe needs it."""
    X, y = _data(n_cases=60)
    clf = RankSCLClassifier(**{**SMALL, "probe": "svm"}).fit(X, y)
    assert clf.probe_.probability is True
    proba = clf.predict_proba(X)
    assert proba.shape == (len(y), clf.n_classes_)
    assert np.allclose(proba.sum(axis=1), 1)
