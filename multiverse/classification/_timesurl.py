"""TimesURL classifier for aeon.

Adapted from the authors' TimesURL implementation:
https://github.com/Alrash/TimesURL

TimesURL is a self-supervised representation learner, not an end-to-end
classifier. The encoder is pretrained with a contrastive objective, the training
collection is encoded, and a linear probe is fitted on those representations.
Classification is therefore a two-stage procedure, which is why this wrapper does
not share the training loop used by the other ported networks.

The authors' package is vendored under ``_timesurl_original`` and imported as a
subpackage. Three changes were made to it, all recorded in that package's
``__init__``: its sibling imports were rewritten as relative imports so no
``sys.path`` manipulation is needed, a stray ``from .encoder import TSEncoder``
at package level was dropped because ``encoder`` lives under ``models``, and one
unconditional ``print`` of the training tensor shape was silenced. The
architecture and training procedure are untouched.

This wrapper is designed for aeon and therefore assumes input X is a 3D NumPy
array with shape (n_cases, n_channels, n_timepoints). The original implementation
expects (n_cases, n_timepoints, n_channels) with an appended time coordinate and
an observation mask, both of which are constructed here.

The original source is distributed under the MIT License.

MIT License

Copyright (c) 2024 Jiexi Liu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

__maintainer__ = ["TonyBagnall"]
__all__ = ["TimesURLClassifier"]

import random
from types import SimpleNamespace

import numpy as np
from aeon.classification import BaseClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils import check_random_state


class TimesURLClassifier(BaseClassifier):
    """TimesURL self-supervised pretraining followed by a linear probe.

    The encoder is pretrained on the training collection with the authors'
    contrastive objective, the collection is then encoded, and a logistic
    regression probe is fitted on the representations. At prediction time the
    fitted encoder embeds the new series and the probe classifies them.

    The encoder is fitted exclusively on the training collection. Test series are
    encoded only after pretraining, and never take part in augmentation, masking,
    normalisation or probe fitting, so nothing about the test set informs the
    representation.

    Parameters
    ----------
    output_dims : int, default=320
        Width of the representation the encoder produces.
    hidden_dims : int, default=64
        Width of the encoder's hidden layers.
    depth : int, default=10
        Number of dilated convolution blocks in the encoder.
    n_iters : int, default=200
        Number of pretraining iterations.
    batch_size : int, default=16
        Pretraining batch size.
    learning_rate : float, default=0.001
        Encoder learning rate.
    probe_max_iter : int, default=1000
        Maximum iterations for the logistic regression probe.
    lmd : float, default=0.01
        Weight of the reconstruction term in the authors' loss.
    segment_num : int, default=3
        Number of masked segments per series during pretraining.
    mask_ratio_per_seg : float, default=0.05
        Fraction of the series masked in each segment.
    device : {"auto", "cpu", "cuda"} or torch device string, default="auto"
        Device used for pretraining and encoding. ``"auto"`` selects CUDA when
        available and otherwise CPU.
    verbose : bool, default=False
        Whether the encoder prints pretraining progress.
    random_state : int, RandomState instance or None, default=1234
        Seed controlling encoder initialisation, masking and the probe.

    Attributes
    ----------
    encoder_ : object
        Pretrained TimesURL encoder.
    probe_ : sklearn.linear_model.LogisticRegression
        Linear probe fitted on the encoded training collection.
    device_ : str
        Resolved device.
    n_channels_ : int
        Number of channels seen in ``fit``.
    n_timepoints_ : int
        Series length seen in ``fit``.
    classes_ : np.ndarray
        Class labels, from ``BaseClassifier``.
    n_classes_ : int
        Number of classes, from ``BaseClassifier``.

    References
    ----------
    .. [1] Liu, J. and Chen, S. "TimesURL: Self-supervised Contrastive Learning
       for Universal Time Series Representation Learning." AAAI, 2024.

    Examples
    --------
    >>> from aeon.testing.data_generation import make_example_3d_numpy
    >>> from multiverse.classification import TimesURLClassifier
    >>> X, y = make_example_3d_numpy(n_cases=8, n_channels=2, n_timepoints=20)
    >>> clf = TimesURLClassifier(n_iters=2, device="cpu")  # doctest: +SKIP
    >>> clf.fit(X, y)  # doctest: +SKIP
    """

    _tags = {
        "X_inner_type": "numpy3D",
        "capability:multivariate": True,
        "capability:unequal_length": False,
        "algorithm_type": "deeplearning",
        "non_deterministic": True,
        "python_dependencies": "torch",
    }

    def __init__(
        self,
        output_dims: int = 320,
        hidden_dims: int = 64,
        depth: int = 10,
        n_iters: int = 200,
        batch_size: int = 16,
        learning_rate: float = 1e-3,
        probe_max_iter: int = 1000,
        lmd: float = 0.01,
        segment_num: int = 3,
        mask_ratio_per_seg: float = 0.05,
        device: str = "auto",
        verbose: bool = False,
        random_state=1234,
    ):
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.depth = depth
        self.n_iters = n_iters
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.probe_max_iter = probe_max_iter
        self.lmd = lmd
        self.segment_num = segment_num
        self.mask_ratio_per_seg = mask_ratio_per_seg
        self.device = device
        self.verbose = verbose
        self.random_state = random_state
        super().__init__()

    def _validate_parameters(self) -> None:
        """Check constructor parameters before any work is done."""
        for name in [
            "output_dims",
            "hidden_dims",
            "depth",
            "n_iters",
            "batch_size",
            "probe_max_iter",
            "segment_num",
        ]:
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.learning_rate < 0:
            raise ValueError("learning_rate must be non-negative")
        if not 0 <= self.mask_ratio_per_seg < 1:
            raise ValueError("mask_ratio_per_seg must be in [0, 1)")
        if self.lmd < 0:
            raise ValueError("lmd must be non-negative")

    def _resolve_device(self) -> str:
        import torch

        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if str(self.device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return self.device

    @staticmethod
    def _to_original_layout(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Build the authors' (values, mask) input from an aeon collection.

        The original code expects ``(n_cases, n_timepoints, n_channels + 1)``,
        where the final column is a time coordinate on [0, 1], plus a separate
        all-ones observation mask over the real channels.
        """
        series = np.transpose(np.asarray(X, dtype=np.float32), (0, 2, 1))
        n_cases, n_timepoints, n_channels = series.shape
        time = np.broadcast_to(
            np.linspace(0, 1, n_timepoints, dtype=np.float32)[None, :, None],
            (n_cases, n_timepoints, 1),
        )
        values = np.concatenate([series, time], axis=2)
        mask = np.ones((n_cases, n_timepoints, n_channels), dtype=np.float32)
        return values, mask

    def _encode(self, X: np.ndarray) -> np.ndarray:
        """Embed a collection with the fitted encoder, one vector per case."""
        values, mask = self._to_original_layout(X)
        z = self.encoder_.encode(
            {"x": values, "mask": mask}, encoding_window="full_series"
        )
        return z.reshape(z.shape[0], -1)

    def _fit(self, X: np.ndarray, y):
        self._validate_parameters()

        import torch

        from multiverse.classification._timesurl_original.timesurl import TimesURL

        rng = check_random_state(self.random_state)
        seed = int(rng.randint(np.iinfo(np.int32).max))
        # The authors' collator draws from Python's `random` for segment masking
        # and index shuffling, so seeding numpy and torch alone leaves the run
        # irreproducible.
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.device_ = self._resolve_device()
        self.n_channels_, self.n_timepoints_ = X.shape[1], X.shape[2]

        values, mask = self._to_original_layout(X)
        args = SimpleNamespace(
            lmd=self.lmd,
            segment_num=self.segment_num,
            mask_ratio_per_seg=self.mask_ratio_per_seg,
            batch_size=self.batch_size,
        )
        self.encoder_ = TimesURL(
            self.n_channels_,
            self.output_dims,
            self.hidden_dims,
            self.depth,
            device=self.device_,
            lr=self.learning_rate,
            batch_size=self.batch_size,
            args=args,
        )
        self.encoder_.fit(
            {"x": values, "mask": mask},
            n_iters=self.n_iters,
            verbose=self.verbose,
            is_scheduler=False,
        )

        encoded = self._encode(X)
        encoded_y = np.asarray(
            [self._class_dictionary[label] for label in y], dtype=np.int64
        )
        # multi_class is left at its default: the authors pass "auto", which
        # sklearn deprecated in 1.5 and removes in 1.8, and whose behaviour the
        # default already matches for these problems.
        self.probe_ = LogisticRegression(
            max_iter=self.probe_max_iter, random_state=seed
        ).fit(encoded, encoded_y)
        return self

    def _check_shape(self, X: np.ndarray) -> None:
        if X.shape[1] != self.n_channels_:
            raise ValueError(
                f"X has {X.shape[1]} channels, but the classifier was fitted "
                f"with {self.n_channels_}."
            )
        if X.shape[2] != self.n_timepoints_:
            raise ValueError(
                f"X has length {X.shape[2]}, but the classifier was fitted with "
                f"length {self.n_timepoints_}."
            )

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_shape(X)
        return self.probe_.predict_proba(self._encode(X))

    def _predict(self, X: np.ndarray):
        self._check_shape(X)
        return self.classes_[self.probe_.predict(self._encode(X))]

    @classmethod
    def _get_test_params(cls, parameter_set: str = "default") -> dict:
        """Return a small parameter set for aeon estimator checks."""
        return {
            "output_dims": 8,
            "hidden_dims": 8,
            "depth": 2,
            "n_iters": 2,
            "batch_size": 4,
            "probe_max_iter": 50,
            "device": "cpu",
            "random_state": 0,
        }
