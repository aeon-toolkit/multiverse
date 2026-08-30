"""Shared PyTorch components for the ported deep learning classifiers.

ConvTran and PatchMTSC are both built on the positional encodings introduced by
Foumani et al., and the authors of both use the same training procedure. Rather
than carry two copies, the common parts live here:

- :class:`_TimeAbsolutePositionalEncoding`, the tAPE encoding;
- :class:`_EfficientRelativePositionAttention`, the eRPE attention block;
- :class:`_BaseTorchClassifier`, the shared aeon wrapper machinery (device and
  seed resolution, data loaders, the epoch loop with best-epoch checkpointing,
  and prediction).

TimesNet deliberately does not use :class:`_BaseTorchClassifier`. Its training
procedure differs in ways that would change results if unified: it seeds from
``random_state`` directly rather than through ``check_random_state``, selects on
validation *accuracy* with early stopping rather than validation loss, uses
RAdam, and standardises per channel.

See ``_convtran.py`` and ``_patchmtsc.py`` for the licences of the original
implementations these components are adapted from.
"""

__maintainer__ = ["TonyBagnall"]
__all__ = [
    "_BaseTorchClassifier",
    "_EfficientRelativePositionAttention",
    "_TimeAbsolutePositionalEncoding",
]

import math
from copy import deepcopy

import numpy as np
import torch
from aeon.classification import BaseClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import check_random_state
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class _TimeAbsolutePositionalEncoding(nn.Module):
    """Original time absolute position encoding (tAPE).

    Sinusoidal encoding scaled by ``d_model / max_len``, so that the frequency
    of the encoding stays meaningful for the short sequences typical of time
    series classification.
    """

    def __init__(self, d_model, dropout, max_len):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        positional_encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        divisor = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        scale = d_model / max_len
        positional_encoding[:, 0::2] = torch.sin(position * divisor * scale)
        positional_encoding[:, 1::2] = torch.cos(position * divisor * scale)
        self.register_buffer("pe", positional_encoding.unsqueeze(0))

    def forward(self, x):
        """Add tAPE to input shaped (batch, sequence, embedding)."""
        return self.dropout(x + self.pe)


class _EfficientRelativePositionAttention(nn.Module):
    """Original efficient relative position encoding (eRPE) attention.

    Multi-head attention where a learned relative position bias is added to the
    attention matrix *after* the softmax, which is what distinguishes eRPE from
    the usual pre-softmax relative encodings.
    """

    def __init__(self, emb_size, num_heads, seq_len, dropout):
        super().__init__()
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.scale = emb_size**-0.5
        self.key = nn.Linear(emb_size, emb_size, bias=False)
        self.value = nn.Linear(emb_size, emb_size, bias=False)
        self.query = nn.Linear(emb_size, emb_size, bias=False)

        self.relative_bias_table = nn.Parameter(
            torch.zeros((2 * self.seq_len - 1), num_heads)
        )
        coords = torch.meshgrid(
            torch.arange(1), torch.arange(self.seq_len), indexing="ij"
        )
        coords = torch.flatten(torch.stack(coords), 1)
        relative_coords = coords[:, :, None] - coords[:, None, :]
        relative_coords[1] += self.seq_len - 1
        relative_coords = relative_coords.permute(1, 2, 0)
        relative_index = relative_coords.sum(-1).flatten().unsqueeze(1)
        self.register_buffer("relative_index", relative_index)

        # Constructed by the original module but never applied in its forward
        # pass. Retained so the module matches the published architecture; it
        # holds no parameters, so it does not affect results or the state dict.
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.LayerNorm(emb_size)

    def forward(self, x):
        """Apply multi-head attention followed by the original eRPE bias."""
        batch_size, seq_len, _ = x.shape
        key = (
            self.key(x)
            .reshape(batch_size, seq_len, self.num_heads, -1)
            .permute(0, 2, 3, 1)
        )
        value = (
            self.value(x)
            .reshape(batch_size, seq_len, self.num_heads, -1)
            .transpose(1, 2)
        )
        query = (
            self.query(x)
            .reshape(batch_size, seq_len, self.num_heads, -1)
            .transpose(1, 2)
        )

        attention = torch.matmul(query, key) * self.scale
        attention = nn.functional.softmax(attention, dim=-1)

        relative_bias = self.relative_bias_table.gather(
            0, self.relative_index.repeat(1, self.num_heads)
        )
        relative_bias = (
            relative_bias.reshape(self.seq_len, self.seq_len, self.num_heads)
            .permute(2, 0, 1)
            .unsqueeze(0)
        )
        attention = attention + relative_bias

        output = torch.matmul(attention, value)
        output = output.transpose(1, 2).reshape(batch_size, seq_len, -1)
        return self.to_out(output)


class _BaseTorchClassifier(BaseClassifier):
    """Shared training and prediction machinery for the ported networks.

    Subclasses supply the network and its optimizer by implementing
    :meth:`_build_model` and :meth:`_build_optimizer`, and may extend
    :meth:`_validate_parameters`. Everything else, the held-out validation
    split, the epoch loop, best-epoch checkpointing, and prediction, is shared.

    Subclasses are expected to define the constructor parameters this class
    reads: ``n_epochs``, ``batch_size``, ``learning_rate``, ``validation_size``,
    ``gradient_clip_norm``, ``device``, ``num_workers``, ``verbose``, and
    ``random_state``. This class deliberately defines no ``__init__``, so that
    scikit-learn parameter introspection sees only the concrete classifier.
    """

    def _setup_fit(self, X):
        """Derive any fitted attributes the network needs before construction.

        Called once ``n_channels_`` and ``n_timepoints_`` are known and before
        :meth:`_build_model`. The default does nothing.
        """

    def _build_model(self):
        """Return the network to train, before it is moved to the device."""
        raise NotImplementedError

    def _build_optimizer(self):
        """Return the optimizer for ``self.model_``."""
        raise NotImplementedError

    def _validate_parameters(self):
        """Check the parameters shared by every subclass."""
        if not isinstance(self.n_epochs, int) or self.n_epochs <= 0:
            raise ValueError("n_epochs must be a positive integer")
        if not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if self.learning_rate < 0:
            raise ValueError("learning_rate must be non-negative")
        if not 0 <= self.validation_size < 1:
            raise ValueError("validation_size must be in [0, 1)")
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive or None")
        if not isinstance(self.num_workers, int) or self.num_workers < 0:
            raise ValueError("num_workers must be a non-negative integer")

    @staticmethod
    def _validate_positive_int(value, name):
        """Raise unless ``value`` is a positive integer."""
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    @staticmethod
    def _validate_dropout(value, name):
        """Raise unless ``value`` is a valid dropout rate."""
        if not 0 <= value < 1:
            raise ValueError(f"{name} must be in [0, 1)")

    def _resolve_device(self):
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        resolved = torch.device(self.device)
        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return resolved

    def _seed_torch(self):
        rng = check_random_state(self.random_state)
        self.random_state_ = int(rng.randint(np.iinfo(np.int32).max))
        torch.manual_seed(self.random_state_)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state_)

    def _make_loader(self, X, y=None, shuffle=False):
        X_tensor = torch.as_tensor(X, dtype=torch.float32)
        if y is None:
            dataset = TensorDataset(X_tensor)
        else:
            y_tensor = torch.as_tensor(y, dtype=torch.int64)
            dataset = TensorDataset(X_tensor, y_tensor)
        generator = torch.Generator()
        generator.manual_seed(self.random_state_)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            pin_memory=self.device_.type == "cuda",
            num_workers=self.num_workers,
            generator=generator,
        )

    def _loss(self, loader, train):
        self.model_.train(mode=train)
        total_loss = 0.0
        total_cases = 0
        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device_, non_blocking=True)
                y_batch = y_batch.to(self.device_, non_blocking=True)
                logits = self.model_(X_batch)
                losses = nn.functional.cross_entropy(
                    logits, y_batch, reduction="none"
                )
                loss = losses.mean()
                if train:
                    self.optimizer_.zero_grad()
                    loss.backward()
                    if self.gradient_clip_norm is not None:
                        nn.utils.clip_grad_norm_(
                            self.model_.parameters(), self.gradient_clip_norm
                        )
                    self.optimizer_.step()
                total_loss += losses.detach().sum().item()
                total_cases += len(y_batch)
        return total_loss / total_cases

    def _split_train_validation(self, X, y):
        """Encode labels and split off the internal validation set."""
        encoded_y = np.asarray(
            [self._class_dictionary[label] for label in y], dtype=np.int64
        )
        if self.validation_size <= 0:
            return X, encoded_y, None, None

        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=self.validation_size,
            random_state=self.random_state_,
        )
        train_indices, validation_indices = next(
            splitter.split(np.zeros(len(encoded_y)), encoded_y)
        )
        return (
            X[train_indices],
            encoded_y[train_indices],
            X[validation_indices],
            encoded_y[validation_indices],
        )

    def _fit(self, X, y):
        """Fit the network, retaining the epoch with the lowest held-out loss."""
        self._validate_parameters()
        self._seed_torch()
        self.device_ = self._resolve_device()
        self.n_channels_ = X.shape[1]
        self.n_timepoints_ = X.shape[2]
        self._setup_fit(X)

        X_train, y_train, X_validation, y_validation = self._split_train_validation(
            X, y
        )

        self.model_ = self._build_model().to(self.device_)
        self.optimizer_ = self._build_optimizer()

        train_loader = self._make_loader(X_train, y_train, shuffle=True)
        validation_loader = (
            None
            if X_validation is None
            else self._make_loader(X_validation, y_validation, shuffle=False)
        )

        best_state = None
        best_loss = math.inf
        self.history_ = []
        for epoch in range(self.n_epochs):
            train_loss = self._loss(train_loader, train=True)
            validation_loss = (
                train_loss
                if validation_loader is None
                else self._loss(validation_loader, train=False)
            )
            self.history_.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "validation_loss": validation_loss,
                }
            )
            if validation_loss < best_loss:
                best_loss = validation_loss
                self.best_epoch_ = epoch + 1
                best_state = deepcopy(self.model_.state_dict())
            if self.verbose:
                print(
                    f"Epoch {epoch + 1}/{self.n_epochs}: "
                    f"loss={train_loss:.6f}, val_loss={validation_loss:.6f}"
                )

        self.model_.load_state_dict(best_state)
        self.model_.eval()
        self.best_validation_loss_ = best_loss
        return self

    def _predict_proba(self, X):
        loader = self._make_loader(X, shuffle=False)
        self.model_.eval()
        probabilities = []
        with torch.no_grad():
            for (X_batch,) in loader:
                X_batch = X_batch.to(self.device_, non_blocking=True)
                logits = self.model_(X_batch)
                probabilities.append(nn.functional.softmax(logits, dim=1).cpu())
        return torch.cat(probabilities, dim=0).numpy()

    def _predict(self, X):
        return self.classes_[np.argmax(self._predict_proba(X), axis=1)]
