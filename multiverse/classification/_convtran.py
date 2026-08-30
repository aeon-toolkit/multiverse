"""ConvTran classifier for aeon.

Adapted directly from commit 148afb6 of the authors' implementation:
https://github.com/Navidfoumani/ConvTran

This wrapper is designed for aeon and therefore assumes input X is a 3D NumPy
array with shape (n_cases, n_channels, n_timepoints), which is also the layout
expected by the original ConvTran network.

The original source is distributed under the MIT License.

MIT License

Copyright (c) 2022 Department of Data Science and Artificial Intelligence
@Monash University

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

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

__maintainer__ = ["TonyBagnall"]
__all__ = ["ConvTranClassifier"]


import math

import numpy as np
import torch
from torch import nn
from torch.optim.optimizer import Optimizer

from multiverse.classification._torch_base import (
    _BaseTorchClassifier,
    _EfficientRelativePositionAttention,
    _TimeAbsolutePositionalEncoding,
)


class _ConvTranNetwork(nn.Module):
    """Original ConvTran network using tAPE and eRPE."""

    def __init__(
        self,
        n_channels,
        n_timepoints,
        n_classes,
        emb_size,
        num_heads,
        dim_ff,
        dropout,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_timepoints = n_timepoints

        self.embed_layer = nn.Sequential(
            nn.Conv2d(1, emb_size * 4, kernel_size=(1, 8), padding="same"),
            nn.BatchNorm2d(emb_size * 4),
            nn.GELU(),
        )
        self.embed_layer2 = nn.Sequential(
            nn.Conv2d(
                emb_size * 4,
                emb_size,
                kernel_size=(n_channels, 1),
                padding="valid",
            ),
            nn.BatchNorm2d(emb_size),
            nn.GELU(),
        )
        self.fix_position = _TimeAbsolutePositionalEncoding(
            emb_size, dropout=dropout, max_len=n_timepoints
        )
        self.attention_layer = _EfficientRelativePositionAttention(
            emb_size, num_heads, n_timepoints, dropout
        )
        self.layer_norm = nn.LayerNorm(emb_size, eps=1e-5)
        self.layer_norm2 = nn.LayerNorm(emb_size, eps=1e-5)
        self.feed_forward = nn.Sequential(
            nn.Linear(emb_size, dim_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, emb_size),
            nn.Dropout(dropout),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.out = nn.Linear(emb_size, n_classes)

    def forward(self, x):
        """Return logits for aeon-shaped input (batch, channels, timepoints)."""
        if x.ndim != 3:
            raise ValueError("ConvTran expects a 3D tensor")
        if x.shape[1] != self.n_channels or x.shape[2] != self.n_timepoints:
            raise ValueError(
                "ConvTran input shape changed after fitting: expected "
                f"(*, {self.n_channels}, {self.n_timepoints}), got {tuple(x.shape)}"
            )

        x = x.unsqueeze(1)
        x_src = self.embed_layer(x)
        x_src = self.embed_layer2(x_src).squeeze(2)
        x_src = x_src.permute(0, 2, 1)
        x_src_pos = self.fix_position(x_src)
        attention = x_src + self.attention_layer(x_src_pos)
        attention = self.layer_norm(attention)
        output = attention + self.feed_forward(attention)
        output = self.layer_norm2(output)
        output = output.permute(0, 2, 1)
        output = self.gap(output)
        output = self.flatten(output)
        return self.out(output)


class _RAdam(Optimizer):
    """RAdam optimizer copied from the authors' ConvTran implementation."""

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        degenerated_to_sgd=True,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")

        self.degenerated_to_sgd = degenerated_to_sgd
        if (
            isinstance(params, (list, tuple))
            and len(params) > 0
            and isinstance(params[0], dict)
        ):
            for param in params:
                if "betas" in param and param["betas"] != betas:
                    param["buffer"] = [[None, None, None] for _ in range(10)]
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "buffer": [[None, None, None] for _ in range(10)],
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform one optimizer step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad.float()
                if gradient.is_sparse:
                    raise RuntimeError("RAdam does not support sparse gradients")

                parameter_fp32 = parameter.float()
                state = self.state[parameter]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter_fp32)
                    state["exp_avg_sq"] = torch.zeros_like(parameter_fp32)
                else:
                    state["exp_avg"] = state["exp_avg"].type_as(parameter_fp32)
                    state["exp_avg_sq"] = state["exp_avg_sq"].type_as(parameter_fp32)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]
                exp_avg_sq.mul_(beta2).addcmul_(
                    gradient, gradient, value=1 - beta2
                )
                exp_avg.mul_(beta1).add_(gradient, alpha=1 - beta1)

                state["step"] += 1
                buffered = group["buffer"][state["step"] % 10]
                if state["step"] == buffered[0]:
                    n_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state["step"]
                    beta2_t = beta2 ** state["step"]
                    n_sma_max = 2 / (1 - beta2) - 1
                    n_sma = (
                        n_sma_max
                        - 2 * state["step"] * beta2_t / (1 - beta2_t)
                    )
                    buffered[1] = n_sma
                    if n_sma >= 5:
                        step_size = math.sqrt(
                            (1 - beta2_t)
                            * (n_sma - 4)
                            / (n_sma_max - 4)
                            * (n_sma - 2)
                            / n_sma
                            * n_sma_max
                            / (n_sma_max - 2)
                        ) / (1 - beta1 ** state["step"])
                    elif self.degenerated_to_sgd:
                        step_size = 1.0 / (1 - beta1 ** state["step"])
                    else:
                        step_size = -1
                    buffered[2] = step_size

                if n_sma >= 5:
                    if group["weight_decay"] != 0:
                        parameter_fp32.add_(
                            parameter_fp32,
                            alpha=-group["weight_decay"] * group["lr"],
                        )
                    denominator = exp_avg_sq.sqrt().add_(group["eps"])
                    parameter_fp32.addcdiv_(
                        exp_avg,
                        denominator,
                        value=-step_size * group["lr"],
                    )
                    parameter.copy_(parameter_fp32)
                elif step_size > 0:
                    if group["weight_decay"] != 0:
                        parameter_fp32.add_(
                            parameter_fp32,
                            alpha=-group["weight_decay"] * group["lr"],
                        )
                    parameter_fp32.add_(
                        exp_avg, alpha=-step_size * group["lr"]
                    )
                    parameter.copy_(parameter_fp32)
        return loss


class ConvTranClassifier(_BaseTorchClassifier):
    """Convolutional Transformer (ConvTran) classifier.

    This is a direct PyTorch port of the authors' ConvTran model and training
    procedure, adapted to the aeon estimator interface. Input is an aeon-format
    NumPy array with shape ``(n_cases, n_channels, n_timepoints)``. The original
    implementation uses the same layout.

    Parameters
    ----------
    emb_size : int, default=16
        Transformer embedding dimension. Must be even and divisible by
        ``num_heads``.
    dim_ff : int, default=256
        Hidden dimension of the feed-forward block.
    num_heads : int, default=8
        Number of attention heads.
    dropout : float, default=0.01
        Dropout used by tAPE and the feed-forward block.
    n_epochs : int, default=100
        Number of training epochs.
    batch_size : int, default=16
        Training and prediction batch size.
    learning_rate : float, default=0.001
        Learning rate for the authors' RAdam optimizer.
    validation_size : float, default=0.2
        Fraction of the supplied training data held out for validation and
        best-loss checkpoint selection. Set to 0 to train on all supplied data
        and retain the epoch with the lowest training loss.
    gradient_clip_norm : float or None, default=4.0
        Maximum gradient norm, matching the original training loop. No clipping
        is performed when ``None``.
    device : {"auto", "cpu", "cuda"} or torch device string, default="auto"
        Device used for training and prediction. ``"auto"`` selects CUDA when
        available and otherwise CPU.
    num_workers : int, default=0
        Number of data-loader worker processes.
    verbose : bool, default=False
        Whether to print epoch losses.
    random_state : int, RandomState instance or None, default=1234
        Seed controlling the validation split, model initialization, and batch
        shuffling. The default matches the original implementation.

    Attributes
    ----------
    model_ : torch.nn.Module
        Fitted network restored to the epoch with the best validation loss.
    history_ : list of dict
        Training and validation loss for each epoch.
    device_ : torch.device
        Resolved training device.
    best_epoch_ : int
        One-based index of the retained epoch.
    best_validation_loss_ : float
        Loss used to choose the retained epoch.

    References
    ----------
    .. [1] Foumani, N. M., Tan, C. W., Webb, G. I., and Salehi, M.
       "Improving position encoding of transformers for multivariate time
       series classification." Data Mining and Knowledge Discovery 38,
       22-48, 2024.
    """

    _tags = {
        "X_inner_type": "numpy3D",
        "capability:multivariate": True,
        "algorithm_type": "deeplearning",
        "non_deterministic": True,
        "python_dependencies": "torch",
    }

    def __init__(
        self,
        emb_size=16,
        dim_ff=256,
        num_heads=8,
        dropout=0.01,
        n_epochs=100,
        batch_size=16,
        learning_rate=1e-3,
        validation_size=0.2,
        gradient_clip_norm=4.0,
        device="auto",
        num_workers=0,
        verbose=False,
        random_state=1234,
    ):
        self.emb_size = emb_size
        self.dim_ff = dim_ff
        self.num_heads = num_heads
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.validation_size = validation_size
        self.gradient_clip_norm = gradient_clip_norm
        self.device = device
        self.num_workers = num_workers
        self.verbose = verbose
        self.random_state = random_state
        super().__init__()

    def _validate_parameters(self):
        super()._validate_parameters()
        self._validate_positive_int(self.emb_size, "emb_size")
        if self.emb_size % 2 != 0:
            raise ValueError("emb_size must be even for tAPE")
        self._validate_positive_int(self.num_heads, "num_heads")
        if self.emb_size % self.num_heads != 0:
            raise ValueError("emb_size must be divisible by num_heads")
        self._validate_positive_int(self.dim_ff, "dim_ff")
        self._validate_dropout(self.dropout, "dropout")

    def _build_model(self):
        return _ConvTranNetwork(
            n_channels=self.n_channels_,
            n_timepoints=self.n_timepoints_,
            n_classes=self.n_classes_,
            emb_size=self.emb_size,
            num_heads=self.num_heads,
            dim_ff=self.dim_ff,
            dropout=self.dropout,
        )

    def _build_optimizer(self):
        return _RAdam(
            self.model_.parameters(), lr=self.learning_rate, weight_decay=0
        )


    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return a small parameter set for aeon estimator checks."""
        return {
            "emb_size": 8,
            "dim_ff": 16,
            "num_heads": 2,
            "n_epochs": 2,
            "batch_size": 4,
            "validation_size": 0.25,
            "device": "cpu",
            "random_state": 0,
        }
