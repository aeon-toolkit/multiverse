"""PatchMTSC classifier for aeon.

Adapted from the authors' PatchMTSC implementation:
https://github.com/YanxuanWei/PatchMTSC

The original source is split over several modules (``Models/model.py``,
``Models/Attention.py``, ``Models/AbsolutePositionalEncoding.py``). Only the
components reachable from the ``PatchMTSC`` network with the paper's default
configuration (``tAPE`` fixed encoding and ``eRPE`` relative encoding) are
reproduced here, so that this module is self-contained. Two mechanical changes
were made:

- the two ``einops.rearrange`` calls in the eRPE attention are expressed with
  ``reshape``/``permute``, removing the ``einops`` dependency;
- the classification head is an ``nn.Linear`` with the width derived from the
  architecture rather than the original hard-coded 512-wide layer, so that
  arbitrary channel counts and patch lengths are supported.

This wrapper is designed for aeon and therefore assumes input X is a 3D NumPy
array with shape (n_cases, n_channels, n_timepoints), which is also the layout
expected by the original PatchMTSC network.

The original source is distributed under the MIT License.

MIT License

Copyright (c) 2025 Yanxuan Wei

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

__maintainer__ = ["TonyBagnall"]
__all__ = ["PatchMTSCClassifier"]


import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from multiverse.classification._torch_base import (
    _BaseTorchClassifier,
    _EfficientRelativePositionAttention,
    _TimeAbsolutePositionalEncoding,
)


class _DotGraphConstruction(nn.Module):
    """Learned fully connected graph over the nodes of a sequence."""

    def __init__(self, input_dim):
        super().__init__()
        self.mapping = nn.Linear(input_dim, input_dim)

    def forward(self, node_features):
        """Return a row-normalised adjacency matrix per case."""
        node_features = self.mapping(node_features)
        n_nodes = node_features.shape[1]

        adjacency = torch.bmm(node_features, torch.transpose(node_features, 1, 2))
        eye = torch.eye(n_nodes, device=node_features.device).expand_as(adjacency)
        adjacency = F.leaky_relu(adjacency - eye * 1e8)
        adjacency = F.softmax(adjacency, dim=-1)
        return adjacency + eye


def _mask_matrix(time_length, decay_rate):
    """Return the exponential distance decay mask used to weight the graph."""
    indices = torch.arange(time_length)
    return decay_rate ** torch.abs(indices[:, None] - indices[None, :]).float()


class _MessagePassing(nn.Module):
    """k-hop message passing over the constructed graph."""

    def __init__(self, input_dim, output_dim, k):
        super().__init__()
        self.k = k
        self.theta = nn.ModuleList(
            [nn.Linear(input_dim, output_dim) for _ in range(k)]
        )
        self.bn = nn.BatchNorm1d(output_dim)

    def forward(self, x, adjacency):
        """Sum the k hop convolutions, batch normalise, and activate."""
        hop = adjacency
        output = None
        for k in range(self.k):
            if k > 0:
                hop = torch.bmm(hop, adjacency)
            step = self.theta[k](torch.bmm(hop, x))
            output = step if output is None else output + step

        output = torch.transpose(output, -1, -2)
        output = self.bn(output)
        output = torch.transpose(output, -1, -2)
        return F.leaky_relu(output)


class _GraphConvPoolBlock(nn.Module):
    """Graph construction, distance masking, and message passing."""

    def __init__(self, input_dim, output_dim, time_length, decay):
        super().__init__()
        self.graph_construction = _DotGraphConstruction(input_dim)
        self.bn = nn.BatchNorm1d(input_dim)
        self.message_passing = _MessagePassing(input_dim, output_dim, k=1)
        self.register_buffer("pre_relation", _mask_matrix(time_length, decay))

    def forward(self, x):
        """Return message-passed node features for input (batch, nodes, dim)."""
        adjacency = self.graph_construction(x) * self.pre_relation

        normalised = torch.transpose(x, -1, -2)
        normalised = self.bn(normalised)
        normalised = torch.transpose(normalised, -1, -2)
        return self.message_passing(normalised, adjacency)


class _PatchAveragePooling(nn.Module):
    """Patch average pooling (PAP), reducing redundancy from overlapping patches."""

    def __init__(self, dim, dropout):
        super().__init__()
        self.dim = dim
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """Average over the patch axis and apply dropout."""
        return self.dropout(torch.mean(x, self.dim))


class _PatchMTSCNetwork(nn.Module):
    """Original PatchMTSC network using tAPE, eRPE, and patch graph blocks."""

    def __init__(
        self,
        n_channels,
        n_timepoints,
        n_classes,
        emb_size,
        num_heads,
        dim_ff,
        d_model_patch,
        patch_len,
        stride,
        dropout,
        head_dropout,
        pap_dropout,
        graph_decay,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_timepoints = n_timepoints
        self.patch_len = patch_len
        self.stride = stride

        patch_num = int((n_timepoints - patch_len) / stride + 1) + 1
        self.patch_num = patch_num
        self.padding_patch_layer = nn.ReplicationPad1d((0, stride))

        # Convolutional channel mixing: a temporal convolution followed by a
        # convolution spanning every channel, collapsing the channel axis.
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

        self.w_p = nn.Linear(patch_num, d_model_patch)

        self.fix_position = _TimeAbsolutePositionalEncoding(
            d_model_patch, dropout=dropout, max_len=patch_len
        )
        self.attention_layer = _EfficientRelativePositionAttention(
            d_model_patch, num_heads, patch_len, dropout
        )
        self.layer_norm = nn.LayerNorm(d_model_patch, eps=1e-5)
        self.layer_norm2 = nn.LayerNorm(d_model_patch, eps=1e-5)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model_patch, dim_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model_patch),
            nn.Dropout(dropout),
        )

        # The original builds two blocks that differ only by a `stride`
        # argument its forward pass never reads, so they are two independently
        # initialised copies whose outputs are concatenated.
        self.graph_block1 = _GraphConvPoolBlock(
            d_model_patch, d_model_patch, patch_len, graph_decay
        )
        self.graph_block2 = _GraphConvPoolBlock(
            d_model_patch, d_model_patch, patch_len, graph_decay
        )

        self.pap = _PatchAveragePooling(1, pap_dropout)
        self.head_dropout = nn.Dropout(head_dropout)
        self.out = nn.Linear(emb_size * 2 * d_model_patch, n_classes)

    def forward(self, x):
        """Return logits for aeon-shaped input (batch, channels, timepoints)."""
        if x.ndim != 3:
            raise ValueError("PatchMTSC expects a 3D tensor")
        if x.shape[1] != self.n_channels or x.shape[2] != self.n_timepoints:
            raise ValueError(
                "PatchMTSC input shape changed after fitting: expected "
                f"(*, {self.n_channels}, {self.n_timepoints}), got {tuple(x.shape)}"
            )

        x = x.unsqueeze(1)
        x_src = self.embed_layer(x)
        x_src = self.embed_layer2(x_src).squeeze(2)

        x_src = self.padding_patch_layer(x_src)
        x_src = x_src.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        n_vars = x_src.shape[1]
        # Channel-independent patch sequences: (batch * n_vars, patch_len,
        # patch_num), then projected to the patch embedding dimension.
        x_src = x_src.reshape(-1, x_src.shape[2], x_src.shape[3])
        x_src = x_src.permute(0, 2, 1)
        x_src = self.w_p(x_src)

        x_src_pos = self.fix_position(x_src)
        attention = x_src + self.attention_layer(x_src_pos)
        attention = self.layer_norm(attention)
        output = attention + self.feed_forward(attention)
        output = self.layer_norm2(output)

        features = torch.cat(
            [self.pap(self.graph_block1(output)), self.pap(self.graph_block2(output))],
            dim=-1,
        )
        features = features.reshape(-1, n_vars * features.shape[-1])
        return self.out(self.head_dropout(features))


class PatchMTSCClassifier(_BaseTorchClassifier):
    """Patch-based multivariate time series classifier (PatchMTSC).

    This is a direct PyTorch port of the authors' PatchMTSC model and training
    procedure, adapted to the aeon estimator interface. Input is an aeon-format
    NumPy array with shape ``(n_cases, n_channels, n_timepoints)``. The original
    implementation uses the same layout.

    A convolutional channel-mixing front end models inter-channel correlations
    and collapses the channel axis to ``emb_size`` embedding channels. The
    result is replication padded and segmented into overlapping patches. Each
    channel-independent patch sequence is encoded by a Transformer block using
    tAPE followed by eRPE. Two fully connected graph blocks then model
    dependencies across patches, and patch average pooling reduces the
    redundancy introduced by the patch overlap before classification.

    Parameters
    ----------
    emb_size : int, default=16
        Number of embedding channels produced by the channel-mixing front end.
        Must be divisible by 4, since the first convolution uses
        ``emb_size * 4`` filters.
    d_model_patch : int, default=16
        Patch embedding dimension used by the Transformer and graph blocks.
        Must be divisible by ``num_heads``.
    dim_ff : int, default=256
        Hidden dimension of the feed-forward block.
    num_heads : int, default=8
        Number of attention heads.
    patch_len : int, default=16
        Patch length. Clamped to ``n_timepoints`` during ``fit`` when the series
        are shorter than the requested patch.
    stride : int, default=8
        Stride between consecutive patches. Clamped to the resolved patch length.
    dropout : float, default=0.01
        Dropout used by tAPE, the attention block, and the feed-forward block.
    head_dropout : float, default=0.0
        Dropout applied to the pooled features before the output layer.
    pap_dropout : float, default=0.0
        Dropout applied by patch average pooling.
    graph_decay : float, default=5e-4
        Decay rate of the exponential distance mask applied to the constructed
        graph. This is ``weight_decay`` in the original configuration, where it
        is used as a graph mask decay rather than an optimizer setting.
    n_epochs : int, default=100
        Number of training epochs.
    batch_size : int, default=16
        Training and prediction batch size.
    learning_rate : float, default=0.001
        Learning rate for Adam.
    validation_size : float, default=0.2
        Fraction of the supplied training data held out for validation and
        best-loss checkpoint selection. Set to 0 to train on all supplied data
        and retain the epoch with the lowest training loss.
    gradient_clip_norm : float or None, default=None
        Maximum gradient norm. No clipping is performed when ``None``, matching
        the original training loop.
    device : {"auto", "cpu", "cuda"} or torch device string, default="auto"
        Device used for training and prediction. ``"auto"`` selects CUDA when
        available and otherwise CPU.
    num_workers : int, default=0
        Number of data-loader worker processes.
    verbose : bool, default=False
        Whether to print epoch losses.
    random_state : int, RandomState instance or None, default=1234
        Seed controlling the validation split, model initialization, and batch
        shuffling.

    Attributes
    ----------
    model_ : torch.nn.Module
        Fitted network restored to the epoch with the best validation loss.
    history_ : list of dict
        Training and validation loss for each epoch.
    device_ : torch.device
        Resolved training device.
    patch_len_ : int
        Patch length actually used, after clamping to the series length.
    stride_ : int
        Patch stride actually used, after clamping to ``patch_len_``.
    best_epoch_ : int
        One-based index of the retained epoch.
    best_validation_loss_ : float
        Loss used to choose the retained epoch.

    References
    ----------
    .. [1] Wei, Y., et al. "PatchMTSC: patch-based multivariate time series
       classification", 2025.

    Examples
    --------
    >>> from aeon.testing.data_generation import make_example_3d_numpy
    >>> from multiverse.classification import PatchMTSCClassifier
    >>> X, y = make_example_3d_numpy(n_cases=8, n_channels=2, n_timepoints=20)
    >>> clf = PatchMTSCClassifier(  # doctest: +SKIP
    ...     n_epochs=1, patch_len=4, stride=2, device="cpu"
    ... )
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
        emb_size=16,
        d_model_patch=16,
        dim_ff=256,
        num_heads=8,
        patch_len=16,
        stride=8,
        dropout=0.01,
        head_dropout=0.0,
        pap_dropout=0.0,
        graph_decay=5e-4,
        n_epochs=100,
        batch_size=16,
        learning_rate=1e-3,
        validation_size=0.2,
        gradient_clip_norm=None,
        device="auto",
        num_workers=0,
        verbose=False,
        random_state=1234,
    ):
        self.emb_size = emb_size
        self.d_model_patch = d_model_patch
        self.dim_ff = dim_ff
        self.num_heads = num_heads
        self.patch_len = patch_len
        self.stride = stride
        self.dropout = dropout
        self.head_dropout = head_dropout
        self.pap_dropout = pap_dropout
        self.graph_decay = graph_decay
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
        if self.emb_size % 4 != 0:
            raise ValueError("emb_size must be divisible by 4")
        self._validate_positive_int(self.d_model_patch, "d_model_patch")
        if self.d_model_patch % 2 != 0:
            raise ValueError("d_model_patch must be even for tAPE")
        self._validate_positive_int(self.num_heads, "num_heads")
        if self.d_model_patch % self.num_heads != 0:
            raise ValueError("d_model_patch must be divisible by num_heads")
        self._validate_positive_int(self.dim_ff, "dim_ff")
        self._validate_positive_int(self.patch_len, "patch_len")
        self._validate_positive_int(self.stride, "stride")
        for name in ["dropout", "head_dropout", "pap_dropout"]:
            self._validate_dropout(getattr(self, name), name)

    def _setup_fit(self, X):
        # The paper's default patch length is 16, but some Multiverse-core
        # problems are shorter. Clamp the patch geometry to the observed series
        # rather than rejecting an otherwise valid dataset.
        self.patch_len_ = min(self.patch_len, self.n_timepoints_)
        self.stride_ = min(self.stride, self.patch_len_)

    def _build_model(self):
        return _PatchMTSCNetwork(
            n_channels=self.n_channels_,
            n_timepoints=self.n_timepoints_,
            n_classes=self.n_classes_,
            emb_size=self.emb_size,
            num_heads=self.num_heads,
            dim_ff=self.dim_ff,
            d_model_patch=self.d_model_patch,
            patch_len=self.patch_len_,
            stride=self.stride_,
            dropout=self.dropout,
            head_dropout=self.head_dropout,
            pap_dropout=self.pap_dropout,
            graph_decay=self.graph_decay,
        )

    def _build_optimizer(self):
        return torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)


    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return a small parameter set for aeon estimator checks."""
        return {
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
        }
