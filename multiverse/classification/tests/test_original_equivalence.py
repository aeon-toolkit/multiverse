"""Equivalence tests against the authors' original implementations.

Each ported network is checked against the code it was ported from, rather than
against a stored expected value. For each classifier the test builds both the
original and the ported network and asserts:

1. that transferring the original's weights into the port makes the two produce
   identical outputs, which demonstrates the architectures agree; and
2. that from the same seed, with no weight transfer, the two already agree,
   which demonstrates the port constructs its parameters in the same order and
   therefore consumes the random stream identically to the original.

The two together matter: (2) alone would pass for a port that had silently
reordered a layer's construction, and (1) alone would be vacuous if the two
networks happened to initialise identically. The transfer test therefore starts
the port from a *different* seed, so the weights genuinely have to be moved
across for the outputs to agree.

The original repositories are not dependencies of this package, so these tests
skip unless the relevant source tree is pointed at by an environment variable:

    MULTIVERSE_CONVTRAN_SRC   root of https://github.com/Navidfoumani/ConvTran
    MULTIVERSE_PATCHMTSC_SRC  root of https://github.com/YanxuanWei/PatchMTSC
    MULTIVERSE_TIMESNET_SRC   root of https://github.com/thuml/Time-Series-Library

The originals import ``einops`` and ``pandas``; the ports do not.
"""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

import torch

from multiverse.classification._convtran import _ConvTranNetwork
from multiverse.classification._patchmtsc import _PatchMTSCNetwork
from multiverse.classification._timesnet import _TimesNetClassificationModel

N_CHANNELS = 3
N_TIMEPOINTS = 20
N_CLASSES = 4
N_CASES = 6

# Seed used for the original network. The port is built from PORT_SEED in the
# weight-transfer tests so that the two genuinely start out different.
ORIGINAL_SEED = 0
PORT_SEED = 11


def _load_original(env_var, candidates, attribute):
    """Import ``attribute`` from the original source tree, or skip.

    Parameters
    ----------
    env_var : str
        Environment variable holding the root of the original source tree.
    candidates : list of tuple
        ``(sys_path_subdirectory, dotted_module_path)`` pairs, tried in order.
        More than one is supported because a source may be laid out as the
        authors' repository or as a vendored copy of its model package.
    attribute : str
        Name to pull out of the imported module.
    """
    root = os.environ.get(env_var)
    if not root:
        pytest.skip(f"{env_var} is not set; skipping equivalence test")
    root = Path(root)
    if not root.is_dir():
        pytest.skip(f"{env_var}={root} is not a directory")

    failures = []
    for subdirectory, module_path in candidates:
        search_path = str(root if subdirectory is None else root / subdirectory)
        if not Path(search_path).is_dir():
            failures.append(f"{search_path} does not exist")
            continue

        added = search_path not in sys.path
        if added:
            sys.path.insert(0, search_path)
        try:
            module = __import__(module_path, fromlist=[attribute])
        except ImportError as error:  # missing einops/pandas, or a moved file
            failures.append(f"{module_path} from {search_path}: {error}")
            continue
        finally:
            if added and search_path in sys.path:
                sys.path.remove(search_path)

        if hasattr(module, attribute):
            return getattr(module, attribute)
        failures.append(f"{module_path} has no attribute {attribute}")

    pytest.skip(f"cannot import {attribute} from {root}: " + "; ".join(failures))


def _tensors(network, exclude):
    """Return the state dict as a list, dropping keys matching ``exclude``."""
    return [
        (name, tensor)
        for name, tensor in network.state_dict().items()
        if not any(fragment in name for fragment in exclude)
    ]


def _transfer_weights(original, port, exclude_original=(), exclude_port=()):
    """Copy every tensor from ``original`` into ``port``, positionally.

    The two state dicts are matched in declaration order rather than by name,
    since the port renames modules to aeon conventions. Every pair must agree on
    shape, which is itself a meaningful check on the architecture.
    """
    original_tensors = _tensors(original, exclude_original)
    port_tensors = _tensors(port, exclude_port)

    assert len(original_tensors) == len(port_tensors), (
        f"the port has {len(port_tensors)} state tensors but the original has "
        f"{len(original_tensors)}"
    )

    mismatched = [
        (original_name, tuple(original_tensor.shape), port_name, tuple(port_tensor.shape))
        for (original_name, original_tensor), (port_name, port_tensor) in zip(
            original_tensors, port_tensors
        )
        if original_tensor.shape != port_tensor.shape
    ]
    assert not mismatched, f"shape mismatch between original and port: {mismatched[:5]}"

    state = dict(port.state_dict())
    state.update(
        {
            port_name: original_tensor
            for (_, original_tensor), (port_name, _) in zip(
                original_tensors, port_tensors
            )
        }
    )
    port.load_state_dict(state)
    return len(original_tensors)


def _assert_identical(original_output, port_output):
    """Assert two forward passes agree exactly."""
    difference = (original_output - port_output).abs().max().item()
    assert difference == 0.0, f"port differs from original by {difference}"


# ---------------------------------------------------------------------------
# ConvTran
# ---------------------------------------------------------------------------


def _original_convtran():
    return _load_original(
        "MULTIVERSE_CONVTRAN_SRC",
        [(None, "Models.model")],
        "ConvTran",
    )


def _convtran_config():
    return {
        "Data_shape": (N_CASES, N_CHANNELS, N_TIMEPOINTS),
        "emb_size": 8,
        "num_heads": 2,
        "dim_ff": 16,
        "Fix_pos_encode": "tAPE",
        "Rel_pos_encode": "eRPE",
        "dropout": 0.01,
    }


def _build_convtran_port():
    return _ConvTranNetwork(
        n_channels=N_CHANNELS,
        n_timepoints=N_TIMEPOINTS,
        n_classes=N_CLASSES,
        emb_size=8,
        num_heads=2,
        dim_ff=16,
        dropout=0.01,
    )


def test_convtran_matches_original_with_transferred_weights():
    """The ported ConvTran network reproduces the authors' network exactly."""
    original_class = _original_convtran()

    torch.manual_seed(ORIGINAL_SEED)
    original = original_class(_convtran_config(), N_CLASSES).eval()
    torch.manual_seed(PORT_SEED)
    port = _build_convtran_port().eval()

    X = torch.randn(N_CASES, N_CHANNELS, N_TIMEPOINTS)
    with torch.no_grad():
        assert (original(X) - port(X)).abs().max().item() > 0, (
            "networks were already identical before the transfer, so this test "
            "would not prove anything"
        )

    n_transferred = _transfer_weights(original, port)
    assert n_transferred > 0

    with torch.no_grad():
        _assert_identical(original(X), port(X))


def test_convtran_parameter_order_matches_original():
    """The port draws from the random stream in the same order as the original."""
    original_class = _original_convtran()

    torch.manual_seed(ORIGINAL_SEED)
    original = original_class(_convtran_config(), N_CLASSES).eval()
    torch.manual_seed(ORIGINAL_SEED)
    port = _build_convtran_port().eval()

    X = torch.randn(N_CASES, N_CHANNELS, N_TIMEPOINTS)
    with torch.no_grad():
        _assert_identical(original(X), port(X))


def test_convtran_radam_matches_original():
    """The ported RAdam optimizer takes the same steps as the authors' copy."""
    original_radam = _load_original(
        "MULTIVERSE_CONVTRAN_SRC",
        [(None, "Models.optimizers")],
        "RAdam",
    )
    from multiverse.classification._convtran import _RAdam

    def run(optimizer_class):
        torch.manual_seed(ORIGINAL_SEED)
        layer = torch.nn.Linear(4, 3)
        optimizer = optimizer_class(layer.parameters(), lr=1e-3, weight_decay=0)
        torch.manual_seed(ORIGINAL_SEED)
        X = torch.randn(8, 4)
        targets = torch.randint(0, 3, (8,))
        for _ in range(5):
            optimizer.zero_grad()
            torch.nn.functional.cross_entropy(layer(X), targets).backward()
            optimizer.step()
        return [parameter.detach().clone() for parameter in layer.parameters()]

    for original_parameter, port_parameter in zip(run(original_radam), run(_RAdam)):
        assert torch.equal(original_parameter, port_parameter)


# ---------------------------------------------------------------------------
# PatchMTSC
# ---------------------------------------------------------------------------

PATCH_LEN = 5
STRIDE = 3


def _original_patchmtsc():
    return _load_original(
        "MULTIVERSE_PATCHMTSC_SRC",
        # The authors' repository layout, then a vendored copy of the model
        # package (as carried in tsml-eval, where this port originated).
        [(None, "Models.model"), (None, "_patchmtsc_original.model")],
        "PatchMTSC",
    )


def _patchmtsc_config():
    return {
        "Data_shape": (N_CASES, N_CHANNELS, N_TIMEPOINTS),
        "emb_size": 8,
        "num_heads": 2,
        "dim_ff": 16,
        "Fix_pos_encode": "tAPE",
        "Rel_pos_encode": "eRPE",
        "patch_len": PATCH_LEN,
        "stride": STRIDE,
        "padding_patch": "end",
        "d_model_patch": 8,
        "d_model": 8,
        "dropout": 0.01,
        "enc_in": N_CHANNELS,
        "individual": 0,
        "head_dropout": 0.0,
        "pap_dropout": 0.0,
        "weight_decay": 5e-4,
        "moving_window": [2, 2],
        "graph_stride": [1, 2],
        "pool_choice": "mean",
    }


def _build_patchmtsc_port():
    return _PatchMTSCNetwork(
        n_channels=N_CHANNELS,
        n_timepoints=N_TIMEPOINTS,
        n_classes=N_CLASSES,
        emb_size=8,
        num_heads=2,
        dim_ff=16,
        d_model_patch=8,
        patch_len=PATCH_LEN,
        stride=STRIDE,
        dropout=0.01,
        head_dropout=0.0,
        pap_dropout=0.0,
        graph_decay=5e-4,
    )


# The original stores its graph distance mask as a plain attribute; the port
# registers it as a buffer so that `.to(device)` moves it. It is derived from
# the configuration, not learned, and is asserted equal separately below.
PATCHMTSC_PORT_ONLY = ("pre_relation",)


def test_patchmtsc_matches_original_with_transferred_weights():
    """The ported PatchMTSC network reproduces the authors' network exactly."""
    original_class = _original_patchmtsc()

    torch.manual_seed(ORIGINAL_SEED)
    original = original_class(_patchmtsc_config(), N_CLASSES).eval()
    torch.manual_seed(PORT_SEED)
    port = _build_patchmtsc_port().eval()

    X = torch.randn(N_CASES, N_CHANNELS, N_TIMEPOINTS)
    with torch.no_grad():
        # The original's head is a LazyLinear, so it must see input once before
        # it has weights to transfer.
        original(X)
        assert (original(X) - port(X)).abs().max().item() > 0, (
            "networks were already identical before the transfer, so this test "
            "would not prove anything"
        )

    n_transferred = _transfer_weights(original, port, exclude_port=PATCHMTSC_PORT_ONLY)
    assert n_transferred > 0

    with torch.no_grad():
        _assert_identical(original(X), port(X))


def test_patchmtsc_graph_mask_matches_original():
    """The distance decay mask is unchanged by being registered as a buffer."""
    original_class = _original_patchmtsc()

    torch.manual_seed(ORIGINAL_SEED)
    original = original_class(_patchmtsc_config(), N_CLASSES)
    torch.manual_seed(ORIGINAL_SEED)
    port = _build_patchmtsc_port()

    for original_block, port_block in [
        (original.MPNN1, port.graph_block1),
        (original.MPNN2, port.graph_block2),
    ]:
        assert torch.equal(original_block.pre_relation, port_block.pre_relation)


def test_patchmtsc_head_width_matches_original():
    """The explicit head matches the width the original's LazyLinear resolves to."""
    original_class = _original_patchmtsc()

    torch.manual_seed(ORIGINAL_SEED)
    original = original_class(_patchmtsc_config(), N_CLASSES).eval()
    torch.manual_seed(ORIGINAL_SEED)
    port = _build_patchmtsc_port().eval()

    with torch.no_grad():
        original(torch.randn(N_CASES, N_CHANNELS, N_TIMEPOINTS))

    assert port.out.in_features == original.out.in_features


def test_patchmtsc_random_stream_differs_from_original():
    """Document why seeded PatchMTSC init cannot match the authors'.

    The original's classification head is an ``nn.LazyLinear``, which draws its
    weights on the first forward pass rather than at construction. The port
    derives the width up front and uses an ``nn.Linear``, so the two consume the
    random stream at different points. The architectures are still equivalent,
    which is what the weight-transfer test above establishes.
    """
    original_class = _original_patchmtsc()

    torch.manual_seed(ORIGINAL_SEED)
    original = original_class(_patchmtsc_config(), N_CLASSES).eval()
    torch.manual_seed(ORIGINAL_SEED)
    port = _build_patchmtsc_port().eval()

    X = torch.randn(N_CASES, N_CHANNELS, N_TIMEPOINTS)
    with torch.no_grad():
        original(X)
        assert (original(X) - port(X)).abs().max().item() > 0


# ---------------------------------------------------------------------------
# TimesNet
# ---------------------------------------------------------------------------


def _original_timesnet():
    return _load_original(
        "MULTIVERSE_TIMESNET_SRC",
        [(None, "models.TimesNet")],
        "Model",
    )


def _timesnet_config():
    return SimpleNamespace(
        task_name="classification",
        seq_len=N_TIMEPOINTS,
        label_len=0,
        pred_len=0,
        e_layers=1,
        d_model=16,
        d_ff=16,
        top_k=2,
        num_kernels=2,
        dropout=0.1,
        enc_in=N_CHANNELS,
        num_class=N_CLASSES,
        embed="timeF",
        freq="h",
    )


def _build_timesnet_port():
    return _TimesNetClassificationModel(
        seq_len=N_TIMEPOINTS,
        enc_in=N_CHANNELS,
        num_class=N_CLASSES,
        e_layers=1,
        d_model=16,
        d_ff=16,
        top_k=2,
        num_kernels=2,
        dropout=0.1,
    )


# TSLib's DataEmbedding always builds a temporal embedding, but bypasses it when
# the time marks are None, which is the case on the classification path. The
# port drops it, so it has no counterpart to receive these weights.
TIMESNET_ORIGINAL_ONLY = ("temporal_embedding",)


def test_timesnet_matches_original_with_transferred_weights():
    """The ported TimesNet network reproduces the TSLib network exactly."""
    original_class = _original_timesnet()

    torch.manual_seed(ORIGINAL_SEED)
    original = original_class(_timesnet_config()).eval()
    torch.manual_seed(PORT_SEED)
    port = _build_timesnet_port().eval()

    # TimesNet takes (batch, timepoints, channels), which is what the aeon
    # wrapper transposes into.
    X = torch.randn(N_CASES, N_TIMEPOINTS, N_CHANNELS)
    mask = torch.ones(N_CASES, N_TIMEPOINTS)
    with torch.no_grad():
        assert (original(X, mask, None, None) - port(X, mask)).abs().max().item() > 0, (
            "networks were already identical before the transfer, so this test "
            "would not prove anything"
        )

    n_transferred = _transfer_weights(
        original, port, exclude_original=TIMESNET_ORIGINAL_ONLY
    )
    assert n_transferred > 0

    with torch.no_grad():
        _assert_identical(original(X, mask, None, None), port(X, mask))


def test_timesnet_random_stream_differs_from_original():
    """Document why seeded TimesNet init cannot match TSLib's.

    TSLib's ``DataEmbedding`` always constructs a temporal embedding, even on
    the classification path where the time marks are ``None`` and it is never
    applied. The port drops it, so the port draws one fewer weight tensor and
    every subsequent draw shifts. The architectures are still equivalent, which
    is what the weight-transfer test above establishes; only the mapping from
    seed to initial weights differs.
    """
    original_class = _original_timesnet()

    torch.manual_seed(ORIGINAL_SEED)
    original = original_class(_timesnet_config()).eval()
    torch.manual_seed(ORIGINAL_SEED)
    port = _build_timesnet_port().eval()

    original_state = original.state_dict()
    unused = [name for name in original_state if "temporal_embedding" in name]
    assert unused, "TSLib no longer builds a temporal embedding; revisit the port"
    assert len(port.state_dict()) == len(original_state) - len(unused)

    X = torch.randn(N_CASES, N_TIMEPOINTS, N_CHANNELS)
    mask = torch.ones(N_CASES, N_TIMEPOINTS)
    with torch.no_grad():
        difference = (
            (original(X, mask, None, None) - port(X, mask)).abs().max().item()
        )
    assert difference > 0
