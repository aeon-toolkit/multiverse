# Multivariate time series classification algorithms

A wide range of classifiers are available in ``aeon``. You can list all those capable of 
learning from multiverse data using the 

```python
from aeon.utils.discovery import all_estimators
all = all_estimators("classifier", tag_filter={"capability:multivariate": True})
```
you can also filter on the capability to handle unequal length multivariate like so
```python
from aeon.utils.discovery import all_estimators
all = all_estimators("classifier", tag_filter={"capability:multivariate": True, "unequal_length":True})
```
there is extensive documentation with references about these classifiers in ``aeon``.

# Wrapped classifiers in this package

Some classifiers, particularly deep learning, are not implemented in aeon and do not
have a scikit learn compatible interface. One reason for this is that they tend to separate training and validation
datasets external to the fit function. We believe this increases the danger of leakage between train and test. Hence,
we have wrapped some of the classifiers and encapsulated validation as an option in ``fit``.
These are stored in the multiverse package ``multiverse.classification``.

```python
from multiverse.classification import (
    ConvTranClassifier,
    PatchMTSCClassifier,
    TimesNetClassifier,
)
```

## TimesNet

TimesNet is frequently used as a benchmark. It finds the dominant periods of a series
with an FFT, folds the series into a stack of 2D representations, and applies
inception-style 2D convolutions in stacked TimesBlocks. Ported from the THUML
[Time-Series-Library](https://github.com/thuml/Time-Series-Library).

Wu, H., Hu, T., Liu, Y., Zhou, H., Wang, J., and Long, M. "TimesNet: Temporal
2D-Variation Modeling for General Time Series Analysis." ICLR, 2023.

## ConvTran

ConvTran combines a convolutional front end with a Transformer encoder, and introduces
two positional encodings designed for time series: time absolute position encoding
(tAPE) and efficient relative position encoding (eRPE). Ported from the authors'
[implementation](https://github.com/Navidfoumani/ConvTran).

Foumani, N. M., Tan, C. W., Webb, G. I., and Salehi, M. "Improving position encoding of
transformers for multivariate time series classification." Data Mining and Knowledge
Discovery 38, 22-48, 2024.

## PatchMTSC

PatchMTSC mixes channels with a convolutional front end, segments the result into
overlapping patches, encodes each channel-independent patch sequence with a Transformer
using tAPE and eRPE, and models dependencies across patches with fully connected graph
blocks before patch average pooling. Ported from the authors'
[implementation](https://github.com/YanxuanWei/PatchMTSC).

Wei, Y., et al. "PatchMTSC: patch-based multivariate time series classification", 2025.

## Notes on the ports

All three wrappers take aeon's ``numpy3D`` collections, shape
``(n_cases, n_channels, n_timepoints)``, and transpose internally where the original
network expects a different layout. Each holds out ``validation_size`` of the training
data inside ``fit`` and restores the best epoch, so no external validation split is
required. The networks require ``torch``, which is not a hard dependency of this
package: install it with ``pip install aeon-multiverse[deep-learning]``.

### What was ported, and from where

| Classifier | Upstream source | Ported from |
|---|---|---|
| TimesNet | [thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library) | `models/TimesNet.py`, `layers/Embed.py`, `layers/Conv_Blocks.py` |
| ConvTran | [Navidfoumani/ConvTran](https://github.com/Navidfoumani/ConvTran), commit `148afb6` | `Models/model.py`, `Models/Attention.py`, `Models/AbsolutePositionalEncoding.py`, `Models/optimizers.py` |
| PatchMTSC | [YanxuanWei/PatchMTSC](https://github.com/YanxuanWei/PatchMTSC) | `Models/model.py`, `Models/Attention.py`, `Models/AbsolutePositionalEncoding.py` |

Each port is a single self-contained module. Only the components the published
architecture actually reaches are reproduced: for ConvTran and PatchMTSC that is the
``tAPE`` fixed encoding and ``eRPE`` relative encoding path, so the alternative
encodings (``Sin``, ``Learn``, ``Vector``) and the unused ``Transformer`` and
``CasualConvTran`` variants are not carried over.

ConvTran and PatchMTSC share the tAPE and eRPE modules, and both sets of authors use the
same training procedure, so those live once in ``multiverse/classification/_torch_base.py``
rather than being duplicated per classifier.

### Deliberate deviations from the original code

These are the only changes that touch behaviour or interface. Everything else is a
faithful transcription.

| Change | Where | Why it is safe |
|---|---|---|
| ``einops.rearrange`` replaced by ``reshape``/``permute`` | eRPE attention | Removes the ``einops`` dependency; the two calls are pure axis manipulations |
| ``nn.LazyLinear`` head replaced by ``nn.Linear`` | PatchMTSC | The original hard-codes a 512-wide layer; the width is now derived from the architecture (``emb_size * 2 * d_model_patch``), supporting arbitrary channel counts and patch lengths |
| ``patch_len`` and ``stride`` clamped to the series length | PatchMTSC | The paper's default ``patch_len=16`` exceeds the length of some Multiverse-core series, which would otherwise fail outright. Recorded on the fitted estimator as ``patch_len_`` and ``stride_`` |
| Graph mask decay renamed ``weight_decay`` to ``graph_decay`` | PatchMTSC | The original config key is misleading: the value is an exponential distance-decay applied to the constructed graph, not an optimizer setting |
| Dead ``dropout`` and ``graph_stride`` attributes removed | PatchMTSC network | Constructed by the original but never read in its ``forward``. ``nn.Dropout`` holds no parameters, so neither affects results or the state dict |
| Validation split moved inside ``fit`` | all three | The originals split train/validation outside the model, which risks leakage between train and test. See the note at the top of this page |

Note that PatchMTSC's two graph blocks are, in the original, distinguished only by a
``stride`` argument that their ``forward`` never reads. They are therefore two
independently initialised copies of the same block whose outputs are concatenated. The
port preserves this.

TimesNet does not share the common training base class. Its procedure differs in four
ways that would change published results if unified: it seeds from ``random_state``
directly rather than through ``check_random_state``, selects on validation *accuracy*
with early-stopping patience rather than validation loss, uses RAdam, and standardises
per channel.

### Equivalence testing

``multiverse/classification/tests/test_original_equivalence.py`` checks each port
against the authors' own code rather than against a stored expected value. For each
classifier it builds the original network and the ported network, transfers the
original's weights into the port, and asserts the two produce identical outputs. The
port is deliberately built from a *different* seed first, and the test asserts the two
disagree before the transfer, so it cannot pass vacuously.

**All three ports reproduce their original network exactly: maximum absolute difference
``0.0``.** ConvTran's ported RAdam optimizer is also checked to take identical steps to
the authors' copy.

Weight-for-weight equivalence is not the same as seed-for-seed equivalence, and the
tests record which ports have which:

| Port | Same architecture | Same initial weights from the same seed |
|---|---|---|
| ConvTran | yes | yes |
| PatchMTSC | yes | no, the original's head is an ``nn.LazyLinear`` that draws its weights on the first forward pass rather than at construction |
| TimesNet | yes | no, TSLib always builds a temporal embedding that the classification path never applies; dropping it removes one weight tensor and shifts every later draw |

In both negative cases the divergence is in *when* the random stream is consumed, not in
what the network computes. Seeded results from these two ports are therefore reproducible
run to run, but are not expected to match a seeded run of the original code.

These tests need the original repositories, which are not dependencies of this package.
They skip unless the relevant path is provided:

```bash
export MULTIVERSE_CONVTRAN_SRC=/path/to/ConvTran
export MULTIVERSE_PATCHMTSC_SRC=/path/to/PatchMTSC
export MULTIVERSE_TIMESNET_SRC=/path/to/Time-Series-Library
pytest multiverse/classification/tests/test_original_equivalence.py -v
```

ConvTran and PatchMTSC additionally require ``einops`` to import the original modules;
the ports themselves do not.
