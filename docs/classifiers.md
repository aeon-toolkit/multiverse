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
    TimesURLClassifier,
    TS2VecClassifier,
    XCMClassifier,
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

## TimesURL

TimesURL is a self-supervised representation learner rather than an end-to-end
classifier. A contrastive objective pretrains an encoder on the training collection, the
collection is encoded, and a probe is fitted on those representations; at prediction time
the fitted encoder embeds the new series and the probe classifies them. Ported from the
authors' [implementation](https://github.com/Alrash/TimesURL).

The adapter follows the authors' UEA experiment: channels are standardised using
training data only, a normalised time coordinate is appended, and an RBF SVM is fitted to
the full-series representations. `eval_protocol` selects the probe, defaulting to their
`"svm"`, with `"linear"` and `"knn"` also available.

Liu, J. and Chen, S. "TimesURL: Self-supervised Contrastive Learning for Universal Time
Series Representation Learning." AAAI, 2024.

## TS2Vec

TS2Vec is a self-supervised representation learner: a hierarchical contrastive objective
pretrains an encoder, the collection is encoded to one vector per series, and a
classifier is fitted on the representations. It is the direct comparator to TimesURL,
which is built on its codebase. Ported from the authors'
[implementation](https://github.com/zhihanyue/ts2vec).

Yue, Z., Wang, Y., Duan, J., Yang, T., Huang, C., Tong, Y. and Xu, B. "TS2Vec: Towards
Universal Representation of Time Series." AAAI, 2022.

For UEA the authors evaluate with a support vector machine chosen by grid search over C
(`train.py` passes `eval_protocol='svm'`), so `probe="svm"` is the default. `probe=
"logistic"` selects their linear alternative, which is the probe TimesURL uses, so the
two encoders can be compared without the probe differing between them.

## XCM

XCM is an explainable convolutional network. Two parallel branches see the input
differently: a 2D branch convolves along time within each channel separately, so its
activations stay attributable to individual channels, and a 1D branch convolves along
time across all channels together. The branches are concatenated, passed through a
further 1D convolution, globally average pooled and classified. The channel
attributability of the 2D branch is what the paper's explanations rest on, so the
layer names it refers to are preserved. Ported from the authors'
[implementation](https://github.com/XAIseries/XCM).

Fauvel, K., Lin, T., Masson, V., Fromont, E. and Termier, A. "XCM: An Explainable
Convolutional Neural Network for Multivariate Time Series Classification."
Mathematics, 9(23), 2021.

This is the only Keras port here, following the authors, so it needs `tensorflow`
rather than `torch`. Both are in the `deep-learning` extra.

The XCM results in this repository follow the authors' tuning protocol. Section 4.3 sets
`window_size` and `batch_size` per dataset "by grid search based on the best average
accuracy following a stratified 5-fold cross-validation on the training set", over
windows {0.2, 0.4, 0.6, 0.8, 1.0} and batches {1, 8, 32}. Selection never touches the
test data.

The reported run searches the window on that grid and holds batch size at 32. That is
the one departure, and it is a cost decision rather than a modelling one: batch 1 takes
roughly 32 times the gradient steps, which would turn a day of GPU time into about 900
hours, for a value the published table selects on 4 of 30 datasets.

Both parameters accept a sequence, which triggers the search; a scalar fits once. The
class default is a single fit at **0.8**, the modal published window, because a default
should be cheap, but `XCM` in `tsml-eval` supplies the grid, and `XCM-Fixed` is the
single-fit variant kept for comparison:

```python
from multiverse.classification import XCMClassifier
from multiverse.classification._xcm import PAPER_WINDOW_SIZES, PAPER_BATCH_SIZES

XCMClassifier(window_size=PAPER_WINDOW_SIZES)                          # window only
XCMClassifier(window_size=PAPER_WINDOW_SIZES, batch_size=PAPER_BATCH_SIZES)  # full grid
```

The selected values are on `window_fraction_` and `batch_size_`, and every grid point's
mean and per-fold accuracy on `cv_results_`.

Cost is the reason the class default is a single fit. A fixed-window pass over
Multiverse-core took 1.2 GPU-hours in total; searching the window is five candidates
over five folds, about 20 times that.

That earlier fixed-window pass is what motivated the change. It averaged 0.699 across
the 23 datasets shared with the paper's table against their 0.761, and the paper itself
reports a mean relative accuracy drop of 7.0% +/- 1.3% from using a suboptimal window,
which is the size of the gap observed. Reporting a fixed window would have measured a
configuration the authors never used.

Because `window_size` is a fraction, the kernel grows with the series, and 0.8 of
EigenWorms' 17984 points is a 14387 point kernel. `max_window` bounds it at 100 points
and floors it at 1 for very short series. That bound is ours, not the authors': they run
kernels of this order, 40% of EigenWorms being 7193 points. Set `max_window=None` to
reproduce them, and expect the memory cost to follow.

## Notes on the ports

All classifiers in this package take aeon's ``numpy3D`` collections, shape
``(n_cases, n_channels, n_timepoints)``, and transpose internally where the original
network expects a different layout. Validation and epoch selection differ by port:

* ConvTran and PatchMTSC split off ``validation_size`` inside ``fit`` and restore the
  epoch with the lowest validation loss. With ``validation_size=0`` they select on
  training loss instead.
* TimesNet splits internally and restores the epoch with the best validation accuracy
  when a nonzero split is feasible. With ``validation_size=0`` (or fewer than two
  cases) it selects on training loss instead.
* DisjointCNN passes a sampled ``validation_size`` set to Keras, but samples with
  replacement from the training collection and leaves those cases in training. It
  restores the best monitored weights, but this is not a held-out validation split.
* XCM has no validation split or best-epoch restoration: it trains for a fixed number
  of epochs. Its optional cross-validation selects hyperparameters, not an epoch.
* TimesURL and TS2Vec pretrain on the full training collection and fit their probe on
  the resulting training representations; neither has an internal validation split or
  best-epoch restoration.

Thus no external validation split is required by these wrappers, but the same
validation procedure does not apply to every classifier. The networks require
``torch``, which is not a hard dependency of this package: install it with
``pip install aeon-multiverse[deep-learning]``.

### What was ported, and from where

| Classifier | Upstream source | Ported from |
|---|---|---|
| TimesNet | [thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library) | `models/TimesNet.py`, `layers/Embed.py`, `layers/Conv_Blocks.py` |
| ConvTran | [Navidfoumani/ConvTran](https://github.com/Navidfoumani/ConvTran), commit `148afb6` | `Models/model.py`, `Models/Attention.py`, `Models/AbsolutePositionalEncoding.py`, `Models/optimizers.py` |
| PatchMTSC | [YanxuanWei/PatchMTSC](https://github.com/YanxuanWei/PatchMTSC) | `Models/model.py`, `Models/Attention.py`, `Models/AbsolutePositionalEncoding.py` |
| TimesURL | [Alrash/TimesURL](https://github.com/Alrash/TimesURL) | the whole model package, vendored under `_timesurl_original` |
| TS2Vec | [zhihanyue/ts2vec](https://github.com/zhihanyue/ts2vec) | `ts2vec.py`, `utils.py`, `models/`, vendored under `_ts2vec_original` |
| XCM | [XAIseries/XCM](https://github.com/XAIseries/XCM) | `models/xcm.py` |

ConvTran, PatchMTSC and TimesNet are each a single self-contained module. TimesURL and
TS2Vec are vendored instead, under `_timesurl_original` and `_ts2vec_original`, and
driven by thin wrappers: they are whole model packages rather than single networks, at
about 2,300 and 630 lines respectively.

Note that TimesURL is a fork of TS2Vec, not a user of it. Its copies of `encoder.py` and
`losses.py` are modified: the loss calls mixup variants that TS2Vec does not have, and
the encoder carries a reconstruction head. The two vendored packages are therefore kept
separate rather than sharing a base. Only the components the published
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
| Sibling imports rewritten as relative imports | TimesURL | Upstream is laid out for ``sys.path`` insertion; as a subpackage it needs relative imports. A stray package-level ``from .encoder import TSEncoder`` is dropped, since ``encoder`` lives under ``models`` and the import only went unnoticed because the ``sys.path`` route bypassed it |
| One unconditional ``print`` silenced | TimesURL | ``lib.py`` printed the training tensor shape on every fit, which would pollute benchmark logs. ``verbose`` controls training output instead |
| Python's ``random`` seeded alongside numpy and torch | TimesURL | The authors' collator draws from ``random`` for segment masking and index shuffling, so seeding numpy and torch alone left runs irreproducible |
| Probe probabilities from ``decision_function`` | TimesURL | An ``SVC`` without ``probability=True`` cannot produce probability estimates, and aeon classifiers must implement ``predict_proba``. The decision scores are softmaxed instead, which avoids the internal cross-validation that Platt scaling would add |
| ``probability=True`` on the SVM probe | TS2Vec | The authors' grid sets it False, which leaves an ``SVC`` unable to produce probability estimates. aeon classifiers must implement ``predict_proba``, so it is enabled, adding Platt scaling fitted by internal cross-validation on the training data |
| Layer imports taken from ``tensorflow.keras.layers`` | XCM | The original imports ``Conv1D`` and ``Conv2D`` from ``keras.layers.convolutional``, a path removed in Keras 3. The layers and their arguments are unchanged |
| Kernel length floored at one point | XCM | The original computes ``int(window_size * n)``, which is zero for series shorter than five points and builds an invalid layer |
| Validation split moved inside ``fit`` | ConvTran, PatchMTSC, TimesNet | The originals split train/validation outside the model, which risks leakage between train and test. TSLib is explicit about it: ``exp_classification.py`` sets ``vali_data = self._get_data(flag='TEST')``, so it selects the retained epoch on the test set. See the note at the top of this page |
| Test data scaled with training statistics | TimesNet | TSLib fits its normaliser separately per split, so its test set is scaled by its own statistics. The port fits on train and applies to test |

Note that PatchMTSC's two graph blocks are, in the original, distinguished only by a
``stride`` argument that their ``forward`` never reads. They are therefore two
independently initialised copies of the same block whose outputs are concatenated. The
port preserves this.

TimesNet does not share the common training base class. Three parts of its procedure
are faithful to TSLib and would change published results if unified: it selects on
validation *accuracy* with early-stopping patience rather than validation loss
(``exp_classification.py`` calls ``early_stopping(-val_accuracy, ...)``), it uses RAdam,
and it standardises per channel (TSLib's ``Normalizer(norm_type='standardization')``).
It also seeds from ``random_state`` directly rather than through
``check_random_state``; that one is not a fidelity constraint, since TSLib simply sets a
global seed in ``run.py``, and it could be unified with the other two ports.

TimesNet reproduces TSLib's learning rate schedule, which the classification loop
applies every five epochs. With the default ``lr_adjust="type1"`` the rate becomes
``learning_rate * 0.5 ** (epoch - 1)`` at epochs 5, 10, 15 and so on, so from the
published ``learning_rate=0.001`` it falls to 6.3e-5 by epoch 5 and 6.1e-8 by epoch 15:
over a default 30 epoch run the model is effectively frozen well before the end. Pass
``lr_adjust=None`` to train at a constant rate instead. Omitting this schedule, as
earlier versions of this port did, is a materially different optimisation and makes
results incomparable with the published ones.

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
| TimesURL | vendored verbatim, so identical by construction | n/a |
| TS2Vec | vendored verbatim, so identical by construction | n/a |
| XCM | yes, the built graph matches layer for layer with equal parameter counts | n/a, Keras rather than torch |
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
export MULTIVERSE_TIMESURL_SRC=/path/to/TimesURL/model/modules
export MULTIVERSE_TS2VEC_SRC=/path/to/ts2vec
export MULTIVERSE_XCM_SRC=/path/to/XCM
pytest multiverse/classification/tests/test_original_equivalence.py -v
```

ConvTran and PatchMTSC additionally require ``einops`` to import the original modules;
the ports themselves do not.
