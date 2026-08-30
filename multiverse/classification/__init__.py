"""Classifiers that are not currently in aeon.

Each classifier is imported or ported in and wrapped in the aeon interface.
"""

__all__ = [
    "ConvTranClassifier",
    "PatchMTSCClassifier",
    "TimesNetClassifier",
]

from multiverse.classification._convtran import ConvTranClassifier
from multiverse.classification._patchmtsc import PatchMTSCClassifier
from multiverse.classification._timesnet import TimesNetClassifier
