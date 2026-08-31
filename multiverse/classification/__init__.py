"""Classifiers that are not currently in aeon.

Each classifier is imported or ported in and wrapped in the aeon interface.
"""

__all__ = [
    "ConvTranClassifier",
    "PatchMTSCClassifier",
    "TimesNetClassifier",
    "TS2VecClassifier",
    "XCMClassifier",
    "TimesURLClassifier",
]

from multiverse.classification._convtran import ConvTranClassifier
from multiverse.classification._patchmtsc import PatchMTSCClassifier
from multiverse.classification._timesnet import TimesNetClassifier
from multiverse.classification._ts2vec import TS2VecClassifier
from multiverse.classification._xcm import XCMClassifier
from multiverse.classification._timesurl import TimesURLClassifier
