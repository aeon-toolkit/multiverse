"""Vendored copy of the authors' TimesURL implementation.

Imported as a proper subpackage with relative imports. The upstream code is laid
out for ``sys.path`` insertion and imports its siblings absolutely; those imports
are rewritten here, and the stray ``from .encoder import TSEncoder`` that the
upstream copy carries at this level is dropped, since ``encoder`` lives under
``models``. No other changes.
"""
