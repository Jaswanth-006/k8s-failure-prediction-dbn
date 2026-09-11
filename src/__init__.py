"""
PREFACE-DBN.

Importing dll_compat here is load-bearing on Windows, not tidiness. Several
modules in this package (inference_adapter, autoencoder_phase3, preface_baseline)
import pandas before torch, and pandas pulls in pyarrow, whose native libraries
prevent torch's c10.dll from initialising. Because Python runs this file before
any submodule body, doing it here guarantees torch loads first no matter which
`src.*` module the caller reaches for.

Entry-point scripts that import pandas at module level before touching `src`
still need their own explicit `import src.dll_compat` ahead of pandas.
"""

from . import dll_compat  # noqa: F401
