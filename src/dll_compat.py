"""
Windows DLL load-order workaround. Import this before pandas.

Symptom
-------
    OSError: [WinError 1114] A dynamic link library (DLL) initialization
    routine failed. Error loading "...\\torch\\lib\\c10.dll"

Cause
-----
`pandas` imports `pyarrow`, and pyarrow ships native libraries that clash with
torch's when they are loaded first. The failure depends purely on import order:

    import torch                  -> fine
    import numpy;  import torch   -> fine
    import pandas; import torch   -> WinError 1114
    import pyarrow; import torch  -> WinError 1114

`KMP_DUPLICATE_LIB_OK=TRUE` does not help, so this is not the usual duplicate
OpenMP runtime problem.

Fix
---
Import torch first. This module does that and nothing else, so a script only has
to place `import src.dll_compat` above its pandas import:

    import os, sys
    sys.path.insert(0, <repo root>)
    import src.dll_compat   # noqa: F401  - must precede pandas
    import pandas as pd

The import is wrapped in try/except because several scripts (synthetic modes,
the evaluator) run fine without torch installed, and a missing torch must not
turn into an import error here. Non-Windows platforms are unaffected and skip
the work entirely.
"""

import sys

torch_available = False

if sys.platform == "win32":
    try:
        import torch  # noqa: F401

        torch_available = True
    except Exception:
        # Either torch is absent, or it is genuinely broken. Both are the
        # caller's problem to report, not this shim's.
        torch_available = False
else:
    try:
        import torch  # noqa: F401

        torch_available = True
    except Exception:
        torch_available = False
