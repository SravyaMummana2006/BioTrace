"""BIOTrace AI Engine - Biological Intelligence & Media Forensics Backend."""

import os
import tempfile

# Ensure writable Matplotlib cache directory before any submodules import matplotlib
if not os.environ.get("MPLCONFIGDIR"):
    _mpl_cache = os.path.join(tempfile.gettempdir(), "biotrace_mpl")
    try:
        os.makedirs(_mpl_cache, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = _mpl_cache
    except Exception:
        pass

__version__ = "0.1.0"
