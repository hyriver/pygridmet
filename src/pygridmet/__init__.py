"""Top-level package for PyGridMET."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from pygridmet import exceptions
from pygridmet.core import GridMET
from pygridmet.print_versions import show_versions
from pygridmet.pygridmet import get_bycoords, get_bygeom, get_conus

try:
    __version__ = version("pygridmet")
except PackageNotFoundError:
    __version__ = "999"

__all__ = [
    "GridMET",
    "get_bycoords",
    "get_bygeom",
    "get_conus",
    "show_versions",
    "exceptions",
    "__version__",
]
