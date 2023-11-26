"""Top-level package for PyGridMet."""
from importlib.metadata import PackageNotFoundError, version

from pygridmet.core import GridMet
from pygridmet.exceptions import (
    InputRangeError,
    InputTypeError,
    InputValueError,
    MissingCRSError,
    MissingItemError,
)
from pygridmet.print_versions import show_versions
from pygridmet.pygridmet import get_bycoords, get_bygeom

try:
    __version__ = version("pygridmet")
except PackageNotFoundError:
    __version__ = "999"

__all__ = [
    "GridMet",
    "get_bycoords",
    "get_bygeom",
    "potential_et",
    "show_versions",
    "InputRangeError",
    "InputTypeError",
    "InputValueError",
    "MissingItemError",
    "MissingCRSError",
]
