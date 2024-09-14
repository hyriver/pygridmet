"""Configuration for pytest."""

from __future__ import annotations

import pytest
from click.testing import CliRunner


@pytest.fixture
def runner():
    """Return a CliRunner."""
    return CliRunner()


@pytest.fixture(autouse=True)
def _add_standard_imports(doctest_namespace):
    """Add pygridmet namespace for doctest."""
    import pygridmet as gridmet

    doctest_namespace["gridmet"] = gridmet
