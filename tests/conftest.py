from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fixtures.make_fixture import make_fixture  # noqa: E402

from ring_sentinel.config import load_config  # noqa: E402
from ring_sentinel.data.load import prepare  # noqa: E402
from ring_sentinel.data.split import temporal_split  # noqa: E402
from ring_sentinel.entities.resolve import add_entities  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def raw():
    """Prepared fixture frame: sorted by TransactionDT, day/hour derived."""
    return prepare(make_fixture())


@pytest.fixture(scope="session")
def entities(raw):
    return add_entities(raw)


@pytest.fixture(scope="session")
def splits(raw):
    return temporal_split(raw)
