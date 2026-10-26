import pytest

from sentinel.data import generate
from sentinel.detectors import build_detectors


@pytest.fixture(scope="session")
def ds():
    return generate(seed=42)


@pytest.fixture(scope="session")
def dets(ds):
    return build_detectors(ds.train)
