"""Shared test fixtures.

Tests that need the processed data, checkpoints and saved predictions run only when SITESEER_DATA_ROOT points at a
directory laid out like the Zenodo bundle (dataset/, data/, checkpoints/, results/); they are skipped otherwise.
They read from it and never write into it.
"""
import os

import pytest

DATA_ROOT = os.environ.get("SITESEER_DATA_ROOT")


@pytest.fixture
def data_root():
    if not DATA_ROOT or not os.path.isdir(DATA_ROOT):
        pytest.skip("SITESEER_DATA_ROOT not set")
    return DATA_ROOT


def need(root: str, rel: str) -> str:
    """Path of rel under the data root; skips the test if it is not there."""
    p = os.path.join(root, rel)
    if not os.path.exists(p):
        pytest.skip(f"{rel} not under {root}")
    return p
