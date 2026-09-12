"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest


FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture
def sample_ticks_5dp(fixtures_dir: Path) -> Path:
    return fixtures_dir / "sample_ticks_5dp.csv"


@pytest.fixture
def sample_ticks_3dp(fixtures_dir: Path) -> Path:
    return fixtures_dir / "sample_ticks_3dp.csv"


@pytest.fixture
def sample_ohlc_1h(fixtures_dir: Path) -> Path:
    return fixtures_dir / "sample_ohlc_1h.csv"
