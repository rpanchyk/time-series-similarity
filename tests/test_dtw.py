"""DTW metric tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas.testing as pdt
import pytest

from src.metrics import dtw_distance, pairwise_z_dtw
from src.prepare import load_ohlc
from src.similarity import find_similar, find_similar_many


def test_dtw_identical_is_zero():
    x = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    assert dtw_distance(x, x) == 0.0


def test_dtw_handles_time_shift_better_than_pointwise():
    """Shifted hump: DTW cost stays small; pointwise L2 does not."""
    a = np.array([0.0, 0.0, 1.0, 2.0, 1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 2.0, 1.0, 0.0, 0.0, 0.0])
    dtw = dtw_distance(a, b)
    l2 = float(np.linalg.norm(a - b))
    assert dtw < l2
    assert dtw < 2.0


def test_pairwise_z_dtw_ranks_identical_first():
    q = np.array([1.0, 2.0, 3.0, 2.5, 2.0])
    cands = np.vstack(
        [
            q * 10.0 + 5.0,  # same shape after z-norm
            np.array([5.0, 4.0, 3.0, 2.0, 1.0]),
            np.linspace(0, 1, 5),
        ]
    )
    dists = pairwise_z_dtw(q, cands)
    assert np.argmin(dists) == 0
    assert dists[0] < 1e-9


def test_find_similar_dtw_on_fixture(sample_ohlc_1h: Path):
    ohlc = load_ohlc(path=sample_ohlc_1h)
    results = find_similar(
        ohlc,
        window=5,
        stride=1,
        top_k=3,
        query_start=10,
        horizon=5,
        price="close",
        search_type="dtw",
    )
    assert len(results) == 3
    assert list(results["search_type"].unique()) == ["dtw"]
    assert results["distance"].is_monotonic_increasing
    # planted duplicate pattern at 40 should be a strong match
    assert int(results.iloc[0]["match_start"]) == 40


def test_find_similar_rejects_unknown_search_type(sample_ohlc_1h: Path):
    ohlc = load_ohlc(path=sample_ohlc_1h)
    with pytest.raises(ValueError, match="Unsupported search type"):
        find_similar(ohlc, window=5, horizon=5, query_start=10, search_type="nope")


def test_find_similar_many_parallel_matches_sequential(sample_ohlc_1h: Path):
    ohlc = load_ohlc(path=sample_ohlc_1h)
    types = ["z-euclidean", "pearson", "dtw"]
    kwargs = dict(
        window=5,
        stride=1,
        top_k=3,
        query_start=10,
        horizon=5,
        price="close",
    )
    sequential = find_similar_many(ohlc, types, workers=1, **kwargs)
    parallel = find_similar_many(ohlc, types, workers=2, **kwargs)
    assert [st for st, _ in sequential] == types
    assert [st for st, _ in parallel] == types
    for (_, seq), (_, par) in zip(sequential, parallel):
        pdt.assert_frame_equal(
            seq.reset_index(drop=True),
            par.reset_index(drop=True),
            check_dtype=False,
        )
