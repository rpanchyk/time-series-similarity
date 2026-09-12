"""Similarity metric unit and integration tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from main import parse_args
from src.config import SEARCH_TYPES
from src.metrics import (
    candle_shape_vector,
    dtw_multiv_distance,
    ohlc_joint_vector,
    pairwise_cid,
    pairwise_cosine,
    pairwise_distances,
    pairwise_euclidean,
    pairwise_pearson,
    pairwise_spearman,
    pairwise_z_euclidean,
    pairwise_z_manhattan,
    softdtw_distance,
    to_logret_windows,
    z_normalize,
)
from src.prepare import load_ohlc
from src.similarity import find_similar


EXTENDED_METRICS = (
    "cosine",
    "logret",
    "softdtw",
    "ohlc-joint",
    "candle-shape",
    "dtw-multiv",
    "spearman",
    "cid",
)


def test_cli_accepts_pearson_and_z_manhattan():
    for metric in ("pearson", "z-manhattan"):
        args = parse_args(
            [
                "--similarity",
                "--symbol",
                "EURUSD",
                "--timeframe",
                "1h",
                "--search-type",
                metric,
            ]
        )
        assert args.search_type == [metric]


def test_all_extended_metrics_registered():
    for m in EXTENDED_METRICS:
        assert m in SEARCH_TYPES


def test_cli_accepts_all_extended_metrics():
    for m in EXTENDED_METRICS:
        args = parse_args(
            [
                "--similarity",
                "--symbol",
                "EURUSD",
                "--timeframe",
                "1h",
                "--search-type",
                m,
            ]
        )
        assert args.search_type == [m]


def test_pairwise_pearson_identical_is_zero():
    q = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    cands = np.vstack([q * 2 + 5, np.linspace(0, 1, 5)])
    dists = pairwise_pearson(q, cands)
    assert dists[0] < 1e-9
    assert dists[1] > dists[0]


def test_pairwise_pearson_anticorrelated_near_two():
    q = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    anti = q[::-1]
    d = pairwise_pearson(q, anti.reshape(1, -1))[0]
    assert abs(d - 2.0) < 1e-9


def test_pairwise_z_manhattan_identical_shape_is_zero():
    q = np.array([1.0, 2.0, 3.0, 2.5, 2.0])
    cands = np.vstack([q * 10 + 3, np.array([5.0, 4.0, 3.0, 2.0, 1.0])])
    dists = pairwise_z_manhattan(q, cands)
    assert dists[0] < 1e-9
    assert np.argmin(dists) == 0


def test_pairwise_cosine_identical_is_zero():
    q = np.array([1.0, 2.0, 3.0, 4.0])
    cands = np.vstack([q * 3.0, np.array([4.0, 3.0, 2.0, 1.0])])
    d = pairwise_cosine(q, cands)
    assert d[0] < 1e-9
    assert d[1] > d[0]


def test_pairwise_spearman_monotonic():
    q = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mono = q**2
    anti = q[::-1]
    d = pairwise_spearman(q, np.vstack([mono, anti]))
    assert d[0] < 1e-9
    assert abs(d[1] - 2.0) < 1e-9


def test_pairwise_cid_identical_shape_small():
    q = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    same = q * 2 + 1
    other = np.linspace(0, 1, 5)
    d = pairwise_cid(q, np.vstack([same, other]))
    assert np.argmin(d) == 0


def test_logret_windows_and_distance():
    prices = np.array([[1.0, 1.1, 1.21], [2.0, 2.2, 2.42]])
    lr = to_logret_windows(prices)
    assert lr.shape == (2, 2)
    d = pairwise_distances("logret", lr[0], lr[1:])
    assert d[0] < 1e-9


def test_softdtw_identical_near_zero():
    x = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    y = np.array([1.0, 1.5, 3.5, 1.2, 0.5])
    assert softdtw_distance(x, x) < 1e-6
    assert softdtw_distance(x, y) > softdtw_distance(x, x)


def test_ohlc_joint_and_candle_shape_vectors():
    o = np.array([1.0, 1.1, 1.2, 1.15])
    h = np.array([1.05, 1.2, 1.25, 1.2])
    l = np.array([0.95, 1.05, 1.1, 1.1])
    c = np.array([1.02, 1.15, 1.18, 1.12])
    joint = ohlc_joint_vector(o, h, l, c, 0, 4)
    assert joint.shape == (8,)
    shape = candle_shape_vector(o, h, l, c, 0, 4)
    assert shape.shape == (12,)


def test_struct_metrics_use_plain_euclidean_not_double_znorm():
    """ohlc-joint / candle-shape must not apply a second z-norm pass."""
    # Two identical pre-normalized feature rows with non-unit scale/offset
    # would stay identical under plain Euclidean, but diverge if z-normed again
    # against a differently scaled third candidate — check dispatch path.
    q = np.array([0.0, 1.0, 0.0, 1.0], dtype=float)
    same = q.copy()
    scaled = q * 10.0  # different after z-norm of whole vector vs plain L2
    plain = pairwise_euclidean(q, np.vstack([same, scaled]))
    zed = pairwise_z_euclidean(q, np.vstack([same, scaled]))
    assert plain[0] < 1e-12
    assert plain[1] > 1.0
    assert zed[0] < 1e-12
    assert zed[1] < 1e-12  # scale-invariant under z-euclidean

    for metric in ("ohlc-joint", "candle-shape"):
        d = pairwise_distances(metric, q, np.vstack([same, scaled]))
        np.testing.assert_allclose(d, plain)


def test_dtw_multiv_identical_zero():
    x = np.column_stack(
        [
            z_normalize(np.array([1.0, 2.0, 3.0])),
            z_normalize(np.array([3.0, 2.0, 1.0])),
            z_normalize(np.array([1.0, 1.5, 1.0])),
            z_normalize(np.array([2.0, 2.5, 3.0])),
        ]
    )
    assert dtw_multiv_distance(x, x) == 0.0


def test_find_similar_pearson_on_fixture(sample_ohlc_1h: Path):
    ohlc = load_ohlc(path=sample_ohlc_1h)
    results = find_similar(
        ohlc,
        window=5,
        horizon=5,
        top_k=3,
        query_start=10,
        search_type="pearson",
    )
    assert len(results) == 3
    assert list(results["search_type"].unique()) == ["pearson"]
    assert results["distance"].is_monotonic_increasing
    assert int(results.iloc[0]["match_start"]) == 40


def test_find_similar_z_manhattan_on_fixture(sample_ohlc_1h: Path):
    ohlc = load_ohlc(path=sample_ohlc_1h)
    results = find_similar(
        ohlc,
        window=5,
        horizon=5,
        top_k=3,
        query_start=10,
        search_type="z-manhattan",
    )
    assert len(results) == 3
    assert list(results["search_type"].unique()) == ["z-manhattan"]
    assert results["distance"].is_monotonic_increasing
    assert int(results.iloc[0]["match_start"]) == 40


@pytest.mark.parametrize("metric", EXTENDED_METRICS)
def test_find_similar_each_extended_metric_on_fixture(
    metric: str, sample_ohlc_1h: Path
):
    ohlc = load_ohlc(path=sample_ohlc_1h)
    results = find_similar(
        ohlc,
        window=5,
        horizon=5,
        top_k=3,
        query_start=10,
        search_type=metric,
        price="close",
    )
    assert len(results) == 3
    assert list(results["search_type"].unique()) == [metric]
    assert results["distance"].is_monotonic_increasing


def test_find_similar_planted_match_for_vector_metrics(sample_ohlc_1h: Path):
    """Planted duplicate at 40 should rank first for shape-based 1D metrics."""
    ohlc = load_ohlc(path=sample_ohlc_1h)
    for metric in ("cosine", "spearman", "cid", "logret", "softdtw"):
        results = find_similar(
            ohlc,
            window=5,
            horizon=5,
            top_k=1,
            query_start=10,
            search_type=metric,
        )
        assert int(results.iloc[0]["match_start"]) == 40, metric
