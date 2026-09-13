"""
Backtest which --search-type best predicts query horizon bias (TASK.md rule).

EURUSD 1h grid:
  - N random --start-date from history
  - windows: 5, 10, 20, 30, 40
  - --skip-weekends, --top-k 10, --horizon 20

--predict chooses how top-k becomes a forecast (see PREDICT_MODES / TASK.md).
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from src.config import DEFAULT_WORKERS, OUTPUT_DIR, SEARCH_TYPES, resolve_workers
from src.prepare import load_ohlc
from src.similarity import (
    _forward_stats,
    find_similar_many,
    segment_has_gap,
    weekend_gap_flags,
)

BIAS_MARGIN = 0.20  # percentage points; see TASK.md
DEFAULT_MARGIN_CONF = 0.20  # (n_best - n_second) / k for --predict margin
DEFAULT_WINDOWS = (5, 10, 20, 30, 40)
DEFAULT_EXCLUDE = ("softdtw", "dtw-multiv")

# All cheap post-aggregators over the same top-k matches.
PREDICT_MODES = (
    # label votes
    "mode",
    "soft",
    "margin",
    "distance-weighted",
    "softmax",
    "softmax-margin",
    "rank-weighted",
    "gaussian",
    "recency-weighted",
    # magnitude aggregates → bias rule
    "mean",
    "median",
    "trimmed-mean",
    "mean-weighted",
    "rank-mean-weighted",
    "gaussian-mean",
    "recency-mean-weighted",
    "quantile-75",
    "signed-score",
    "signed-median",
    # first extreme in horizon (own actual target)
    "first-touch",
    "first-touch-weighted",
    "first-touch-margin",
)

FIRST_TOUCH_PREDICTS = frozenset(
    {"first-touch", "first-touch-weighted", "first-touch-margin"}
)


def bias_from_up_down(up: float, down: float) -> str:
    """LONG / SHORT / SKIP from upside/downside magnitudes (percent)."""
    if not np.isfinite(up) or not np.isfinite(down):
        return "skip"
    up = abs(float(up))
    down = abs(float(down))
    if up - down > BIAS_MARGIN:
        return "long"
    if down - up > BIAS_MARGIN:
        return "short"
    return "skip"


def bias_from_devs(max_dev_high: float, max_dev_low: float) -> str:
    return bias_from_up_down(abs(float(max_dev_high)), abs(float(max_dev_low)))


def bias_from_signed(s: float) -> str:
    """s = up - down (percentage points)."""
    if not np.isfinite(s):
        return "skip"
    if s > BIAS_MARGIN:
        return "long"
    if s < -BIAS_MARGIN:
        return "short"
    return "skip"


def bias_counts(labels: list[str]) -> dict[str, int]:
    return {k: labels.count(k) for k in ("long", "skip", "short")}


def mode_bias(labels: list[str]) -> str:
    if not labels:
        return "skip"
    counts = bias_counts(labels)
    best_n = max(counts.values())
    for key in ("long", "short", "skip"):
        if counts[key] == best_n:
            return key
    return "skip"


def normalize_weights(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    s = float(w.sum())
    if s <= 0 or len(w) == 0:
        n = max(len(w), 1)
        return np.ones(len(w), dtype=float) / n if len(w) else w
    return w / s


def distance_weights(distances: np.ndarray) -> np.ndarray:
    d = np.asarray(distances, dtype=float)
    if len(d) == 0:
        return d
    inv = 1.0 / (d - np.nanmin(d) + 1e-6)
    return normalize_weights(inv)


def softmax_weights(distances: np.ndarray) -> np.ndarray:
    """Softmax on negative shifted distances (closer → higher weight)."""
    d = np.asarray(distances, dtype=float)
    if len(d) == 0:
        return d
    x = -(d - np.nanmin(d))
    scale = float(np.median(np.abs(x))) + 1e-6
    z = x / scale
    z = z - np.nanmax(z)
    return normalize_weights(np.exp(z))


def gaussian_weights(distances: np.ndarray) -> np.ndarray:
    d = np.asarray(distances, dtype=float)
    if len(d) == 0:
        return d
    d0 = d - np.nanmin(d)
    sigma = float(np.median(d0)) + 1e-6
    return normalize_weights(np.exp(-0.5 * (d0 / sigma) ** 2))


def rank_weights(n: int) -> np.ndarray:
    """Assume rows are distance-sorted ascending: rank1 (closest) heaviest."""
    if n <= 0:
        return np.zeros(0, dtype=float)
    return normalize_weights(np.arange(n, 0, -1, dtype=float))


def recency_weights(match_times, query_time, half_life_days: float = 365.0) -> np.ndarray:
    """More recent analogs (closer in calendar to query) weigh more."""
    qt = pd.Timestamp(query_time)
    ages = []
    for t in match_times:
        tt = pd.Timestamp(t)
        if pd.isna(tt):
            ages.append(np.inf)
        else:
            ages.append(max((qt - tt).total_seconds() / 86400.0, 0.0))
    ages = np.asarray(ages, dtype=float)
    return normalize_weights(np.exp(-ages / half_life_days))


def trimmed_mean(values: np.ndarray, trim_frac: float = 0.1) -> float:
    a = np.sort(np.asarray(values, dtype=float))
    n = len(a)
    if n == 0:
        return 0.0
    k = int(n * trim_frac)
    if 2 * k >= n:
        return float(np.mean(a))
    return float(np.mean(a[k : n - k]))


def first_touch_label(high_time, low_time) -> str:
    ht = pd.Timestamp(high_time)
    lt = pd.Timestamp(low_time)
    if pd.isna(ht) or pd.isna(lt):
        return "skip"
    if ht < lt:
        return "long"
    if lt < ht:
        return "short"
    return "skip"


def weighted_mode(labels: list[str], weights: np.ndarray) -> str:
    wsum = {"long": 0.0, "skip": 0.0, "short": 0.0}
    for lab, w in zip(labels, weights):
        wsum[lab] += float(w)
    best = max(wsum.values())
    for key in ("long", "short", "skip"):
        if wsum[key] == best:
            return key
    return "skip"


def weighted_probs(labels: list[str], weights: np.ndarray) -> dict[str, float]:
    w = normalize_weights(weights)
    out = {"long": 0.0, "skip": 0.0, "short": 0.0}
    for lab, wi in zip(labels, w):
        out[lab] += float(wi)
    return out


def margin_predict(labels: list[str], margin_conf: float) -> str:
    """Mode only if lead over #2 is >= margin_conf * k; else SKIP."""
    if not labels:
        return "skip"
    counts = bias_counts(labels)
    ordered = sorted(counts.values(), reverse=True)
    best_n = ordered[0]
    second_n = ordered[1] if len(ordered) > 1 else 0
    conf = (best_n - second_n) / len(labels)
    if conf < margin_conf:
        return "skip"
    return mode_bias(labels)


def margin_from_probs(probs: dict[str, float], margin_conf: float) -> tuple[str, float]:
    ordered = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    best_lab, best_p = ordered[0]
    second_p = ordered[1][1] if len(ordered) > 1 else 0.0
    conf = best_p - second_p
    if conf < margin_conf:
        return "skip", conf
    # tie-break directional preference among equal max
    best = max(probs.values())
    for key in ("long", "short", "skip"):
        if probs[key] == best:
            return key, conf
    return best_lab, conf


def predict_topk(
    *,
    method: str,
    highs: np.ndarray,
    lows: np.ndarray,
    distances: np.ndarray,
    labels: list[str],
    margin_conf: float,
    high_times=None,
    low_times=None,
    match_times=None,
    query_time=None,
    ft_labels: list[str] | None = None,
) -> tuple[str, dict]:
    """
    Returns (pred_bias, extras).

    extras may include signed_s, conf, up, down.
    """
    highs = np.asarray(highs, dtype=float)
    lows = np.asarray(lows, dtype=float)
    ups = np.abs(highs)
    downs = np.abs(lows)
    extras: dict = {}
    n = len(ups)

    def amp_from_weights(w: np.ndarray) -> str:
        up = float(np.sum(w * ups)) if n else 0.0
        down = float(np.sum(w * downs)) if n else 0.0
        extras.update(up=up, down=down, signed_s=up - down)
        return bias_from_up_down(up, down)

    if method == "mode":
        return mode_bias(labels), extras

    if method == "soft":
        return mode_bias(labels), extras

    if method == "mean":
        up = float(np.mean(ups)) if n else 0.0
        down = float(np.mean(downs)) if n else 0.0
        extras.update(up=up, down=down, signed_s=up - down)
        return bias_from_up_down(up, down), extras

    if method == "median":
        up = float(np.median(ups)) if n else 0.0
        down = float(np.median(downs)) if n else 0.0
        extras.update(up=up, down=down, signed_s=up - down)
        return bias_from_up_down(up, down), extras

    if method == "trimmed-mean":
        up = trimmed_mean(ups)
        down = trimmed_mean(downs)
        extras.update(up=up, down=down, signed_s=up - down)
        return bias_from_up_down(up, down), extras

    if method == "quantile-75":
        up = float(np.quantile(ups, 0.75)) if n else 0.0
        down = float(np.quantile(downs, 0.75)) if n else 0.0
        extras.update(up=up, down=down, signed_s=up - down)
        return bias_from_up_down(up, down), extras

    if method == "distance-weighted":
        return weighted_mode(labels, distance_weights(distances)), extras

    if method == "mean-weighted":
        return amp_from_weights(distance_weights(distances)), extras

    if method == "softmax":
        return weighted_mode(labels, softmax_weights(distances)), extras

    if method == "softmax-margin":
        probs = weighted_probs(labels, softmax_weights(distances))
        pred, conf = margin_from_probs(probs, margin_conf)
        extras["conf"] = conf
        return pred, extras

    if method == "rank-weighted":
        return weighted_mode(labels, rank_weights(n)), extras

    if method == "rank-mean-weighted":
        return amp_from_weights(rank_weights(n)), extras

    if method == "gaussian":
        return weighted_mode(labels, gaussian_weights(distances)), extras

    if method == "gaussian-mean":
        return amp_from_weights(gaussian_weights(distances)), extras

    if method == "recency-weighted":
        if match_times is None or query_time is None:
            raise ValueError("recency-weighted needs match_times and query_time")
        return weighted_mode(
            labels, recency_weights(match_times, query_time)
        ), extras

    if method == "recency-mean-weighted":
        if match_times is None or query_time is None:
            raise ValueError("recency-mean-weighted needs match_times and query_time")
        return amp_from_weights(recency_weights(match_times, query_time)), extras

    if method == "margin":
        pred = margin_predict(labels, margin_conf)
        counts = bias_counts(labels)
        ordered = sorted(counts.values(), reverse=True)
        conf = (ordered[0] - (ordered[1] if len(ordered) > 1 else 0)) / max(n, 1)
        extras["conf"] = conf
        return pred, extras

    if method == "signed-score":
        s = float(np.mean(ups - downs)) if n else 0.0
        extras["signed_s"] = s
        return bias_from_signed(s), extras

    if method == "signed-median":
        s = float(np.median(ups - downs)) if n else 0.0
        extras["signed_s"] = s
        return bias_from_signed(s), extras

    if method in FIRST_TOUCH_PREDICTS:
        ftl = ft_labels
        if ftl is None:
            if high_times is None or low_times is None:
                raise ValueError("first-touch needs high_times/low_times or ft_labels")
            ftl = [
                first_touch_label(ht, lt)
                for ht, lt in zip(high_times, low_times)
            ]
        if method == "first-touch":
            return mode_bias(ftl), extras
        if method == "first-touch-weighted":
            return weighted_mode(ftl, distance_weights(distances)), extras
        # first-touch-margin
        pred = margin_predict(ftl, margin_conf)
        counts = bias_counts(ftl)
        ordered = sorted(counts.values(), reverse=True)
        conf = (ordered[0] - (ordered[1] if len(ordered) > 1 else 0)) / max(
            len(ftl), 1
        )
        extras["conf"] = conf
        return pred, extras

    raise ValueError(f"Unknown --predict: {method}")


def query_actuals(
    ohlc: pd.DataFrame, query_start: int, window: int, horizon: int
) -> dict:
    match_end = query_start + window - 1
    fwd = _forward_stats(
        ohlc["close"].to_numpy(float),
        ohlc["high"].to_numpy(float),
        ohlc["low"].to_numpy(float),
        ohlc["datetime"].to_numpy(),
        match_end,
        horizon,
    )
    high = float(fwd["max_dev_high"])
    low = float(fwd["max_dev_low"])
    signed = abs(high) - abs(low)
    ft = first_touch_label(fwd["max_dev_high_time"], fwd["max_dev_low_time"])
    return {
        "amplitude": bias_from_devs(high, low),
        "first_touch": ft,
        "high": high,
        "low": low,
        "signed": signed,
        "high_time": fwd["max_dev_high_time"],
        "low_time": fwd["max_dev_low_time"],
    }


def sample_start_indices(
    ohlc: pd.DataFrame,
    *,
    n: int,
    max_window: int,
    horizon: int,
    skip_weekends: bool,
    rng: np.random.Generator,
    max_tries: int = 50_000,
) -> list[int]:
    """Uniform random bar starts valid for all windows up to max_window."""
    n_bars = len(ohlc)
    last = n_bars - max_window - horizon
    if last < 1:
        raise ValueError("OHLC too short for max_window/horizon")

    weekend_flags = weekend_gap_flags(ohlc["datetime"]) if skip_weekends else None
    chosen: list[int] = []
    seen: set[int] = set()
    tries = 0
    while len(chosen) < n and tries < max_tries:
        tries += 1
        idx = int(rng.integers(0, last + 1))
        if idx in seen:
            continue
        if skip_weekends and segment_has_gap(weekend_flags, idx, max_window):
            continue
        if skip_weekends and segment_has_gap(
            weekend_flags, idx, max_window + horizon
        ):
            continue
        seen.add(idx)
        chosen.append(idx)
    if len(chosen) < n:
        raise RuntimeError(
            f"Could only sample {len(chosen)}/{n} valid start indices "
            f"(skip_weekends={skip_weekends})"
        )
    return chosen


def rank_key_columns(predict: str) -> list[str]:
    """Primary/secondary sort columns for best search-type (desc)."""
    if predict == "soft":
        return ["soft_mean", "hit_rate"]
    if predict in ("signed-score", "signed-median"):
        return ["signed_agree_mean", "hit_rate", "soft_mean"]
    return ["hit_rate", "soft_mean"]


def parse_predict_modes(raw: str) -> list[str]:
    raw = str(raw).strip().lower()
    if raw in ("all", "*"):
        return list(PREDICT_MODES)
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError("--predict must be a mode, comma-list, or 'all'")
    unknown = sorted({p for p in parts if p not in PREDICT_MODES})
    if unknown:
        raise ValueError(
            f"Unknown --predict mode(s): {', '.join(unknown)}. "
            f"Choose from: {', '.join(PREDICT_MODES)} or 'all'"
        )
    seen: set[str] = set()
    out: list[str] = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def load_window_plan(path: str) -> dict[int, dict[str, list[str]]]:
    """
    Load per-window search_type → predict list.

    JSON shape:
      {"windows": {"10": [{"search_type": "...", "predict": ["mode", ...]}, ...]}}
    or flat: {"10": [{"search_type": "...", "predict": [...]}, ...]}
    """
    from pathlib import Path

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "windows" in raw:
        raw = raw["windows"]
    if not isinstance(raw, dict) or not raw:
        raise ValueError("window-plan must be a non-empty object keyed by window")

    plan: dict[int, dict[str, list[str]]] = {}
    for w_key, entries in raw.items():
        window = int(w_key)
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"window {window}: expected non-empty list of entries")
        by_type: dict[str, list[str]] = {}
        for i, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise ValueError(f"window {window}[{i}]: expected object")
            st = str(entry.get("search_type", "")).strip()
            if st not in SEARCH_TYPES:
                raise ValueError(
                    f"window {window}[{i}]: unknown search_type {st!r}"
                )
            preds = parse_predict_modes(
                ",".join(str(p) for p in entry.get("predict", []))
            )
            if not preds:
                raise ValueError(f"window {window}/{st}: empty predict list")
            if st in by_type:
                # merge unique predicts for same type
                seen = set(by_type[st])
                for p in preds:
                    if p not in seen:
                        by_type[st].append(p)
                        seen.add(p)
            else:
                by_type[st] = preds
        plan[window] = by_type
    return plan


def print_plan_pair_summary(df: pd.DataFrame) -> list[dict]:
    """Hit table for each (window, search_type, predict) cell."""
    ok = df[df["hit"].notna()].copy()
    if ok.empty:
        print("\n=== Plan pairs: no scored rows ===")
        return []
    g = (
        ok.groupby(["window", "search_type", "predict"], sort=True)
        .agg(
            hit_rate=("hit", "mean"),
            soft_mean=("soft", "mean"),
            n=("hit", "count"),
        )
        .reset_index()
    )
    print("\n=== Plan pairs (amplitude) ===")
    rows: list[dict] = []
    for r in g.itertuples(index=False):
        item = {
            "window": int(r.window),
            "search_type": r.search_type,
            "predict": r.predict,
            "hit_rate": round(float(r.hit_rate), 4),
            "soft_mean": round(float(r.soft_mean), 4),
            "n": int(r.n),
        }
        rows.append(item)
        print(
            f"  w={item['window']:<3d}  {item['search_type']:14s}  "
            f"{item['predict']:20s}  hit={item['hit_rate']:.0%}  "
            f"soft={item['soft_mean']:.2f}  n={item['n']}"
        )
    by_w = (
        ok.groupby("window")
        .agg(hit_rate=("hit", "mean"), soft_mean=("soft", "mean"), n=("hit", "count"))
        .reset_index()
    )
    print("\n=== Plan mean hit by window ===")
    for r in by_w.itertuples(index=False):
        print(
            f"  w={int(r.window):<3d}  hit={float(r.hit_rate):.0%}  "
            f"soft={float(r.soft_mean):.2f}  n={int(r.n)}"
        )
    overall_hit = float(ok["hit"].mean())
    print(f"\nOverall plan hit={overall_hit:.0%}  n={len(ok)}")
    return rows


PREDICT_HELP = {
    "mode": "mode of per-match amplitude bias labels",
    "soft": "rank by P(actual) among top-k; pred label still mode",
    "margin": "mode only if (n_best-n_second)/k >= margin_conf; else SKIP",
    "distance-weighted": "1/(d-dmin) weighted mode of bias labels",
    "softmax": "softmax(-distance) weighted mode of bias labels",
    "softmax-margin": "softmax probs; SKIP unless P1-P2 >= margin_conf",
    "rank-weighted": "rank-weighted mode (closest heaviest)",
    "gaussian": "Gaussian kernel on distance → weighted mode",
    "recency-weighted": "exp(-age/365d) weighted mode of bias labels",
    "mean": "bias(mean|high|, mean|low|)",
    "median": "bias(median|high|, median|low|)",
    "trimmed-mean": "bias(10% trimmed mean|high|, trimmed mean|low|)",
    "mean-weighted": "bias(distance-weighted mean|high|/|low|)",
    "rank-mean-weighted": "bias(rank-weighted mean|high|/|low|)",
    "gaussian-mean": "bias(Gaussian-weighted mean|high|/|low|)",
    "recency-mean-weighted": "bias(recency-weighted mean|high|/|low|)",
    "quantile-75": "bias(P75|high|, P75|low|)",
    "signed-score": "s=mean(|high|-|low|); bias(s); rank by -|s-s_actual|",
    "signed-median": "s=median(|high|-|low|); bias(s); rank by -|s-s_actual|",
    "first-touch": "mode of which extreme came first (own actual)",
    "first-touch-weighted": "distance-weighted mode of first-touch labels",
    "first-touch-margin": "first-touch mode with margin abstain",
}


def build_report_for_predict(
    df: pd.DataFrame,
    *,
    predict: str,
    meta_base: dict,
    margin_conf: float,
) -> dict:
    ok = df[(df["predict"] == predict) & df["hit"].notna()].copy()
    sort_cols = rank_key_columns(predict)

    def ranking_frame(g: pd.DataFrame) -> pd.DataFrame:
        agg = (
            g.groupby("search_type")
            .agg(
                hit_rate=("hit", "mean"),
                soft_mean=("soft", "mean"),
                signed_agree_mean=("signed_agree", "mean"),
                n=("hit", "count"),
            )
            .reset_index()
        )
        return agg.sort_values(sort_cols, ascending=[False] * len(sort_cols))

    per_date: dict[str, dict] = {}
    if not ok.empty:
        for date_i, g in ok.groupby("date_i"):
            start_date = g["start_date"].iloc[0]
            by_type = ranking_frame(g)
            best = by_type.iloc[0]
            per_date[str(int(date_i))] = {
                "start_date": start_date,
                "best_search_type": best["search_type"],
                "hit_rate": round(float(best["hit_rate"]), 4),
                "soft_mean": round(float(best["soft_mean"]), 4),
                "signed_agree_mean": (
                    round(float(best["signed_agree_mean"]), 4)
                    if pd.notna(best["signed_agree_mean"])
                    else None
                ),
                "ranking": [
                    {
                        "search_type": r.search_type,
                        "hit_rate": round(float(r.hit_rate), 4),
                        "soft_mean": round(float(r.soft_mean), 4),
                        "signed_agree_mean": (
                            round(float(r.signed_agree_mean), 4)
                            if pd.notna(r.signed_agree_mean)
                            else None
                        ),
                    }
                    for r in by_type.itertuples(index=False)
                ],
            }

    overall = ranking_frame(ok) if not ok.empty else pd.DataFrame()
    overall_best = (
        overall.iloc[0]["search_type"] if len(overall) else None
    )
    help_txt = PREDICT_HELP[predict]
    if predict == "margin":
        help_txt = (
            f"mode only if (n_best-n_second)/k >= {margin_conf}; else SKIP"
        )

    return {
        "meta": {
            **meta_base,
            "predict": predict,
            "predict_rule": help_txt,
            "margin_conf": margin_conf,
            "bias_margin": BIAS_MARGIN,
            "rank_by": sort_cols,
            "rule": (
                f"LONG if abs(high)-abs(low)>{BIAS_MARGIN}; "
                f"SHORT if abs(low)-abs(high)>{BIAS_MARGIN}; else SKIP. "
                f"Predict={predict}. "
                "soft = fraction of top-k matching actual."
            ),
        },
        "per_start_date": per_date,
        "overall": {
            "best_search_type": overall_best,
            "ranking": [
                {
                    "search_type": r.search_type,
                    "hit_rate": round(float(r.hit_rate), 4),
                    "soft_mean": round(float(r.soft_mean), 4),
                    "signed_agree_mean": (
                        round(float(r.signed_agree_mean), 4)
                        if pd.notna(r.signed_agree_mean)
                        else None
                    ),
                    "n": int(r.n),
                }
                for r in overall.itertuples(index=False)
            ]
            if len(overall)
            else [],
        },
    }


def print_predict_summary(report: dict) -> None:
    predict = report["meta"]["predict"]
    sort_cols = report["meta"]["rank_by"]
    per_date = report["per_start_date"]
    overall_best = report["overall"]["best_search_type"]
    print(f"\n=== Per start-date best (--predict {predict}) ===")
    for key in sorted(per_date, key=int):
        d = per_date[key]
        print(
            f"  date[{key}] {d['start_date']}: "
            f"{d['best_search_type']} "
            f"(hit_rate={d['hit_rate']:.0%}, soft={d['soft_mean']:.2f})"
        )
    print(f"\n=== Overall ranking (predict={predict}, by {sort_cols}) ===")
    for r in report["overall"]["ranking"]:
        print(
            f"  {r['search_type']:14s}  hit={r['hit_rate']:.0%}  "
            f"soft={r['soft_mean']:.2f}  n={r['n']}"
        )
    print(f"Overall best ({predict}): {overall_best}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Backtest search-type accuracy vs query horizon bias"
    )
    ap.add_argument("--symbol", type=str, default="EURUSD")
    ap.add_argument("--timeframe", type=str, default="1h")
    ap.add_argument("--n-dates", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--windows",
        type=str,
        default=",".join(str(w) for w in DEFAULT_WINDOWS),
        help="Comma-separated windows (default: 5,10,20,30,40); "
        "ignored when --window-plan is set",
    )
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--price", type=str, default="close")
    ap.add_argument(
        "--predict",
        type=str,
        default="mode",
        help="Forecast mode, comma-list, or 'all' "
        f"(modes: {', '.join(PREDICT_MODES)}); "
        "ignored when --window-plan is set",
    )
    ap.add_argument(
        "--window-plan",
        type=str,
        default=None,
        help="JSON plan: per-window list of "
        "{search_type, predict:[...]} (same dates for all cells)",
    )
    ap.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Output filename tag (default: derived from predict / 'plan')",
    )
    ap.add_argument(
        "--margin-conf",
        type=float,
        default=DEFAULT_MARGIN_CONF,
        help="For --predict margin: min (n_best-n_second)/k "
        f"(default: {DEFAULT_MARGIN_CONF})",
    )
    ap.add_argument(
        "--workers",
        type=str,
        default=DEFAULT_WORKERS,
        help="Parallel processes per window run (default: max)",
    )
    ap.add_argument(
        "--exclude",
        type=str,
        default=",".join(DEFAULT_EXCLUDE),
        help="Comma-separated types to skip "
        f"(default: {','.join(DEFAULT_EXCLUDE)}); "
        "ignored when --window-plan is set",
    )
    args = ap.parse_args()

    if args.margin_conf < 0 or args.margin_conf > 1:
        raise SystemExit("--margin-conf must be in [0, 1]")

    window_plan: dict[int, dict[str, list[str]]] | None = None
    if args.window_plan:
        try:
            window_plan = load_window_plan(args.window_plan)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise SystemExit(f"--window-plan: {exc}") from exc
        windows = sorted(window_plan)
        predict_modes = sorted(
            {
                p
                for by_type in window_plan.values()
                for preds in by_type.values()
                for p in preds
            }
        )
        search_types_all = sorted(
            {st for by_type in window_plan.values() for st in by_type}
        )
        excluded: set[str] = set()
    else:
        try:
            predict_modes = parse_predict_modes(args.predict)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        windows = [int(x.strip()) for x in args.windows.split(",") if x.strip()]
        if not windows:
            raise SystemExit("--windows must be non-empty")
        excluded = {p.strip() for p in args.exclude.split(",") if p.strip()}
        unknown = sorted(excluded - set(SEARCH_TYPES))
        if unknown:
            raise SystemExit(f"Unknown --exclude: {', '.join(unknown)}")
        search_types_all = sorted(st for st in SEARCH_TYPES if st not in excluded)
        if not search_types_all:
            raise SystemExit("No search types left after --exclude")

    max_window = max(windows)

    try:
        workers = resolve_workers(args.workers)
    except (TypeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    print(f"Loading {args.symbol} {args.timeframe}...")
    ohlc = load_ohlc(timeframe=args.timeframe, symbol=args.symbol)
    print(
        f"Loaded {len(ohlc):,} bars "
        f"({ohlc['datetime'].iloc[0]} -> {ohlc['datetime'].iloc[-1]})"
    )

    rng = np.random.default_rng(args.seed)
    start_indices = sample_start_indices(
        ohlc,
        n=args.n_dates,
        max_window=max_window,
        horizon=args.horizon,
        skip_weekends=True,
        rng=rng,
    )
    start_dates = [
        str(pd.Timestamp(ohlc["datetime"].iloc[i])) for i in start_indices
    ]
    print(f"seed={args.seed} random start-dates (shared across windows/types):")
    for i, (idx, ds) in enumerate(zip(start_indices, start_dates), start=1):
        print(f"  [{i}] index={idx}  {ds}")
    if window_plan is not None:
        print("window-plan:")
        for w in windows:
            for st, preds in window_plan[w].items():
                print(f"  w={w}  {st}: {','.join(preds)}")
    print(
        f"predict={predict_modes} windows={windows} horizon={args.horizon} "
        f"top_k={args.top_k} skip_weekends=True bias_margin={BIAS_MARGIN} "
        f"margin_conf={args.margin_conf} "
        f"types={search_types_all} exclude={sorted(excluded) or '[]'} "
        f"workers_cap={workers}"
    )

    rows: list[dict] = []
    t_all = time.perf_counter()
    n_jobs = len(start_indices) * len(windows)
    job = 0

    for date_i, (q_idx, start_date) in enumerate(
        zip(start_indices, start_dates), start=1
    ):
        for window in windows:
            job += 1
            q_start = q_idx
            actual_ts = pd.Timestamp(ohlc["datetime"].iloc[q_start])

            if window_plan is not None:
                type_predicts = window_plan[window]
                search_types = sorted(type_predicts)
            else:
                search_types = search_types_all
                type_predicts = {st: predict_modes for st in search_types}

            workers_n = min(workers, len(search_types))

            actuals = query_actuals(ohlc, q_start, window, args.horizon)
            actual_amp = actuals["amplitude"]
            actual_ft = actuals["first_touch"]
            act_high = actuals["high"]
            act_low = actuals["low"]
            act_signed = actuals["signed"]
            print(
                f"\n[{job}/{n_jobs}] date={actual_ts} window={window} "
                f"types={search_types} "
                f"actual={actual_amp} ft={actual_ft} "
                f"(high={act_high:+.3f} low={act_low:+.3f})",
                flush=True,
            )

            t0 = time.perf_counter()
            try:
                typed = find_similar_many(
                    ohlc,
                    search_types,
                    workers=workers_n,
                    window=window,
                    stride=1,
                    top_k=args.top_k,
                    query_start=q_start,
                    horizon=args.horizon,
                    price=args.price,
                    skip_weekends=True,
                )
            except ValueError as exc:
                print(f"  SKIP run: {exc}", flush=True)
                for st in search_types:
                    for predict in type_predicts[st]:
                        actual = (
                            actual_ft
                            if predict in FIRST_TOUCH_PREDICTS
                            else actual_amp
                        )
                        rows.append(
                            {
                                "date_i": date_i,
                                "start_date": start_date,
                                "query_start": q_start,
                                "window": window,
                                "search_type": st,
                                "predict": predict,
                                "actual_bias": actual,
                                "actual_max_dev_high": act_high,
                                "actual_max_dev_low": act_low,
                                "actual_signed_s": act_signed,
                                "actual_first_touch": actual_ft,
                                "pred": None,
                                "hit": None,
                                "soft": None,
                                "signed_agree": None,
                                "n_long": None,
                                "n_skip": None,
                                "n_short": None,
                                "error": str(exc),
                            }
                        )
                continue

            elapsed = time.perf_counter() - t0
            print(f"  search done in {elapsed:.1f}s", flush=True)

            for st, results in typed:
                highs = results["max_dev_high"].to_numpy(float)
                lows = results["max_dev_low"].to_numpy(float)
                distances = results["distance"].to_numpy(float)
                high_times = results["max_dev_high_time"].to_numpy()
                low_times = results["max_dev_low_time"].to_numpy()
                match_times = results["match_end_time"].to_numpy()
                labels = [
                    bias_from_devs(h, lo) for h, lo in zip(highs, lows)
                ]
                ft_labels = [
                    first_touch_label(ht, lt)
                    for ht, lt in zip(high_times, low_times)
                ]
                counts = bias_counts(labels)
                soft_amp = (
                    sum(1 for lab in labels if lab == actual_amp) / len(labels)
                    if labels
                    else 0.0
                )
                soft_ft = (
                    sum(1 for lab in ft_labels if lab == actual_ft)
                    / len(ft_labels)
                    if ft_labels
                    else 0.0
                )
                for predict in type_predicts[st]:
                    actual = (
                        actual_ft
                        if predict in FIRST_TOUCH_PREDICTS
                        else actual_amp
                    )
                    soft = (
                        soft_ft
                        if predict in FIRST_TOUCH_PREDICTS
                        else soft_amp
                    )
                    pred, extras = predict_topk(
                        method=predict,
                        highs=highs,
                        lows=lows,
                        distances=distances,
                        labels=labels,
                        margin_conf=args.margin_conf,
                        high_times=high_times,
                        low_times=low_times,
                        match_times=match_times,
                        query_time=actual_ts,
                        ft_labels=ft_labels,
                    )
                    hit = int(pred == actual)
                    s_pred = extras.get("signed_s")
                    if s_pred is None and len(highs):
                        s_pred = float(
                            np.mean(np.abs(highs) - np.abs(lows))
                        )
                    signed_agree = (
                        -abs(float(s_pred) - act_signed)
                        if s_pred is not None and np.isfinite(s_pred)
                        else None
                    )
                    rows.append(
                        {
                            "date_i": date_i,
                            "start_date": start_date,
                            "query_start": q_start,
                            "window": window,
                            "search_type": st,
                            "predict": predict,
                            "actual_bias": actual,
                            "actual_max_dev_high": act_high,
                            "actual_max_dev_low": act_low,
                            "actual_signed_s": act_signed,
                            "actual_first_touch": actual_ft,
                            "pred": pred,
                            "hit": hit,
                            "soft": soft,
                            "signed_s": s_pred,
                            "signed_agree": signed_agree,
                            "conf": extras.get("conf"),
                            "n_long": counts["long"],
                            "n_skip": counts["skip"],
                            "n_short": counts["short"],
                            "error": None,
                        }
                    )
                print(
                    f"  {st:14s} soft_amp={soft_amp:.2f} soft_ft={soft_ft:.2f} "
                    f"L/S/K={counts['long']}/{counts['short']}/{counts['skip']}",
                    flush=True,
                )

    df = pd.DataFrame(rows)
    elapsed_s = round(time.perf_counter() - t_all, 2)
    meta_base = {
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "n_dates": args.n_dates,
        "seed": args.seed,
        "start_dates": start_dates,
        "start_indices": [int(i) for i in start_indices],
        "windows": windows,
        "horizon": args.horizon,
        "top_k": args.top_k,
        "skip_weekends": True,
        "price": args.price,
        "predict_modes": predict_modes,
        "search_types": search_types_all,
        "excluded": sorted(excluded),
        "window_plan": (
            {
                str(w): {
                    st: preds for st, preds in by_type.items()
                }
                for w, by_type in window_plan.items()
            }
            if window_plan is not None
            else None
        ),
        "window_plan_path": args.window_plan,
        "workers": workers,
        "elapsed_s": elapsed_s,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.tag:
        tag = args.tag
    elif window_plan is not None:
        tag = "plan"
    elif predict_modes == list(PREDICT_MODES):
        tag = "all"
    else:
        tag = "+".join(predict_modes)
    stem = (
        f"{args.symbol}_{args.timeframe}_bias_backtest_"
        f"seed{args.seed}_n{args.n_dates}_{tag}"
    )
    detail_path = OUTPUT_DIR / f"{stem}_detail.csv"
    df.to_csv(detail_path, index=False)

    plan_pairs = print_plan_pair_summary(df) if window_plan is not None else []

    summary_by_predict: dict[str, dict] = {}
    for predict in predict_modes:
        report = build_report_for_predict(
            df,
            predict=predict,
            meta_base=meta_base,
            margin_conf=args.margin_conf,
        )
        report_path = (
            OUTPUT_DIR
            / f"{args.symbol}_{args.timeframe}_bias_backtest_"
            f"seed{args.seed}_n{args.n_dates}_{tag}_{predict}_report.json"
        )
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        if window_plan is None:
            print_predict_summary(report)
        print(f"Wrote {report_path}")
        best = report["overall"]["best_search_type"]
        top = report["overall"]["ranking"][0] if report["overall"]["ranking"] else None
        summary_by_predict[predict] = {
            "best_search_type": best,
            "hit_rate": top["hit_rate"] if top else None,
            "soft_mean": top["soft_mean"] if top else None,
            "signed_agree_mean": top["signed_agree_mean"] if top else None,
            "report": str(report_path.name),
        }

    summary = {
        "meta": meta_base,
        "by_predict": summary_by_predict,
        "plan_pairs": plan_pairs,
    }
    summary_path = OUTPUT_DIR / f"{stem}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if window_plan is None:
        print("\n=== Summary across --predict modes ===")
        for predict, info in summary_by_predict.items():
            print(
                f"  {predict:24s}  best={info['best_search_type']}  "
                f"hit={info['hit_rate']}  soft={info['soft_mean']}"
            )
    print(f"\nWrote {detail_path}")
    print(f"Wrote {summary_path}")
    print(f"Total elapsed: {elapsed_s}s")


if __name__ == "__main__":
    main()
