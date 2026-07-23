"""Controlled-sparsity pipeline.

Turn a dense ALCDEF light curve into progressively sparser versions that mimic
survey sampling, while keeping the known LCDB rotation period as the label. This
is the experimental core of the project: it lets us measure how period-recovery
degrades as observations are removed.

Two stages:

1. ``prepare_dense_curve`` assembles one clean, calibrated *source* curve for an
   asteroid. It concatenates every observing session, subtracts each session's
   median magnitude (removing the per-session zero-point, so the combined curve
   sits on one consistent photometric system the way a calibrated survey's data
   would), and rejects outliers with a median-absolute-deviation cut.

2. ``downsample`` / ``sparsity_sweep`` draw a subset of the points to simulate a
   given number of survey observations. Because the true period is known, every
   sparse realization is a labeled example.

Note on realism: subtracting the session median uses the full dense session,
which a real sparse survey could not do. That is intentional — it emulates the
calibrated photometry a survey *delivers*, so the down-sampled curve looks like
survey data (consistent zero-point, few points) rather than raw amateur data.
The genuine survey-cadence test comes later from real ZTF / Gaia data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_loading import ALCDEF_ZIP_PATH, load_alcdef_blocks

# Default observation counts for a sparsity sweep, dense -> very sparse.
DEFAULT_SPARSITY_LEVELS: tuple[int, ...] = (200, 100, 50, 30, 20, 10, 5)

_CURVE_COLUMNS = ["jd", "mag", "mag_err", "block_id", "night"]


def prepare_dense_curve(
    number: int,
    *,
    sigma_clip: float = 5.0,
    zip_path=ALCDEF_ZIP_PATH,
) -> pd.DataFrame:
    """Assemble one calibrated, outlier-cleaned source curve for an asteroid.

    Returns a DataFrame sorted by time with columns jd, mag (zero-point removed
    per session), mag_err, block_id, and night (integer JD ~ one night).
    """
    frames = []
    for block in load_alcdef_blocks(number, zip_path):
        if len(block) == 0:
            continue
        frames.append(
            pd.DataFrame(
                {
                    "jd": block.jd,
                    # remove this session's zero-point (calibration proxy)
                    "mag": block.mag - np.median(block.mag),
                    "mag_err": block.mag_err,
                    "block_id": block.block_id,
                    "night": np.floor(block.jd).astype("int64"),
                }
            )
        )
    if not frames:
        return pd.DataFrame(columns=_CURVE_COLUMNS)

    curve = pd.concat(frames, ignore_index=True)

    if sigma_clip:
        med = curve["mag"].median()
        mad = 1.4826 * (curve["mag"] - med).abs().median()
        if mad > 0:
            keep = (curve["mag"] - med).abs() < sigma_clip * mad
            curve = curve[keep]

    return curve.sort_values("jd").reset_index(drop=True)


def densest_apparition(curve: pd.DataFrame, *, gap_days: float = 90.0) -> pd.DataFrame:
    """Return the single apparition (contiguous observing window) with the most points.

    Splitting the curve where consecutive observations are more than ``gap_days``
    apart isolates one apparition. Using one apparition bounds the time baseline,
    which (a) keeps the period search well-conditioned and fast and (b) controls
    the baseline as a confound so the sparsity sweep isolates the effect of the
    *number* of observations rather than how long they are spread over.
    """
    if len(curve) == 0:
        return curve
    curve = curve.sort_values("jd").reset_index(drop=True)
    group = (curve["jd"].diff() > gap_days).cumsum()
    best = curve.groupby(group).size().idxmax()
    return curve[group == best].reset_index(drop=True)


def downsample(
    curve: pd.DataFrame,
    n_points: int,
    *,
    strategy: str = "stratified",
    rng=None,
) -> pd.DataFrame:
    """Draw an ``n_points`` subset of a prepared curve.

    strategy:
      * ``"random"``     — uniform random selection of n_points observations.
      * ``"stratified"`` — spread the selection across distinct nights
        (round-robin over nights, random point within each), so the retained
        points cover the observing baseline the way survey epochs do.

    If ``n_points`` >= len(curve) the whole curve is returned. ``rng`` may be an
    int seed, a numpy Generator, or None.
    """
    rng = np.random.default_rng(rng)
    n = len(curve)
    if n_points >= n:
        return curve.reset_index(drop=True)

    if strategy == "random":
        idx = rng.choice(n, size=n_points, replace=False)
        return curve.iloc[np.sort(idx)].reset_index(drop=True)

    if strategy == "stratified":
        curve = curve.reset_index(drop=True)
        # positional indices per night, shuffled
        per_night = {
            night: rng.permutation(grp.index.values).tolist()
            for night, grp in curve.groupby("night")
        }
        night_order = rng.permutation(list(per_night.keys())).tolist()
        chosen: list[int] = []
        # round-robin across nights until we hit the target or run dry
        while len(chosen) < n_points and any(per_night[k] for k in night_order):
            for night in night_order:
                if per_night[night]:
                    chosen.append(per_night[night].pop())
                    if len(chosen) >= n_points:
                        break
        return curve.iloc[np.sort(chosen)].reset_index(drop=True)

    raise ValueError(f"unknown strategy: {strategy!r}")


def sparsity_sweep(
    curve: pd.DataFrame,
    levels=DEFAULT_SPARSITY_LEVELS,
    *,
    strategy: str = "stratified",
    rng=None,
) -> dict[int, pd.DataFrame]:
    """Return {n_points: down-sampled curve} for each level in ``levels``."""
    rng = np.random.default_rng(rng)
    return {
        int(n): downsample(curve, int(n), strategy=strategy, rng=rng)
        for n in levels
    }


def select_dense_pool(
    coverage: pd.DataFrame,
    *,
    min_points: int = 100,
    min_session_points: int = 30,
) -> list[int]:
    """Asteroid numbers whose coverage is dense enough to serve as source curves."""
    mask = (coverage["n_points"] >= min_points) & (
        coverage["max_session_points"] >= min_session_points
    )
    return coverage.loc[mask, "number"].astype(int).tolist()


def phase_fold(jd: np.ndarray, period_hours: float) -> np.ndarray:
    """Fold Julian dates to rotational phase in [0, 1) at the given period (hours)."""
    return ((np.asarray(jd, dtype=float) * 24.0) % period_hours) / period_hours
