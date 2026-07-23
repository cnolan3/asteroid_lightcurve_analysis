"""Feature extraction for the supervised period-reliability classifier.

The classifier's job is to predict whether the Lomb-Scargle period of a *sparse*
light curve is actually correct — i.e. to tell a trustworthy period determination
from an alias. That is how the model adds value over the periodogram alone:
Lomb-Scargle returns a period for every object, but cannot say which ones to
believe; the classifier can, yielding a high-purity subset of period estimates.

For each sparse curve we build a feature vector from two sources, using only
information available at prediction time (never the true period):

* **Sampling geometry** — how many points, how many nights, the time baseline,
  the typical gap. Sparser, gappier curves are less reliable.
* **Photometry** — robust amplitude and scatter of the magnitudes.
* **Periodogram shape** — the strength of the top peak, how much it dominates the
  second peak, its false-alarm probability, how many competing peaks there are,
  and how close the top frequency sits to a daily (1 cycle/day) alias.

The label is whether the top periodogram period matches the true LCDB period
(alias-aware), computed with `baseline.period_matches`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from astropy.timeseries import LombScargle
from scipy.signal import find_peaks

from baseline import MAX_PERIOD_H, MIN_PERIOD_H, period_matches
from sparsity import (
    DEFAULT_SPARSITY_LEVELS,
    densest_apparition,
    downsample,
    prepare_dense_curve,
)

# Feature columns the classifier consumes (order fixed for reproducibility).
FEATURE_COLUMNS = [
    "n_obs",
    "n_nights",
    "baseline_days",
    "median_dt_days",
    "amp_5_95",
    "mag_std",
    "mag_mad",
    "mag_skew",
    "ls_top_power",
    "ls_top_period_h",
    "ls_peak_ratio",
    "ls_fap",
    "ls_n_peaks",
    "ls_daily_alias_dist",
]


def _separated_peaks(power: np.ndarray, min_distance: int) -> np.ndarray:
    """Indices of well-separated periodogram peaks, strongest first.

    A minimum spacing (``min_distance`` grid points, ~ one peak width) is enforced
    so the "second peak" is a genuinely distinct alias rather than a shoulder of
    the dominant peak — otherwise, on a finely-sampled grid, adjacent grid points
    of the same peak would masquerade as separate peaks.
    """
    if power.size < 3:
        return np.array([int(np.argmax(power))]) if power.size else np.array([], int)
    idx, _ = find_peaks(power, distance=max(1, int(min_distance)))
    if idx.size == 0:
        return np.array([int(np.argmax(power))])
    return idx[np.argsort(power[idx])[::-1]]


def extract_features(sub: pd.DataFrame, *, samples_per_peak: int = 10) -> dict:
    """Compute the feature vector for one (sparse) light curve.

    Runs Lomb-Scargle once and derives both the top period and the periodogram
    shape features. Returns a dict including every name in FEATURE_COLUMNS plus
    the estimated period `ls_top_period_h`.
    """
    t = np.asarray(sub["jd"].values, dtype=float)
    y = np.asarray(sub["mag"].values, dtype=float)
    order = np.argsort(t)
    t, y = t[order], y[order]

    # --- sampling geometry ---
    n_obs = t.size
    n_nights = int(np.unique(np.floor(t)).size)
    baseline_days = float(t[-1] - t[0]) if n_obs > 1 else 0.0
    dts = np.diff(t)
    median_dt = float(np.median(dts)) if dts.size else 0.0

    # --- photometry ---
    p5, p95 = np.percentile(y, [5, 95])
    amp_5_95 = float(p95 - p5)
    mag_std = float(np.std(y))
    med = np.median(y)
    mag_mad = float(1.4826 * np.median(np.abs(y - med)))
    if mag_std > 0 and n_obs > 2:
        z = (y - y.mean()) / mag_std
        mag_skew = float(np.mean(z ** 3))
    else:
        mag_skew = 0.0

    # --- periodogram ---
    ls = LombScargle(t, y)  # unweighted (see baseline.py)
    freq, power = ls.autopower(
        minimum_frequency=24.0 / MAX_PERIOD_H,
        maximum_frequency=24.0 / MIN_PERIOD_H,
        samples_per_peak=samples_per_peak,
    )
    peaks = _separated_peaks(power, min_distance=samples_per_peak)
    i_top = int(peaks[0])
    top_power = float(power[i_top])
    top_freq = float(freq[i_top])
    top_period_h = 24.0 / top_freq

    # ratio of second-highest peak to the top (1.0 => no dominant peak)
    if peaks.size > 1:
        peak_ratio = float(power[int(peaks[1])] / top_power) if top_power > 0 else 1.0
    else:
        peak_ratio = 0.0
    # number of peaks at least half as strong as the top
    n_peaks = int(np.sum(power[peaks] >= 0.5 * top_power))
    # distance of the top frequency to the nearest integer cycles/day (alias risk)
    daily_alias_dist = float(abs(top_freq - round(top_freq)))
    try:
        fap = float(ls.false_alarm_probability(top_power, method="baluev"))
    except Exception:
        fap = np.nan

    return {
        "n_obs": n_obs,
        "n_nights": n_nights,
        "baseline_days": baseline_days,
        "median_dt_days": median_dt,
        "amp_5_95": amp_5_95,
        "mag_std": mag_std,
        "mag_mad": mag_mad,
        "mag_skew": mag_skew,
        "ls_top_power": top_power,
        "ls_top_period_h": float(top_period_h),
        "ls_peak_ratio": peak_ratio,
        "ls_fap": fap,
        "ls_n_peaks": n_peaks,
        "ls_daily_alias_dist": daily_alias_dist,
    }


def build_feature_table(
    numbers,
    period_lookup: dict,
    levels=DEFAULT_SPARSITY_LEVELS,
    *,
    tol: float = 0.05,
    samples_per_peak: int = 10,
    seed: int = 0,
    zip_path=None,
) -> pd.DataFrame:
    """Build the (features + label) training table over objects and sparsity levels.

    One row per (object, level): the feature vector, identifiers, and `matched`
    (whether the Lomb-Scargle period is correct, alias-aware).
    """
    kwargs = {} if zip_path is None else {"zip_path": zip_path}
    rows = []
    for k, num in enumerate(numbers):
        curve = densest_apparition(prepare_dense_curve(int(num), **kwargs))
        if len(curve) < 5:
            continue
        p_true = float(period_lookup[int(num)])
        rng = np.random.default_rng(seed + k)
        for level in levels:
            sub = downsample(curve, int(level), rng=rng)
            feats = extract_features(sub, samples_per_peak=samples_per_peak)
            feats["matched"] = period_matches(feats["ls_top_period_h"], p_true, tol=tol)
            feats["number"] = int(num)
            feats["level"] = int(level)
            feats["period_true_h"] = p_true
            rows.append(feats)
    return pd.DataFrame(rows)
