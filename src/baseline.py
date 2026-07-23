"""Lomb-Scargle periodogram baseline.

The Lomb-Scargle periodogram is the standard method for finding periodic signals
in unevenly-sampled data. It is the non-machine-learning reference this project
must beat: given a (down-sampled) light curve, take its strongest periodogram
peak as the estimated period and ask how often that matches the true LCDB period.

Two asteroid-specific conventions matter for scoring:

* **Half-period alias.** Asteroid light curves are typically double-peaked (two
  brightness maxima per rotation), so the dominant Fourier component sits at twice
  the rotation frequency and Lomb-Scargle most often returns *half* the true
  rotation period. A recovered period is therefore counted correct if it matches
  the true period or its 2x / 0.5x harmonic, within a fractional tolerance.

* **Bounded search.** We search photometric periods in a fixed range (default
  1-48 h). Evaluation is restricted to objects whose true period sits well inside
  that range so the search can reach the period and both aliases.

The per-curve outputs here (best period, peak power, false-alarm probability) also
become input features for the supervised classifier in the next step.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from astropy.timeseries import LombScargle

from sparsity import (
    DEFAULT_SPARSITY_LEVELS,
    densest_apparition,
    downsample,
    prepare_dense_curve,
)

MIN_PERIOD_H = 1.0
MAX_PERIOD_H = 48.0
# Evaluation set: true periods comfortably inside the search range (so the true
# period, its half, and its double are all reachable).
EVAL_MIN_PERIOD_H = 2.0
EVAL_MAX_PERIOD_H = 24.0


def lombscargle_best(
    jd,
    mag,
    mag_err=None,
    *,
    min_period_h: float = MIN_PERIOD_H,
    max_period_h: float = MAX_PERIOD_H,
    samples_per_peak: int = 10,
    use_errors: bool = False,
) -> dict:
    """Run Lomb-Scargle and return the strongest peak and its diagnostics.

    Times are Julian dates (days); periods are reported in hours. Measurement
    errors weight the fit only when ``use_errors`` is True and all are finite and
    positive; by default the fit is unweighted, because ALCDEF's per-observer
    error estimates are heterogeneous and empirically pull the top peak onto
    aliases (see the notebook's weighted-vs-unweighted check).
    """
    t = np.asarray(jd, dtype=float)
    y = np.asarray(mag, dtype=float)

    dy = None
    if use_errors and mag_err is not None:
        e = np.asarray(mag_err, dtype=float)
        if e.size == y.size and np.all(np.isfinite(e)) and np.all(e > 0):
            dy = e

    # frequency in cycles/day; period_days = period_h / 24  ->  f = 24 / period_h
    f_min = 24.0 / max_period_h
    f_max = 24.0 / min_period_h

    ls = LombScargle(t, y, dy=dy)
    freq, power = ls.autopower(
        minimum_frequency=f_min,
        maximum_frequency=f_max,
        samples_per_peak=samples_per_peak,
    )
    i = int(np.argmax(power))
    best_period_h = 24.0 / freq[i]
    best_power = float(power[i])

    try:
        fap = float(ls.false_alarm_probability(power[i], method="baluev"))
    except Exception:
        fap = np.nan

    return {
        "best_period_h": float(best_period_h),
        "best_power": best_power,
        "fap": fap,
        "n_freq": int(freq.size),
    }


def period_matches(
    p_est_h: float,
    p_true_h: float,
    *,
    tol: float = 0.05,
    factors=(1.0, 2.0, 0.5),
) -> bool:
    """True if the estimate matches the true period or its 2x / 0.5x alias.

    `factors` multiply the *estimate*: factor 2 accepts the case where Lomb-Scargle
    found the half-period (2 x estimate == true), factor 0.5 the double-period case.
    """
    if not (np.isfinite(p_est_h) and np.isfinite(p_true_h) and p_true_h > 0):
        return False
    return any(abs(p_est_h * f - p_true_h) <= tol * p_true_h for f in factors)


def run_baseline(
    numbers,
    period_lookup: dict,
    levels=DEFAULT_SPARSITY_LEVELS,
    *,
    tol: float = 0.05,
    strategy: str = "stratified",
    samples_per_peak: int = 10,
    seed: int = 0,
    zip_path=None,
) -> pd.DataFrame:
    """Down-sample each object at each level, run Lomb-Scargle, and score recovery.

    Returns one row per (object, sparsity level) with the estimated period, peak
    diagnostics, and whether it matched the true period (alias-aware).
    """
    kwargs = {} if zip_path is None else {"zip_path": zip_path}
    rows = []
    for k, num in enumerate(numbers):
        # Restrict to the densest single apparition so the time baseline is
        # bounded and controlled across the sparsity sweep.
        curve = densest_apparition(prepare_dense_curve(int(num), **kwargs))
        if len(curve) < 5:
            continue
        p_true = float(period_lookup[int(num)])
        # one reproducible RNG per object; reused across levels
        rng = np.random.default_rng(seed + k)
        for level in levels:
            sub = downsample(curve, int(level), strategy=strategy, rng=rng)
            res = lombscargle_best(
                sub["jd"].values,
                sub["mag"].values,
                sub["mag_err"].values,
                samples_per_peak=samples_per_peak,
            )
            rows.append(
                {
                    "number": int(num),
                    "period_true_h": p_true,
                    "level": int(level),
                    "n_obs": int(len(sub)),
                    "period_ls_h": res["best_period_h"],
                    "power": res["best_power"],
                    "fap": res["fap"],
                    "matched": period_matches(res["best_period_h"], p_true, tol=tol),
                }
            )
    return pd.DataFrame(rows)
