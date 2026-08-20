"""Real sparse survey photometry: load + reduce Gaia DR3 asteroid light curves.

Everything upstream in this project simulates sparsity by down-sampling dense
ALCDEF curves; this module prepares the genuine article — Gaia DR3's
`sso_observation` epoch photometry, fetched raw by scripts/fetch_survey_data.py
— into the same shape the pipeline expects (`jd`, `mag`, `mag_err`, `night`),
so Lomb-Scargle, the feature extractor, and the trained classifiers run on real
survey data unchanged.

Two reduction steps are new relative to ALCDEF (whose per-session zero-point
removal hid them):

1. **Per-transit aggregation.** A Gaia field transit produces one photometric
   measurement repeated across ~2-9 per-CCD astrometry rows a few seconds
   apart; those rows are one observation, not several, so we collapse each
   transit to a single point (mean epoch, first non-null magnitude).

2. **Geometry removal via observed - predicted.** Over months, the apparent
   magnitude is dominated by changing distances and phase angle (often > 1 mag
   — several times the rotation amplitude). Miriade's ephemeris `VMag` predicts
   exactly those effects (distances + the standard H,G phase law), so the
   residual `g_mag - VMag` leaves rotation, a constant color offset (Gaia G vs
   Johnson V — removed with the median), and slow phase-law errors, which a
   residual linear phase-angle detrend absorbs.

The result is analogous to `sparsity.prepare_dense_curve` output: a calibrated,
outlier-clipped, zero-centered curve. Use `sparsity.densest_apparition` on it
to match the single-apparition time spans the models were trained on.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SURVEY_DIR = Path("data/raw/surveys")

# Gaia epoch fields are days since JD 2455197.5 (2010-01-01, the DPAC
# convention); the guard tolerates an archive serving absolute JD instead.
GAIA_EPOCH_OFFSET = 2455197.5


def _epoch_to_jd(epoch: float) -> float:
    return epoch if epoch > 2.4e6 else epoch + GAIA_EPOCH_OFFSET


def load_gaia_rows(survey_dir=SURVEY_DIR) -> pd.DataFrame:
    """All downloaded Gaia sso_observation rows, one DataFrame.

    Columns: number, transit_id, jd (UTC), g_mag, g_flux, g_flux_error.
    Rows without photometry (g_mag null) are dropped.
    """
    frames = []
    for path in sorted((survey_dir / "gaia").glob("chunk_*.json")):
        payload = json.loads(path.read_text())
        cols = [c["name"] for c in payload["metadata"]]
        df = pd.DataFrame(payload["data"], columns=cols)
        df = df[df["g_mag"].notna()]
        frames.append(pd.DataFrame({
            "number": df["number_mp"].astype(int),
            "transit_id": df["transit_id"].astype(np.int64),
            "jd": df["epoch_utc"].astype(float).map(_epoch_to_jd),
            "g_mag": df["g_mag"].astype(float),
            "g_flux": df["g_flux"].astype(float),
            "g_flux_error": df["g_flux_error"].astype(float),
        }))
    if not frames:
        return pd.DataFrame(columns=["number", "transit_id", "jd", "g_mag",
                                     "g_flux", "g_flux_error"])
    return pd.concat(frames, ignore_index=True)


def load_miriade(number: int, survey_dir=SURVEY_DIR) -> pd.DataFrame | None:
    """Miriade ephemerides for one asteroid: jd, vmag_pred, phase_deg, dobs_au.

    Returns None if the geometry file is missing or unparsable (e.g. the fetch
    is still in progress).
    """
    path = survey_dir / "miriade" / f"{number}.json"
    if not path.exists():
        return None
    try:
        rows = json.loads(path.read_text())["data"]
        eph = pd.DataFrame({
            "jd": [float(r["Date"]) for r in rows],
            "vmag_pred": [float(r["VMag"]) for r in rows],
            "phase_deg": [float(r["Phase"]) for r in rows],
            "dobs_au": [float(r["Dobs"]) for r in rows],
        })
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return eph.sort_values("jd").reset_index(drop=True)


def gaia_numbers(survey_dir=SURVEY_DIR) -> list[int]:
    """Asteroid numbers with both Gaia photometry and Miriade geometry."""
    have_geom = {int(p.stem) for p in (survey_dir / "miriade").glob("*.json")}
    have_phot = set(load_gaia_rows(survey_dir)["number"].unique())
    return sorted(have_phot & have_geom)


def prepare_gaia_curve(
    number: int,
    gaia_rows: pd.DataFrame,
    *,
    survey_dir=SURVEY_DIR,
    match_tol_s: float = 5.0,
    sigma_clip: float = 5.0,
) -> pd.DataFrame:
    """One reduced, pipeline-shaped Gaia light curve for an asteroid.

    Steps: match each Gaia row to its Miriade prediction by epoch, form the
    observed-minus-predicted residual, collapse per-CCD rows to one point per
    transit, detrend the residual linearly in phase angle, clip outliers, and
    remove the median. Returns a DataFrame with columns
    ``jd, mag, mag_err, phase_deg, night`` sorted by time (empty if the
    asteroid lacks data or geometry).
    """
    empty = pd.DataFrame(columns=["jd", "mag", "mag_err", "phase_deg", "night"])
    obs = gaia_rows[gaia_rows["number"] == int(number)]
    eph = load_miriade(number, survey_dir)
    if len(obs) == 0 or eph is None or len(eph) == 0:
        return empty

    # match observation epochs to ephemeris epochs (identical up to rounding)
    obs = obs.sort_values("jd")
    merged = pd.merge_asof(obs, eph, on="jd", direction="nearest",
                           tolerance=match_tol_s / 86400.0).dropna(
                               subset=["vmag_pred"])
    if len(merged) == 0:
        return empty

    merged["resid"] = merged["g_mag"] - merged["vmag_pred"]
    # 1.0857 = 2.5 / ln(10): flux error -> magnitude error
    merged["mag_err"] = 1.0857 * merged["g_flux_error"] / merged["g_flux"]

    # one photometric point per field transit (per-CCD rows repeat it)
    per_transit = merged.groupby("transit_id").agg(
        jd=("jd", "mean"),
        resid=("resid", "median"),
        mag_err=("mag_err", "median"),
        phase_deg=("phase_deg", "mean"),
    ).sort_values("jd").reset_index(drop=True)

    # absorb slow H,G phase-law errors with a linear phase-angle detrend
    if len(per_transit) >= 5 and per_transit["phase_deg"].std() > 0:
        slope, intercept = np.polyfit(per_transit["phase_deg"],
                                      per_transit["resid"], 1)
        per_transit["resid"] -= slope * per_transit["phase_deg"] + intercept

    # outlier rejection + zero-centering, mirroring prepare_dense_curve
    if sigma_clip:
        med = per_transit["resid"].median()
        mad = 1.4826 * (per_transit["resid"] - med).abs().median()
        if mad > 0:
            per_transit = per_transit[
                (per_transit["resid"] - med).abs() < sigma_clip * mad]

    curve = pd.DataFrame({
        "jd": per_transit["jd"].values,
        "mag": (per_transit["resid"] - per_transit["resid"].median()).values,
        "mag_err": per_transit["mag_err"].values,
        "phase_deg": per_transit["phase_deg"].values,
        "night": np.floor(per_transit["jd"].values).astype("int64"),
    })
    return curve.sort_values("jd").reset_index(drop=True)
