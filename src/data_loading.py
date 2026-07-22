"""Data loading utilities for the asteroid rotation-period project.

This module parses the raw public data sources into tidy pandas objects:

* LCDB (Asteroid Lightcurve Database) summary table -> the *labels* (published
  rotation periods and U quality codes), one row per asteroid.
* ALCDEF light curves -> the dense photometric time series (added later).

The LCDB summary file (`lc_summary_pub.txt`) is a fixed-width text file, not a
delimited one: asteroid names contain spaces and empty fields are just blanks,
so it must be parsed with explicit column boundaries. The boundaries below come
from section 4.1.3 ("LC_SUMMARY AND LC_DETAILS COLUMN MAP") of the LCDB
readme.pdf shipped with the 2023-Oct release. Positions in that document are
1-indexed and inclusive; pandas `read_fwf` wants 0-indexed half-open intervals,
which is what `_LCDB_COLSPECS` encodes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Repo root = parent of this file's parent (src/ -> project root).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LCDB_SUMMARY_PATH = PROJECT_ROOT / "data" / "raw" / "lcdb" / "lc_summary_pub.txt"

# (name, (start, end)) with 0-indexed, half-open positions derived from the
# readme's 1-indexed inclusive "Pos" column. Only the fields this project uses
# are pulled out by name; the rest of the fixed-width line is ignored.
_LCDB_COLSPECS: list[tuple[str, tuple[int, int]]] = [
    ("number", (0, 7)),        # I7   1-7    MPC number (blank if unnumbered)
    ("entry_flag", (8, 9)),    # A1   9      '*' = vetted new/revised record
    ("name", (10, 40)),        # A30  11-40  name, or designation if unnamed
    ("desig", (41, 61)),       # A20  42-61  MPC primary designation
    ("family", (62, 70)),      # A8   63-70  orbital group / collisional family
    ("class", (73, 83)),       # A10  74-83  taxonomic class
    ("diam_km", (88, 96)),     # F8.3 89-96  adopted diameter (km)
    ("H", (99, 105)),          # F6.3 100-105 adopted absolute magnitude
    ("period_h", (145, 158)),  # F13.8 146-158 rotation period (hours)
    ("amp_flag", (175, 176)),  # A1   176    amplitude qualifier (< or >)
    ("amp_min", (177, 181)),   # F4.2 178-181 min reported amplitude (mag)
    ("amp_max", (182, 186)),   # F4.2 183-186 max reported amplitude (mag)
    ("U", (187, 189)),         # A2   188-189 lightcurve quality code
    ("binary", (196, 199)),    # A3   197-199 ?/B/M flag
    ("survey", (204, 209)),    # A5   205-209 survey source, if any
]

_NUMERIC_COLS = ["number", "diam_km", "H", "period_h", "amp_min", "amp_max"]

# The LCDB U (quality) code is an ordered half-step scale, not a plain integer.
# From best to worst: 3, 3-, 2+, 2, 2-, 1+, 1, 1-, 0. We encode it as a float so
# it can be filtered and plotted while preserving that order, mapping the +/-
# modifiers to +/-0.3 around the base integer. This keeps every half-step
# distinct and correctly ordered (e.g. 2- = 1.7 < 2 = 2.0 < 2+ = 2.3 < 3- = 2.7).
_U_MODIFIER = {"+": 0.3, "-": -0.3, "": 0.0}


def parse_u_code(raw) -> float:
    """Convert a raw LCDB U string (e.g. '2', '2-', '3', '1+') to an ordered float.

    Returns NaN for blank/missing codes (the summary line legitimately omits a U
    code when no detail record was deemed reliable enough).
    """
    if raw is None:
        return np.nan
    s = str(raw).strip()
    if not s or s.lower() == "nan":
        return np.nan
    base_char = s[0]
    if not base_char.isdigit():
        return np.nan
    modifier = s[1] if len(s) > 1 and s[1] in _U_MODIFIER else ""
    return float(base_char) + _U_MODIFIER[modifier]


def load_lcdb_summary(path: Path | str = LCDB_SUMMARY_PATH) -> pd.DataFrame:
    """Load the LCDB summary table as a tidy DataFrame (one row per asteroid).

    All whitespace-padded strings are stripped, numeric columns (stored as
    strings in the source to preserve precision) are coerced to floats, and the
    U quality code is parsed into an ordered numeric column `U_num` alongside the
    raw `U` string.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"LCDB summary not found at {path}. Download + extract "
            "LCLIST_PUB_2023OCT.zip into data/raw/lcdb/ first."
        )

    names = [n for n, _ in _LCDB_COLSPECS]
    colspecs = [span for _, span in _LCDB_COLSPECS]

    # The file has 5 preamble/header lines before the first data row:
    #   1: "ASTEROID LIGHTCURVE DATABASE (LCDB) SUMMARY (Complete)"
    #   2: "GENERATED: ..."
    #   3: blank
    #   4: column header
    #   5: dashed rule
    df = pd.read_fwf(
        path,
        colspecs=colspecs,
        names=names,
        skiprows=5,
        dtype=str,
        encoding="latin-1",  # LCDB uses ASCII-ish names but a few stray bytes appear
    )

    # Strip padding from every string field.
    for col in df.columns:
        df[col] = df[col].str.strip()

    # Coerce numeric columns (stored as strings in the source).
    for col in _NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Parse the U code into an ordered numeric scale.
    df["U_num"] = df["U"].map(parse_u_code)

    return df


def reliable_labels(
    df: pd.DataFrame | None = None,
    *,
    min_u: float = 2.0,
    require_period: bool = True,
) -> pd.DataFrame:
    """Filter the LCDB summary down to trustworthy period labels.

    Parameters
    ----------
    df:
        A DataFrame from `load_lcdb_summary`. If None, it is loaded from disk.
    min_u:
        Minimum U quality on the ordered numeric scale (see `parse_u_code`).
        Default 2.0 keeps U >= 2 but *excludes* 2- (=1.7). Pass 1.7 to also
        include 2-, which the LCDB readme calls "the minimum reliability code
        that we accept for statistical analysis."
    require_period:
        Drop rows with no numeric period (default True).
    """
    if df is None:
        df = load_lcdb_summary()

    mask = df["U_num"] >= min_u
    if require_period:
        mask &= df["period_h"].notna()
    return df.loc[mask].reset_index(drop=True)
