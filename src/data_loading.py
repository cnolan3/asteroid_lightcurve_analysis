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

import re
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

# Repo root = parent of this file's parent (src/ -> project root).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LCDB_SUMMARY_PATH = PROJECT_ROOT / "data" / "raw" / "lcdb" / "lc_summary_pub.txt"
ALCDEF_ZIP_PATH = PROJECT_ROOT / "data" / "raw" / "alcdef" / "ALCDEF_ALL.zip"

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


# ---------------------------------------------------------------------------
# ALCDEF light curves
# ---------------------------------------------------------------------------
# The ALCDEF archive (ALCDEF_ALL.zip) holds one .txt file per asteroid, named
# ALCDEF_<number>_<name>.txt (number 0 = unnumbered). Each file is a sequence of
# observing-session *blocks*:
#
#     STARTMETADATA
#     KEY=VALUE            (one per line; e.g. LCBLOCKID, SESSIONDATE, FILTER,
#     ...                   MAGBAND, DELIMITER, DIFFERMAGS)
#     ENDMETADATA
#     DATA=<jd><d><mag><d><magerr>   (repeated; <d> = the block's DELIMITER)
#     ...
#     STARTMETADATA        (the next block begins)
#
# There are no STARTDATA/ENDDATA markers: data rows run from ENDMETADATA to the
# next STARTMETADATA. A survey of the archive found DELIMITER is always PIPE and
# every DATA row has 3 columns, but the parser reads the DELIMITER field and
# tolerates a missing error column so it stays correct on any stray variants.
#
# Each block has its own magnitude zero-point (magnitudes are unreduced, and
# some blocks are differential). Blocks are therefore kept separate rather than
# merged into one series, so downstream code can handle per-session offsets.

_ALCDEF_DELIMITERS = {
    "PIPE": "|",
    "TAB": "\t",
    "COMMA": ",",
    "SEMICOLON": ";",
    "SPACE": " ",
}


@dataclass
class LightCurveBlock:
    """One ALCDEF observing session: metadata plus its (JD, mag, mag_err) arrays."""

    metadata: dict
    jd: np.ndarray
    mag: np.ndarray
    mag_err: np.ndarray

    @property
    def block_id(self) -> str | None:
        return self.metadata.get("LCBLOCKID")

    @property
    def session_date(self) -> str | None:
        return self.metadata.get("SESSIONDATE")

    @property
    def filter(self) -> str | None:
        return self.metadata.get("FILTER")

    @property
    def magband(self) -> str | None:
        return self.metadata.get("MAGBAND")

    @property
    def is_differential(self) -> bool:
        return str(self.metadata.get("DIFFERMAGS", "")).upper() == "TRUE"

    def __len__(self) -> int:
        return int(self.jd.size)


def parse_alcdef_text(text: str) -> list[LightCurveBlock]:
    """Parse the full text of an ALCDEF file into a list of observing-session blocks."""
    blocks: list[LightCurveBlock] = []
    meta: dict = {}
    jd: list[float] = []
    mag: list[float] = []
    err: list[float] = []
    delim = "|"
    state = "idle"  # idle -> meta -> data, resetting at each STARTMETADATA

    def flush() -> None:
        if meta and jd:
            blocks.append(
                LightCurveBlock(
                    metadata=dict(meta),
                    jd=np.asarray(jd, dtype=float),
                    mag=np.asarray(mag, dtype=float),
                    mag_err=np.asarray(err, dtype=float),
                )
            )

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line == "STARTMETADATA":
            flush()
            meta, jd, mag, err, delim, state = {}, [], [], [], "|", "meta"
            continue
        if line == "ENDMETADATA":
            delim = _ALCDEF_DELIMITERS.get(
                meta.get("DELIMITER", "PIPE").upper(), "|"
            )
            state = "data"
            continue
        if state == "meta":
            if "=" in line:
                key, value = line.split("=", 1)
                meta[key.strip()] = value.strip()
            continue
        if state == "data" and line.startswith("DATA="):
            parts = line[5:].split(delim)
            if len(parts) < 2:
                continue
            try:
                j = float(parts[0])
                m = float(parts[1])
            except ValueError:
                continue  # skip malformed row, keep the rest of the block
            e = np.nan
            if len(parts) >= 3 and parts[2].strip():
                try:
                    e = float(parts[2])
                except ValueError:
                    e = np.nan
            jd.append(j)
            mag.append(m)
            err.append(e)

    flush()  # emit the final block (no trailing STARTMETADATA to trigger it)
    return blocks


@lru_cache(maxsize=1)
def _alcdef_index(zip_path: str) -> dict:
    """Map asteroid number -> member filename, read once from the zip directory."""
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.endswith(".txt")]
    by_number: dict[int, str] = {}
    for name in names:
        m = re.match(r"(?:.*/)?ALCDEF_(\d+)_.*\.txt$", name)
        if m:
            num = int(m.group(1))
            if num > 0:
                by_number[num] = name
    return {"names": names, "by_number": by_number}


def has_alcdef(number: int, zip_path: Path | str = ALCDEF_ZIP_PATH) -> bool:
    """True if the archive contains a light-curve file for this asteroid number."""
    return int(number) in _alcdef_index(str(zip_path))["by_number"]


def load_alcdef_blocks(
    number: int, zip_path: Path | str = ALCDEF_ZIP_PATH
) -> list[LightCurveBlock]:
    """Read and parse all observing-session blocks for one asteroid (by number)."""
    index = _alcdef_index(str(zip_path))
    member = index["by_number"].get(int(number))
    if member is None:
        raise KeyError(f"No ALCDEF file for asteroid number {number}")
    with zipfile.ZipFile(zip_path) as zf:
        text = zf.read(member).decode("latin-1")
    return parse_alcdef_text(text)


_PHOTOMETRY_COLUMNS = [
    "jd",
    "mag",
    "mag_err",
    "block_id",
    "session_date",
    "filter",
    "magband",
    "is_differential",
]


def object_photometry(
    number: int, zip_path: Path | str = ALCDEF_ZIP_PATH
) -> pd.DataFrame:
    """Return one tidy long-form DataFrame of all photometry for an asteroid.

    One row per observation, with the originating session's `block_id` and
    metadata carried alongside so sessions remain distinguishable (needed
    because each session has its own magnitude zero-point).
    """
    frames = []
    for block in load_alcdef_blocks(number, zip_path):
        frames.append(
            pd.DataFrame(
                {
                    "jd": block.jd,
                    "mag": block.mag,
                    "mag_err": block.mag_err,
                    "block_id": block.block_id,
                    "session_date": block.session_date,
                    "filter": block.filter,
                    "magband": block.magband,
                    "is_differential": block.is_differential,
                }
            )
        )
    if not frames:
        return pd.DataFrame(columns=_PHOTOMETRY_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def object_summary(number: int, zip_path: Path | str = ALCDEF_ZIP_PATH) -> dict:
    """Quick per-object coverage stats for EDA (no periodogram, just counts/spans)."""
    blocks = load_alcdef_blocks(number, zip_path)
    n_points = sum(len(b) for b in blocks)
    all_jd = np.concatenate([b.jd for b in blocks]) if blocks else np.array([])
    bands = {b.magband for b in blocks if b.magband}
    return {
        "number": int(number),
        "n_sessions": len(blocks),
        "n_points": int(n_points),
        # distinct integer JDs ~ distinct nights of observation
        "n_nights": int(np.unique(np.floor(all_jd)).size) if all_jd.size else 0,
        "jd_span_days": float(all_jd.max() - all_jd.min()) if all_jd.size else 0.0,
        "bands": sorted(bands),
    }

