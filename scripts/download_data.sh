#!/usr/bin/env bash
# Download and lay out the raw data for the asteroid rotation-period project.
#
# Produces the layout the code expects (see src/data_loading.py):
#   data/raw/lcdb/lc_summary_pub.txt    <- extracted from the LCDB release zip
#   data/raw/alcdef/ALCDEF_ALL.zip      <- kept AS A ZIP (light curves are read
#                                          directly out of the archive)
#
# Safe to re-run: existing files are kept, interrupted downloads resume.

set -euo pipefail

LCDB_URL="https://minplanobs.org/mpinfo/datazips/LCLIST_PUB_2023OCT.zip"
ALCDEF_URL="https://alcdef.org/docs/ALCDEF_ALL.zip"

# Repo root = parent of this script's directory, so it works from anywhere.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW="$ROOT/data/raw"
LCDB_ZIP="$RAW/$(basename "$LCDB_URL")"
LCDB_TXT="$RAW/lcdb/lc_summary_pub.txt"
ALCDEF_ZIP="$RAW/alcdef/$(basename "$ALCDEF_URL")"

mkdir -p "$RAW/lcdb" "$RAW/alcdef"

fetch() { # fetch <url> <dest> — resumable, atomic (downloads to .part first)
    local url="$1" dest="$2"
    echo "downloading $(basename "$dest") ..."
    curl --fail --location --continue-at - --output "$dest.part" "$url"
    mv "$dest.part" "$dest"
}

# --- LCDB labels: download the release zip, extract the summary table -------
if [[ -f "$LCDB_TXT" ]]; then
    echo "ok: $LCDB_TXT already present, skipping LCDB"
else
    [[ -f "$LCDB_ZIP" ]] || fetch "$LCDB_URL" "$LCDB_ZIP"
    # The zip also ships a readme + frequency-diameter plots; extract it all,
    # the loader only reads lc_summary_pub.txt.
    unzip -o -q "$LCDB_ZIP" -d "$RAW/lcdb"
    [[ -f "$LCDB_TXT" ]] || { echo "error: $LCDB_TXT missing after unzip" >&2; exit 1; }
    echo "ok: extracted $(basename "$LCDB_ZIP") -> data/raw/lcdb/"
fi

# --- ALCDEF light curves: keep the archive zipped where the loader expects it
if [[ -f "$ALCDEF_ZIP" ]]; then
    echo "ok: $ALCDEF_ZIP already present, skipping ALCDEF"
else
    echo "note: ALCDEF_ALL.zip is large (hundreds of MB); this may take a while"
    fetch "$ALCDEF_URL" "$ALCDEF_ZIP"
    # Sanity-check the archive without extracting it.
    unzip -t -qq "$ALCDEF_ZIP" > /dev/null \
        || { echo "error: $ALCDEF_ZIP failed integrity check" >&2; exit 1; }
    echo "ok: $(basename "$ALCDEF_ZIP") -> data/raw/alcdef/ (left zipped)"
fi

echo
echo "data/raw is ready:"
ls -lh "$LCDB_TXT" "$ALCDEF_ZIP"
