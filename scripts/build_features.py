"""Build the full feature table (data/processed/features.parquet).

Mirrors the notebook's fallback build (Section 10, cell "FEAT_PATH") but at the
full sample size the report describes: ~2,000 asteroids x 7 sparsity levels.
The sample is drawn from the modeling dataset (reliable label + ALCDEF curve),
restricted to

  * true period inside the baseline's evaluation range (2-24 h), and
  * the dense source pool (>= 100 points, with a >= 30-point session), so every
    sparsity level is a genuine down-sampling of a dense curve (Section 8).

Seeding matters and is deliberately identical to the notebook:

  * SAMPLE_SEED picks WHICH asteroids (the notebook fallback's 424242);
  * build seed 0 with the sample in sorted order fixes each object's
    down-sampling draws -- neural.build_sequence_dataset regenerates curves as
    per-object seed = BUILD_SEED + index over sorted numbers and asserts them
    against this table, so do not change either without changing neural.py.

Run inside the project environment:  python scripts/build_features.py
(~10-30 min; progress is printed every 100 asteroids. The per-object coverage
cache under data/interim/ is built automatically if missing, so the only
prerequisite is the raw data from scripts/download_data.sh.)
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from baseline import EVAL_MAX_PERIOD_H, EVAL_MIN_PERIOD_H
from data_loading import (
    build_coverage_table,
    has_alcdef,
    load_lcdb_summary,
    reliable_labels,
)
from features import build_feature_table
from sparsity import select_dense_pool

N_ASTEROIDS = 2000
SAMPLE_SEED = 424242            # same convention as the notebook fallback cell
BUILD_SEED = 0                  # must match neural.BUILD_SEED

COV_PATH = ROOT / "data" / "interim" / "coverage_summary.parquet"
OUT_PATH = ROOT / "data" / "processed" / "features.parquet"


def with_progress(nums, every=100):
    t0 = time.time()
    for i, n in enumerate(nums):
        if i and i % every == 0:
            el = time.time() - t0
            print(f"  {i}/{len(nums)} asteroids  ({el/60:.1f} min elapsed, "
                  f"~{el/i*(len(nums)-i)/60:.1f} min remaining)", flush=True)
        yield n


def main() -> None:
    if OUT_PATH.exists():
        sys.exit(f"{OUT_PATH} already exists -- delete it first to rebuild "
                 f"(the cached .npz files under data/processed/ must then be "
                 f"deleted too, they are keyed to this table).")

    labels = reliable_labels(load_lcdb_summary(), min_u=2.0)

    # Coverage cache: reuse the notebook's if present, otherwise build it the
    # same way Section 5 does (and save it where the notebook will find it).
    if COV_PATH.exists():
        cov = pd.read_parquet(COV_PATH)
    else:
        print("coverage cache missing -- building it (one pass over the "
              "ALCDEF zip, a few minutes) ...", flush=True)
        nums = sorted(int(n) for n in labels["number"].dropna()
                      if has_alcdef(int(n)))
        cov = build_coverage_table(nums)
        COV_PATH.parent.mkdir(parents=True, exist_ok=True)
        cov.to_parquet(COV_PATH, index=False)
        print(f"wrote {COV_PATH}  ({len(cov):,} objects)")
    data = labels.merge(cov, on="number", how="inner")

    dense = set(select_dense_pool(cov, min_points=100, min_session_points=30))
    elig = data[data["period_h"].between(EVAL_MIN_PERIOD_H, EVAL_MAX_PERIOD_H)
                & data["number"].isin(dense)]
    lookup = dict(zip(elig["number"].astype(int), elig["period_h"]))
    print(f"eligible (U>=2, period {EVAL_MIN_PERIOD_H:.0f}-{EVAL_MAX_PERIOD_H:.0f} h, "
          f"dense source): {len(lookup):,} asteroids; sampling {N_ASTEROIDS:,}")

    rng = np.random.default_rng(SAMPLE_SEED)
    sample = sorted(int(x) for x in
                    rng.choice(list(lookup), size=N_ASTEROIDS, replace=False))

    t0 = time.time()
    feat = build_feature_table(with_progress(sample), lookup, seed=BUILD_SEED)
    feat = feat.merge(data.set_index("number")[["amp_max", "class"]],
                      left_on="number", right_index=True, how="left")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    feat.to_parquet(OUT_PATH, index=False)

    print(f"\nwrote {OUT_PATH}  ({time.time() - t0:,.0f} s)")
    print(f"  {len(feat):,} rows | {feat['number'].nunique():,} asteroids | "
          f"levels {sorted(feat['level'].unique())}")
    print(f"  positive rate (LS period correct): {feat['matched'].mean()*100:.1f}%")


if __name__ == "__main__":
    main()
