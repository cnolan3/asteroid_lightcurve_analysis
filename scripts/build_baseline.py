"""Build the full baseline sweep (data/interim/baseline_results.parquet).

Mirrors the notebook's fallback build (Section 9.2, the BASE_PATH cell) at the
full size the report describes: 1,500 sampled asteroids x 7 sparsity levels.
Sampling follows the same eligibility as scripts/build_features.py — reliable
label, true period inside the 2-24 h evaluation range, dense source pool — so
every sparsity level is a genuine down-sampling of a dense curve. (This is why
the cached sweep's recovery rates at 100-200 points exceed the notebook's small
unrestricted fallback: sparse "sources" cannot actually deliver 200 points.)

SAMPLE_SEED matches the notebook fallback's rng; run_baseline uses seed 0 with
the sample in sorted order, the same per-object seeding convention as the rest
of the pipeline.

Run inside the project environment:  python scripts/build_baseline.py
(~5-10 min; the coverage cache under data/interim/ is built automatically if
missing, so the only prerequisite is the raw data from scripts/download_data.sh.)
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from baseline import EVAL_MAX_PERIOD_H, EVAL_MIN_PERIOD_H, run_baseline
from data_loading import (
    build_coverage_table,
    has_alcdef,
    load_lcdb_summary,
    reliable_labels,
)
from sparsity import select_dense_pool

N_ASTEROIDS = 1500
SAMPLE_SEED = 20260722          # same rng as the notebook's fallback cell
RUN_SEED = 0

COV_PATH = ROOT / "data" / "interim" / "coverage_summary.parquet"
OUT_PATH = ROOT / "data" / "interim" / "baseline_results.parquet"


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
        sys.exit(f"{OUT_PATH} already exists -- delete it first to rebuild.")

    labels = reliable_labels(load_lcdb_summary(), min_u=2.0)

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
    base = run_baseline(with_progress(sample), lookup, seed=RUN_SEED).merge(
        data.set_index("number")[["amp_max", "class"]],
        left_on="number", right_index=True, how="left")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    base.to_parquet(OUT_PATH, index=False)

    print(f"\nwrote {OUT_PATH}  ({time.time() - t0:,.0f} s)")
    print(f"  {len(base):,} trials | {base['number'].nunique():,} asteroids")
    print("recovery rate by level:")
    print((base.groupby('level')['matched'].mean() * 100).round(1).to_string())


if __name__ == "__main__":
    main()
