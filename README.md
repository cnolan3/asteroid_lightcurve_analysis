# Recovering Asteroid Rotation Periods from Sparse Light Curves

**Capstone project — Unit 20: Initial Report & Exploratory Data Analysis**

## Research question

Can a supervised machine-learning model accurately recover an asteroid's rotation
period from sparse, irregularly-sampled light-curve data — telling the true period
apart from the aliases that simpler methods produce — reliably enough to be useful
on large survey datasets where dedicated follow-up isn't possible?

## Why it matters

An asteroid's spin rate constrains its internal structure: whether it is a solid
chunk of rock or metal, or a gravitationally-bound "rubble pile." That has
scientific value and bears on planetary defense. We have just entered an era of
asteroid data on an unprecedented scale — the Rubin Observatory is expected to
roughly triple the number of known asteroids — but it observes each object only a
handful of times, producing **sparse** data rather than the dense observations
traditionally needed to measure a rotation period. There aren't enough telescopes
to re-observe millions of asteroids individually. A model that pulls reliable
periods from sparse data turns observations that are already being collected into
usable measurements at no extra cost, and flags the few objects where dedicated
follow-up is actually worth it.

## Data sources & structure

| Source | Role | Structure |
|---|---|---|
| **LCDB** (Asteroid Lightcurve Database) | Labels — published rotation periods + quality codes | Fixed-width summary table, one row per asteroid (~36k rows) |
| **ALCDEF** (Asteroid Lightcurve Data Exchange Format) | Dense light curves — raw material to down-sample | One file per asteroid; per-session `(JD, magnitude, uncertainty)` blocks (24,643 objects) |
| ZTF / Gaia DR3 *(planned, Unit 24)* | Real-world sparse test sets | Wide-survey sparse photometry |

Labels are filtered to LCDB quality code **U ≥ 2** (reliable determinations). An
asteroid enters the study only if it has both a reliable period and an ALCDEF
curve, giving **~20,500 labeled asteroids**. Genuinely *dense* curves suitable as
down-sampling sources (≥ 100 points, with a ≥ 30-point session) number **12,785**.

## Techniques

1. **Controlled-sparsity pipeline** — assemble a calibrated dense source curve per
   asteroid (concatenate sessions, remove per-session magnitude zero-points, reject
   outliers), then progressively down-sample it to a target number of observations,
   spreading the kept points across nights to mimic survey epoch spacing.
2. **Lomb–Scargle baseline** — the classical periodogram for unevenly-sampled data,
   run (unweighted) on each sparse curve; its strongest peak is the estimated
   period. This is the non-ML reference the model must beat. Scoring is alias-aware
   (asteroid curves are double-peaked, so the 2× / 0.5× harmonic counts as correct).
3. **Feature extraction + supervised classifier** — turn each sparse curve and its
   periodogram into a 14-feature vector (sampling geometry, photometric amplitude
   and scatter, periodogram shape), and train a random forest to predict whether the
   Lomb–Scargle period is correct — the "true period vs. alias" task.

## Results

**Lomb–Scargle recovery degrades sharply with sparsity.** Period recovery is
reliable with hundreds of points but collapses in the sparse regime that surveys
actually deliver, and is strongly amplitude-dependent (elongated asteroids stay
recoverable far longer than near-round ones).

| Observations retained | 200 | 100 | 50 | 30 | 20 | 10 | 5 |
|---|---|---|---|---|---|---|---|
| **Recovery rate** | 74% | 64% | 52% | 41% | 32% | 15% | 7% |

**A random forest identifies which sparse-data periods to trust**, predicting
period correctness with **ROC-AUC 0.899** on a held-out set of asteroids (split by
object to prevent leakage). It clearly beats the obvious single-number Lomb–Scargle
confidence signals:

| Model | ROC-AUC |
|---|---|
| **Random forest (all features)** | **0.899** |
| Logistic regression | 0.871 |
| LS false-alarm probability only | 0.833 |
| LS peak power only | 0.406 |

A notable finding: raw peak power alone is *worse than random* (0.406), because
with few points almost any period fits well and inflates the power — precisely the
trap a naive "trust strong peaks" rule falls into on sparse data, and precisely
what the model (which accounts for the number of observations) avoids.

**The practical payoff** is a high-purity period catalog: keeping only the periods
the model is most confident in yields a catalog that is **90% pure while retaining
28% of objects**, versus the raw Lomb–Scargle purity of 40%. The most informative
features are the periodogram false-alarm probability, the number of competing
peaks, peak dominance, and the number of observations — matching physical
intuition. This is exactly the capability the Rubin-era use case needs: automatically
flag the sparse-data periods worth believing.

Full analysis, figures, and code are in
[`asteroid_rotation_technical.ipynb`](asteroid_rotation_technical.ipynb).

## Repository structure

```
asteroid_rotation_technical.ipynb   # main technical report (this Unit's deliverable)
requirements.txt
src/
  data_loading.py     # parse LCDB labels + ALCDEF light curves
  sparsity.py         # controlled down-sampling pipeline
  baseline.py         # Lomb–Scargle periodogram baseline
  features.py         # feature extraction for the classifier
data/                 # raw/interim/processed (git-ignored; see below)
figures/              # saved figures used in the report
docs/                 # capstone overview, proposal, unit requirements
```

## Reproducing

```bash
# environment (conda; the scientific stack from conda-forge)
conda create -n asteroid-lc -c conda-forge -y python=3.12 \
  numpy pandas scipy matplotlib astropy scikit-learn jupyter
conda activate asteroid-lc

# raw data (git-ignored; download into data/raw/)
#   LCDB : https://minplanobs.org/mpinfo/datazips/LCLIST_PUB_2023OCT.zip  -> data/raw/lcdb/
#   ALCDEF: https://alcdef.org/docs/ALCDEF_ALL.zip                        -> data/raw/alcdef/

jupyter notebook asteroid_rotation_technical.ipynb
```

## Next steps (Unit 24)

- Richer models: gradient-boosted trees and a CNN on the periodogram / phase-folded
  representation.
- A candidate-ranking framing that can *recover* periods Lomb–Scargle misses, not
  only filter its top pick.
- Validation on real sparse survey data (ZTF, Gaia DR3), where the sampling is
  genuinely sparse rather than simulated by down-sampling.
