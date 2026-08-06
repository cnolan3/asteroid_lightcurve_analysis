"""1D CNN experiment: judge period reliability from the periodogram itself.

The random forest (notebook Section 10) sees six summary statistics of the
Lomb-Scargle periodogram; the GRU (Section 12) sees no periodogram at all. This
model reads the **entire periodogram** — power on a fixed uniform frequency
grid — as a 1D signal. The motivation is that aliases carry a structured
signature the summary statistics compress away: a false peak produced by
once-per-day sampling sits in a comb of sidelobes offset by integer cycles/day
from the true frequency. On a *linear* frequency grid those offsets are the
same number of grid points everywhere, so a convolution along frequency can
learn the comb as a translation-invariant motif.

Input per row: 3 channels x N_GRID points —
    [power, power / max power, frequency (normalized)]
plus a scalar context [log10 n_obs, log10 baseline_days] joined before the
head, because the meaning of a given peak height depends on how many points
produced it (the same fact the false-alarm probability encodes).

The frequency grid spans the same 1-48 h search range as the baseline and the
feature extractor. Periodograms are computed once from the cached sequence
dataset (`rnn.SEQ_PATH`) and stored in float16 (power is 0-1; ~1e-3 precision
is ample).

Training reuses `rnn.train_model` / `rnn.predict_scores`: encoded batches are
(X, context, lengths, y) and the model's forward accepts (x, ctx, lengths),
ignoring lengths (all periodograms have the same length).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from baseline import MAX_PERIOD_H, MIN_PERIOD_H

PGRAM_PATH = "data/processed/periodograms.npz"

N_GRID = 4096
FREQ_GRID = np.linspace(24.0 / MAX_PERIOD_H, 24.0 / MIN_PERIOD_H, N_GRID)

N_CHANNELS = 3            # power, power/max, normalized frequency
N_CONTEXT_CNN = 2         # log10 n_obs, log10 baseline_days


# --------------------------------------------------------------------------
# dataset construction
# --------------------------------------------------------------------------

def build_periodogram_dataset(seqs: dict, *, verbose: bool = True) -> np.ndarray:
    """Lomb-Scargle power on FREQ_GRID for every cached sequence row.

    Returns an (n_rows, N_GRID) float16 array aligned with the feature table's
    row order (the same order as `seqs["offsets"]`).
    """
    from astropy.timeseries import LombScargle

    offsets = seqs["offsets"]
    n_rows = len(offsets) - 1
    out = np.empty((n_rows, N_GRID), dtype=np.float16)
    for r in range(n_rows):
        sl = slice(offsets[r], offsets[r + 1])
        # unweighted, matching the baseline (see baseline.py)
        power = LombScargle(seqs["jd"][sl], seqs["mag"][sl]).power(FREQ_GRID)
        out[r] = np.clip(np.nan_to_num(power, nan=0.0), 0.0, 1.0)
        if verbose and (r + 1) % 2000 == 0:
            print(f"  periodograms {r + 1}/{n_rows}")
    return out


def save_periodograms(pgrams: np.ndarray, path=PGRAM_PATH) -> None:
    np.savez_compressed(path, power=pgrams)


def load_periodograms(path=PGRAM_PATH) -> np.ndarray:
    with np.load(path) as z:
        return z["power"]


def encode_rows_cnn(pgrams: np.ndarray, feat: pd.DataFrame, rows: np.ndarray):
    """Encode feature-table rows as (X, context, lengths, y) for the CNN.

    X: (n, N_CHANNELS, N_GRID) float32; lengths is a constant vector kept only
    for signature compatibility with `rnn.train_model`.
    """
    power = pgrams[rows].astype(np.float32)
    peak = power.max(axis=1, keepdims=True)
    freq_chan = np.broadcast_to(
        (FREQ_GRID / FREQ_GRID[-1]).astype(np.float32), power.shape)
    X = np.stack([power, power / np.maximum(peak, 1e-6), freq_chan], axis=1)

    ctx = np.stack([
        np.log10(np.maximum(feat["n_obs"].values[rows], 1)),
        np.log10(np.maximum(feat["baseline_days"].values[rows], 1e-2)),
    ], axis=1).astype(np.float32)

    lengths = np.full(len(rows), N_GRID, dtype=np.int64)
    y = feat["matched"].astype(int).values[rows]
    return X, ctx, lengths, y


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------

def make_cnn(dropout: float = 0.2):
    """1D conv stack over the periodogram + scalar-context head.

    Six stride-2 blocks after an initial max-pool give a receptive field of
    several cycles/day, wide enough to see a peak together with its +/-1 and
    +/-2 cycles/day alias sidelobes (~174 grid points per cycle/day).
    """
    import torch
    from torch import nn

    chans = [N_CHANNELS, 32, 48, 64, 96, 128, 128]

    class PeriodogramCNN(nn.Module):
        def __init__(self):
            super().__init__()
            blocks = [nn.MaxPool1d(2)]           # 4096 -> 2048, keeps peak heights
            for c_in, c_out in zip(chans[:-1], chans[1:]):
                blocks += [
                    nn.Conv1d(c_in, c_out, kernel_size=9, stride=2, padding=4),
                    nn.BatchNorm1d(c_out),
                    nn.ReLU(),
                ]
            self.conv = nn.Sequential(*blocks)   # -> (batch, 128, 32)
            self.head = nn.Sequential(
                nn.Linear(2 * chans[-1] + N_CONTEXT_CNN, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1),
            )

        def forward(self, x, ctx, lengths=None):
            h = self.conv(x)
            h = torch.cat([h.max(dim=2).values, h.mean(dim=2)], dim=1)
            return self.head(torch.cat([h, ctx], dim=1)).squeeze(1)

    return PeriodogramCNN()
