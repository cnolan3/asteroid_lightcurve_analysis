"""Neural-network experiments: period reliability from raw data, not features.

The tabular classifiers in the technical notebook judge a Lomb-Scargle period
from hand-crafted summary features. This module holds the two deep-learning
counter-experiments, which give the same judgement different raw inputs:

* **CurveGRU** — a bidirectional GRU over the *raw observation sequence*,
  phase-folded at the candidate period. At the true period the folded points
  trace a coherent repeating shape, at an alias they stay scattered, so the
  fold converts "is this period correct?" into "does this sequence look
  coherent?". Each timestep carries ``[sin 2*pi*phase, cos 2*pi*phase,
  delta_phase, mag, mag_err]`` (the sin/cos pair avoids a fake discontinuity
  at phase 1 -> 0); magnitudes are fed in real (session-zero-point-removed)
  units so amplitude information survives. The network sees **no periodogram
  features at all**.

* **PeriodogramCNN** — a 1D convolutional stack over the *entire Lomb-Scargle
  periodogram* on a fixed 4,096-point frequency grid. A daily alias comes with
  a comb of sidelobes offset by integer cycles/day from the true frequency;
  on a *linear* frequency grid those offsets are the same number of grid
  points everywhere, so a convolution sliding along frequency can learn the
  comb as a position-independent motif. Channels: ``[power, power/max power,
  normalized frequency]``.

Both models also receive a small scalar context vector (log n_obs, ...)
joined to the encoder output before the classification head, because the
meaning of what they see depends on how many observations produced it — the
same fact the false-alarm probability encodes for the periodogram.

Reproducibility. ``build_sequence_dataset`` regenerates, bit-for-bit, the same
down-sampled curves that produced ``data/processed/features.parquet`` (sorted
asteroid numbers, per-object seed = BUILD_SEED + index, levels in
DEFAULT_SPARSITY_LEVELS order), asserting every row against the cached
``n_obs`` and ``mag_std`` so a seed or ordering mismatch fails loudly. Every
sequence therefore lines up with a cached feature row and its label, and the
neural models can be scored on exactly the same asteroid-grouped train/test
split as the tabular models. ``build_periodogram_dataset`` derives the CNN
inputs from those same cached sequences.

Torch is imported lazily (inside the model/training functions) so the rest of
the pipeline works without it installed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from baseline import MAX_PERIOD_H, MIN_PERIOD_H
from sparsity import (
    DEFAULT_SPARSITY_LEVELS,
    densest_apparition,
    downsample,
    prepare_dense_curve,
)

# Seed used when data/processed/features.parquet was built.
BUILD_SEED = 0

SEQ_PATH = "data/processed/rnn_sequences.npz"
PGRAM_PATH = "data/processed/periodograms.npz"

# --- GRU input dimensions ---
N_STEP_FEATURES = 5   # sin, cos, dphase, mag, mag_err
N_CONTEXT = 3         # log10 n_obs, log10 period, log10 baseline_days
MAG_CLIP = 2.0        # clip zero-point-removed magnitudes to +/- this

# --- CNN input dimensions ---
N_GRID = 4096
FREQ_GRID = np.linspace(24.0 / MAX_PERIOD_H, 24.0 / MIN_PERIOD_H, N_GRID)
N_CHANNELS = 3        # power, power/max, normalized frequency
N_CONTEXT_CNN = 2     # log10 n_obs, log10 baseline_days


# --------------------------------------------------------------------------
# sequence dataset (GRU inputs)
# --------------------------------------------------------------------------

def build_sequence_dataset(feat: pd.DataFrame, *, seed: int = BUILD_SEED,
                           zip_path=None, verbose: bool = True) -> dict:
    """Regenerate the sparse curve behind every row of the feature table.

    Returns ragged sequences in concatenated form: {"jd", "mag", "mag_err"}
    concatenated over rows, with "offsets" (len = n_rows + 1) delimiting each
    row's slice, plus "number" and "level" aligned with `feat`'s row order.
    """
    kwargs = {} if zip_path is None else {"zip_path": zip_path}
    numbers = sorted(feat["number"].unique())
    by_row: dict[tuple[int, int], pd.DataFrame] = {}
    for k, num in enumerate(numbers):
        curve = densest_apparition(prepare_dense_curve(int(num), **kwargs))
        if len(curve) < 5:
            continue
        rng = np.random.default_rng(seed + k)
        for level in DEFAULT_SPARSITY_LEVELS:
            by_row[(int(num), int(level))] = downsample(curve, int(level), rng=rng)
        if verbose and (k + 1) % 200 == 0:
            print(f"  regenerated {k + 1}/{len(numbers)} asteroids")

    jd_parts, mag_parts, err_parts, lengths = [], [], [], []
    for row in feat.itertuples():
        sub = by_row[(int(row.number), int(row.level))]
        if len(sub) != int(row.n_obs) or not np.isclose(
                float(np.std(sub["mag"].values)), float(row.mag_std), rtol=1e-6):
            raise AssertionError(
                f"regenerated draw does not match cached features for "
                f"asteroid {row.number} level {row.level}")
        jd_parts.append(sub["jd"].values.astype(np.float64))
        mag_parts.append(sub["mag"].values.astype(np.float32))
        err_parts.append(sub["mag_err"].values.astype(np.float32))
        lengths.append(len(sub))

    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    return {
        "jd": np.concatenate(jd_parts),
        "mag": np.concatenate(mag_parts),
        "mag_err": np.concatenate(err_parts),
        "offsets": offsets,
        "number": feat["number"].values.astype(np.int64),
        "level": feat["level"].values.astype(np.int64),
    }


def save_sequences(seqs: dict, path=SEQ_PATH) -> None:
    np.savez_compressed(path, **seqs)


def load_sequences(path=SEQ_PATH) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def encode_rows(seqs: dict, feat: pd.DataFrame, rows: np.ndarray):
    """Encode feature-table rows as padded (X, context, lengths, y) arrays.

    X:       (n, max_len, N_STEP_FEATURES) phase-sorted, zero-padded
    context: (n, N_CONTEXT) scalar context
    lengths: (n,) true sequence lengths
    y:       (n,) labels (LS period correct?)
    """
    offsets = seqs["offsets"]
    periods = feat["ls_top_period_h"].values
    n_obs_all = feat["n_obs"].values
    base_days = feat["baseline_days"].values
    y_all = feat["matched"].astype(int).values

    lengths = (offsets[rows + 1] - offsets[rows]).astype(np.int64)
    max_len = int(lengths.max())
    X = np.zeros((len(rows), max_len, N_STEP_FEATURES), dtype=np.float32)
    ctx = np.zeros((len(rows), N_CONTEXT), dtype=np.float32)

    for i, r in enumerate(rows):
        sl = slice(offsets[r], offsets[r + 1])
        jd = seqs["jd"][sl]
        mag = np.clip(seqs["mag"][sl], -MAG_CLIP, MAG_CLIP)
        # some ALCDEF sessions report no per-point uncertainty -> NaN
        err = np.clip(np.nan_to_num(seqs["mag_err"][sl], nan=0.0), 0.0, 1.0)

        phase = ((jd * 24.0) % periods[r]) / periods[r]
        order = np.argsort(phase, kind="stable")
        phase, mag, err = phase[order], mag[order], err[order]
        dphase = np.diff(phase, prepend=0.0).astype(np.float32)

        n = len(jd)
        X[i, :n, 0] = np.sin(2 * np.pi * phase)
        X[i, :n, 1] = np.cos(2 * np.pi * phase)
        X[i, :n, 2] = dphase
        X[i, :n, 3] = mag
        X[i, :n, 4] = err
        ctx[i] = (np.log10(max(n_obs_all[r], 1)),
                  np.log10(max(periods[r], 1e-3)),
                  np.log10(max(base_days[r], 1e-2)))

    return X, ctx, lengths, y_all[rows]


# --------------------------------------------------------------------------
# periodogram dataset (CNN inputs)
# --------------------------------------------------------------------------

def build_periodogram_dataset(seqs: dict, *, verbose: bool = True) -> np.ndarray:
    """Lomb-Scargle power on FREQ_GRID for every cached sequence row.

    Returns an (n_rows, N_GRID) float16 array aligned with the feature table's
    row order (power is 0-1, so float16's ~1e-3 precision is ample).
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
    for signature compatibility with `train_model`.
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
# models
# --------------------------------------------------------------------------

def make_model(hidden: int = 64, layers: int = 1, dropout: float = 0.2):
    """Bidirectional GRU over the phase-folded sequence + scalar-context head."""
    import torch
    from torch import nn

    class CurveGRU(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(N_STEP_FEATURES, hidden, num_layers=layers,
                              batch_first=True, bidirectional=True,
                              dropout=dropout if layers > 1 else 0.0)
            self.head = nn.Sequential(
                nn.Linear(2 * hidden + N_CONTEXT, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1),
            )

        def forward(self, x, ctx, lengths):
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            _, h = self.gru(packed)          # h: (layers*2, batch, hidden)
            h = torch.cat([h[-2], h[-1]], dim=1)   # final layer, both directions
            return self.head(torch.cat([h, ctx], dim=1)).squeeze(1)

    return CurveGRU()


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


# --------------------------------------------------------------------------
# training / inference (shared by both models)
# --------------------------------------------------------------------------

def train_model(model, train_data, val_data, *, epochs: int = 30,
                batch_size: int = 128, lr: float = 1e-3, seed: int = 0,
                pos_weight: float | None = None, verbose: bool = True):
    """Train with early selection on validation ROC-AUC; returns history dict.

    The best-AUC epoch's weights are restored into `model` before returning.
    """
    import copy

    import torch
    from sklearn.metrics import roc_auc_score
    from torch import nn

    torch.manual_seed(seed)
    Xtr, ctr, ltr, ytr = train_data
    Xtr_t = torch.from_numpy(Xtr)
    ctr_t = torch.from_numpy(ctr)
    ltr_t = torch.from_numpy(ltr)
    ytr_t = torch.from_numpy(ytr.astype(np.float32))

    pw = None if pos_weight is None else torch.tensor(pos_weight)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    gen = torch.Generator().manual_seed(seed)

    history = {"train_loss": [], "val_auc": []}
    best_auc, best_state = -np.inf, None
    n = len(ytr_t)
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=gen)
        total = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            opt.zero_grad()
            out = model(Xtr_t[idx], ctr_t[idx], ltr_t[idx])
            loss = loss_fn(out, ytr_t[idx])
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(idx)
        sched.step()
        history["train_loss"].append(total / n)

        val_auc = roc_auc_score(val_data[3], predict_scores(model, val_data))
        history["val_auc"].append(val_auc)
        if val_auc > best_auc:
            best_auc, best_state = val_auc, copy.deepcopy(model.state_dict())
        if verbose:
            print(f"  epoch {epoch + 1:>2}/{epochs}  "
                  f"train loss {history['train_loss'][-1]:.4f}  "
                  f"val AUC {val_auc:.4f}")

    model.load_state_dict(best_state)
    return history


def predict_scores(model, data, batch_size: int = 512) -> np.ndarray:
    """Sigmoid scores for encoded rows (higher = more likely a correct period)."""
    import torch

    X, ctx, lengths, _ = data
    model.eval()
    outs = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            sl = slice(start, start + batch_size)
            out = model(torch.from_numpy(X[sl]), torch.from_numpy(ctx[sl]),
                        torch.from_numpy(lengths[sl]))
            outs.append(torch.sigmoid(out).numpy())
    return np.concatenate(outs)
