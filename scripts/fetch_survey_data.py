"""Fetch real sparse survey photometry (ZTF + Gaia DR3) for LCDB-labeled asteroids.

This is the data-acquisition step for the real-survey validation section of the
technical notebook: everything upstream of it is simulated sparsity; this pulls
the genuine article. It is a **raw archiver** — responses are saved to disk
exactly as the services return them, and all parsing/photometric reduction
happens later (src/surveys.py), so a format surprise never corrupts a download.

Sources (per numbered asteroid, so the LCDB crossmatch is just the number):

  * **ZTF via the Fink broker** — one REST call per asteroid returns all of its
    ZTF alert photometry with IMCCE/Miriade ephemerides attached (phase angle
    and distances ride along with `withEphem=true`), i.e. photometry + viewing
    geometry in a single call.
        -> data/raw/surveys/ztf/<number>.json
  * **Gaia DR3 `sso_observation`** — bulk TAP queries (chunks of candidates)
    for the per-epoch G-band photometry of the same candidate list.
        -> data/raw/surveys/gaia/chunk_<i>.json
  * **Miriade ephemerides for the Gaia epochs** — Gaia's table has no
    distances/phase, so for each asteroid with Gaia data we ask Miriade for the
    geometry at exactly those epochs.
        -> data/raw/surveys/miriade/<number>.json

Sample: a seeded random draw of eligible asteroids (reliable LCDB label with
U >= 2 and true period inside the baseline's 2-24 h evaluation range). The draw
is a random *permutation*, walked in order until TARGET_ZTF objects have a
usable ZTF light curve (>= MIN_ZTF_POINTS alerts), so re-runs are
deterministic. The candidate list is written to
data/raw/surveys/candidates.csv for the notebook to reuse.

Every step is resumable: existing files are never re-fetched (delete a file to
force a re-fetch). Failures are logged and skipped; re-run the script to retry.
Runtime is dominated by polite per-call sleeps: roughly 10-25 min on the first
run. Requires network access to api.fink-portal.org, gea.esac.esa.int, and
ssp.imcce.fr.

Run inside the project environment:  python scripts/fetch_survey_data.py
"""

import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from baseline import EVAL_MAX_PERIOD_H, EVAL_MIN_PERIOD_H
from data_loading import load_lcdb_summary, reliable_labels

SAMPLE_SEED = 20260820
N_CANDIDATES = 800        # permutation length walked until the ZTF target is met
TARGET_ZTF = 300          # stop ZTF fetching after this many usable objects
MIN_ZTF_POINTS = 20       # "usable" = at least this many ZTF alerts
MIN_GAIA_POINTS = 10      # fetch Miriade geometry only above this many epochs
SLEEP_S = 1.0             # politeness delay between per-object calls
TIMEOUT_S = 60
GAIA_CHUNK = 200          # candidates per TAP query

# Fink has moved hosts before; try each in order. Their docs use POST+JSON,
# but GET is kept as a fallback for older deployments.
FINK_URLS = ("https://api.fink-portal.org/api/v1/sso",
             "https://fink-portal.org/api/v1/sso")
GAIA_TAP_URL = "https://gea.esac.esa.int/tap-server/tap/sync"
MIRIADE_URL = "https://ssp.imcce.fr/webservices/miriade/api/ephemcc.php"

OUT = ROOT / "data" / "raw" / "surveys"


def fetch_with_retries(method, url, *, tries=3, backoff=10.0, **kwargs):
    """One HTTP call with simple retry/backoff. Returns Response or None."""
    for attempt in range(tries):
        try:
            resp = requests.request(method, url, timeout=TIMEOUT_S, **kwargs)
            if resp.status_code == 200:
                return resp
            print(f"    HTTP {resp.status_code} from {url.split('/')[2]} "
                  f"(attempt {attempt + 1}/{tries})", flush=True)
        except requests.RequestException as exc:
            print(f"    {type(exc).__name__} from {url.split('/')[2]} "
                  f"(attempt {attempt + 1}/{tries})", flush=True)
        time.sleep(backoff * (attempt + 1))
    return None


# ----------------------------------------------------------------------------
# 0. candidate sample (deterministic, written once)
# ----------------------------------------------------------------------------

def candidate_sample() -> pd.DataFrame:
    path = OUT / "candidates.csv"
    if path.exists():
        cand = pd.read_csv(path)
        print(f"candidates: reusing existing list ({len(cand)} asteroids)")
        return cand

    labels = reliable_labels(load_lcdb_summary(), min_u=2.0)
    elig = labels[
        labels["number"].notna()
        & labels["period_h"].between(EVAL_MIN_PERIOD_H, EVAL_MAX_PERIOD_H)
    ][["number", "name", "period_h", "U", "amp_max", "class"]].copy()
    elig["number"] = elig["number"].astype(int)

    rng = np.random.default_rng(SAMPLE_SEED)
    pick = rng.permutation(len(elig))[:N_CANDIDATES]
    cand = elig.iloc[np.sort(pick)].reset_index(drop=True)
    # walk order is the permutation order, not number order
    cand = cand.iloc[rng.permutation(len(cand))].reset_index(drop=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    cand.to_csv(path, index=False)
    print(f"candidates: sampled {len(cand)} of {len(elig)} eligible asteroids "
          f"(U >= 2, period {EVAL_MIN_PERIOD_H}-{EVAL_MAX_PERIOD_H} h)")
    return cand


# ----------------------------------------------------------------------------
# 1. ZTF photometry via Fink (per object, with ephemerides attached)
# ----------------------------------------------------------------------------

def fink_reachable() -> bool:
    for url in FINK_URLS:
        try:
            requests.get(url.rsplit("/", 1)[0] + "/columns", timeout=15)
            return True
        except requests.RequestException:
            continue
    return False


def fetch_ztf(cand: pd.DataFrame) -> None:
    if not fink_reachable():
        print("\nZTF/Fink: service unreachable -- skipping the ZTF step "
              "entirely (re-run later; downloads resume where they left off)")
        return
    zdir = OUT / "ztf"
    zdir.mkdir(parents=True, exist_ok=True)

    def n_alerts(path: Path) -> int:
        try:
            return len(json.loads(path.read_text()))
        except Exception:
            return 0

    usable = sum(n_alerts(p) >= MIN_ZTF_POINTS for p in zdir.glob("*.json"))
    print(f"\nZTF/Fink: {usable} usable objects already on disk "
          f"(target {TARGET_ZTF})")

    for row in cand.itertuples():
        if usable >= TARGET_ZTF:
            break
        path = zdir / f"{row.number}.json"
        if path.exists():
            continue
        resp = None
        for url in FINK_URLS:
            resp = fetch_with_retries(
                "POST", url, tries=2,
                json={"n_or_d": str(row.number), "withEphem": True,
                      "output-format": "json"})
            if resp is None:    # POST refused/unreachable -> legacy GET form
                resp = fetch_with_retries(
                    "GET", url, tries=1,
                    params={"n_or_d": str(row.number), "withEphem": "true",
                            "output-format": "json"})
            if resp is not None:
                break
        if resp is None:
            print(f"  {row.number}: FAILED, will retry on next run", flush=True)
            continue
        path.write_text(resp.text)
        n = n_alerts(path)
        if n >= MIN_ZTF_POINTS:
            usable += 1
        print(f"  {row.number} ({row.name}): {n} alerts "
              f"[{usable}/{TARGET_ZTF} usable]", flush=True)
        time.sleep(SLEEP_S)

    print(f"ZTF/Fink done: {usable} usable objects")


# ----------------------------------------------------------------------------
# 2. Gaia DR3 sso_observation via TAP (bulk, chunked)
# ----------------------------------------------------------------------------

def fetch_gaia(cand: pd.DataFrame) -> None:
    gdir = OUT / "gaia"
    gdir.mkdir(parents=True, exist_ok=True)
    numbers = sorted(cand["number"].astype(int))
    chunks = [numbers[i:i + GAIA_CHUNK] for i in range(0, len(numbers), GAIA_CHUNK)]
    print(f"\nGaia TAP: {len(chunks)} chunks of <= {GAIA_CHUNK} asteroids")

    for i, chunk in enumerate(chunks):
        path = gdir / f"chunk_{i}.json"
        if path.exists():
            print(f"  chunk {i}: already on disk")
            continue
        adql = ("SELECT * FROM gaiadr3.sso_observation "
                f"WHERE number_mp IN ({','.join(map(str, chunk))})")
        resp = fetch_with_retries(
            "POST", GAIA_TAP_URL,
            data={"REQUEST": "doQuery", "LANG": "ADQL",
                  "FORMAT": "json", "QUERY": adql})
        if resp is None:
            print(f"  chunk {i}: FAILED, will retry on next run", flush=True)
            continue
        path.write_text(resp.text)
        try:
            nrows = len(json.loads(resp.text).get("data", []))
        except Exception:
            nrows = -1
        print(f"  chunk {i}: {nrows} observation rows", flush=True)
        time.sleep(SLEEP_S)


# ----------------------------------------------------------------------------
# 3. Miriade geometry at the Gaia epochs (per object)
# ----------------------------------------------------------------------------

def gaia_epochs_by_number() -> dict[int, list[float]]:
    """{asteroid number: sorted UTC JD epochs} from the downloaded Gaia chunks."""
    out: dict[int, list[float]] = {}
    for path in sorted((OUT / "gaia").glob("chunk_*.json")):
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        cols = [c["name"] for c in payload.get("metadata", [])]
        if "number_mp" not in cols or "epoch_utc" not in cols:
            continue
        i_num, i_ep = cols.index("number_mp"), cols.index("epoch_utc")
        for row in payload.get("data", []):
            if row[i_ep] is None:
                continue
            # epoch_utc is nominally days since JD 2455197.5 (2010-01-01, the
            # Gaia convention); guard in case the archive serves absolute JD
            ep = float(row[i_ep])
            jd = ep if ep > 2.4e6 else ep + 2455197.5
            out.setdefault(int(row[i_num]), []).append(jd)
    return {k: sorted(v) for k, v in out.items()}


def fetch_miriade(epochs_by_number: dict[int, list[float]]) -> None:
    mdir = OUT / "miriade"
    mdir.mkdir(parents=True, exist_ok=True)
    todo = {n: eps for n, eps in epochs_by_number.items()
            if len(eps) >= MIN_GAIA_POINTS}
    print(f"\nMiriade: geometry for {len(todo)} asteroids with "
          f">= {MIN_GAIA_POINTS} Gaia epochs")

    for k, (num, epochs) in enumerate(sorted(todo.items())):
        path = mdir / f"{num}.json"
        if path.exists():
            continue
        # POST an explicit epoch list; observer 258 = Gaia's MPC code, so the
        # distances/phase are Gaia-centric rather than geocentric
        resp = fetch_with_retries(
            "POST", MIRIADE_URL,
            data={"-name": f"a:{num}", "-type": "Asteroid",
                  "-tscale": "UTC", "-observer": "258",
                  "-mime": "json", "-output": "--jd"},
            files={"epochs": ("epochs",
                              "\n".join(f"{e:.8f}" for e in epochs))})
        if resp is None:
            print(f"  {num}: FAILED, will retry on next run", flush=True)
            continue
        path.write_text(resp.text)
        if (k + 1) % 25 == 0:
            print(f"  {k + 1}/{len(todo)} done", flush=True)
        time.sleep(SLEEP_S)

    print("Miriade done")


def connectivity_report() -> None:
    """One quick probe per service so failures are visible before any work."""
    probes = [("ZTF/Fink", FINK_URLS[0].rsplit("/", 1)[0] + "/columns"),
              ("ZTF/Fink (alt)", FINK_URLS[1].rsplit("/", 1)[0] + "/columns"),
              ("Gaia TAP", GAIA_TAP_URL.rsplit("/", 2)[0] + "/tap/capabilities"),
              ("Miriade", MIRIADE_URL)]
    print("connectivity:")
    for name, url in probes:
        try:
            code = requests.get(url, timeout=15).status_code
            print(f"  {name:<16} HTTP {code}")
        except requests.RequestException as exc:
            print(f"  {name:<16} UNREACHABLE ({type(exc).__name__}: {exc})")


if __name__ == "__main__":
    # optionally run a subset of steps: python fetch_survey_data.py gaia miriade
    steps = set(sys.argv[1:]) or {"ztf", "gaia", "miriade"}
    OUT.mkdir(parents=True, exist_ok=True)
    connectivity_report()
    cand = candidate_sample()
    if "ztf" in steps:
        fetch_ztf(cand)
    if "gaia" in steps:
        fetch_gaia(cand)
    if "miriade" in steps:
        fetch_miriade(gaia_epochs_by_number())

    n_ztf = len(list((OUT / "ztf").glob("*.json")))
    n_gaia = len(list((OUT / "gaia").glob("chunk_*.json")))
    n_mir = len(list((OUT / "miriade").glob("*.json")))
    print(f"\nsummary: {n_ztf} ZTF files, {n_gaia} Gaia chunks, "
          f"{n_mir} Miriade files under {OUT}")
