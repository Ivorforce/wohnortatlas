"""GENESIS / Regionaldatenbank (regionalstatistik.de) — per-Kreis sector employment
for the Branche Anbindung target (04h_jobs.py).

Pulls table 52111-07-01-4 (Unternehmensregister: "Abhängig Beschäftigte der
Niederlassungen (B-N, P-S) nach ausgewählten Wirtschaftsabschnitten, Kreise") and
folds the WZ-Abschnitte into the JOBS_BUCKETS. Workplace-based employment by sector
= the opportunity mass `O` a Branche target reaches.

Access notes (learned the hard way):
- POST-only (`data/table`); GET 405s. Credentials go in the HTTP HEADER (a Code-15
  "Zugangsdaten nicht erkannt" means they were in the body). Token in `username`
  (no password) OR `username`+`password` of a free account — set GENESIS_API_TOKEN,
  or GENESIS_USER + GENESIS_PASS. The `logincheck` method "succeeds" even anonymously,
  so it is NOT a credential test; only a data call authenticates.
- The response embeds a wide German ffcsv in JSON `Object`: `;`-delimited, a metadata
  preamble, then nested Nationalität/Geschlecht-style headers, then data rows
  `year;AGS;label;Insgesamt;<section…>`. ~398 real Kreise (5-digit AGS) + 74
  placeholder all-`-` rows (dropped). `-` = true zero, `. / x …` = Geheimhaltung (NaN).
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import csv as _csv
import io
import json
import re

import numpy as np
import pandas as pd

from wohnen.config import CACHE, GENESIS_API, JOBS_BUCKETS, JOBS_TABLE
from wohnen.dl import _session

# GENESIS missing-value markers: "-" is an exact zero, the rest are suppression.
_ZERO = {"-"}
_SUPPRESSED = {".", "/", "x", "...", ""}


def _creds() -> dict:
    """HTTP-header credentials: token in `username` (no password), or user+pass."""
    tok = os.environ.get("GENESIS_API_TOKEN")
    if tok:
        return {"username": tok}
    u, p = os.environ.get("GENESIS_USER"), os.environ.get("GENESIS_PASS")
    if u and p:
        return {"username": u, "password": p}
    raise RuntimeError(
        "GENESIS credentials missing — set GENESIS_API_TOKEN, or "
        "GENESIS_USER + GENESIS_PASS (free account at regionalstatistik.de).")


def _fetch_table_csv(name: str) -> str:
    """POST data/table (header auth) → the embedded ffcsv string, disk-cached."""
    cache = CACHE / "genesis" / f"{name}.csv"
    if cache.exists():
        return cache.read_text(encoding="utf-8")
    body = {"name": name, "area": "all", "format": "ffcsv",
            "compress": "false", "transpose": "false", "job": "false", "language": "de"}
    r = _session.post(f"{GENESIS_API}/data/table", data=body, headers=_creds(), timeout=180)
    r.raise_for_status()
    resp = r.json()
    status = resp.get("Status", {})
    if status.get("Code") not in (0, None):
        raise RuntimeError(f"GENESIS {name}: {status}")
    obj = resp.get("Object")
    csv_text = obj.get("Content") if isinstance(obj, dict) else obj
    if not isinstance(csv_text, str):
        raise RuntimeError(f"GENESIS {name}: unexpected Object {type(obj)}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(csv_text, encoding="utf-8")
    return csv_text


def _val(s: str):
    """One cell → float, with GENESIS markers: '-'→0.0, suppression→NaN."""
    s = s.strip()
    if s in _ZERO:
        return 0.0
    if s in _SUPPRESSED:
        return np.nan
    try:
        return float(s.replace(".", "").replace(",", "."))  # thousands-dot / decimal-comma safe
    except ValueError:
        return np.nan


def parse_jobs_table(csv_text: str) -> pd.DataFrame:
    """Wide ffcsv → DataFrame indexed by 5-digit Kreis AGS, one column per JOBS_BUCKETS
    key (employment). Sections summed per bucket; **Industrie residual-filled** where C
    is suppressed (`Insgesamt − Σ present non-Industrie sections`, recovering company
    towns like Wolfsburg). A single-section bucket whose only section is suppressed is
    NaN for that Kreis (→ 0 mass downstream)."""
    rows = list(_csv.reader(io.StringIO(csv_text), delimiter=";"))
    hdr = next(r for r in rows if any("Verarbeitendes Gewerbe" in c for c in r))
    # col0=year, col1=AGS, col2=label, col3=Insgesamt, col4.. = sections
    sec_codes = []
    for lab in hdr[3:]:
        m = re.search(r"\(([A-Z])\)", lab)
        sec_codes.append(m.group(1) if m else "Insg")  # col3 has no "(X)" → Insgesamt

    # AGS code length = regional level: 5-digit Kreis, 2-digit Land, 8-digit Gemeinde/Bezirk,
    # "DG" Deutschland. We want Kreise — but the Stadtstaaten that ARE a single Kreis (Berlin,
    # Hamburg) report their total only at the 2-digit Land code (their Bezirke ride at 8 digits).
    # Detect that GENERICALLY (no magic codes): a Land with no 5-digit Kreis rows is such a
    # city-state → alias its "LL" row to the Kreis AGS "LL000" the VG250 join uses. Bremen (04)
    # HAS 5-digit Kreise (04011/04012), so it is correctly NOT aliased.
    laender_with_kreise = {r[1].strip()[:2] for r in rows
                           if len(r) > 1 and re.fullmatch(r"\d{5}", r[1].strip())}

    industrie = set(JOBS_BUCKETS["industrie"])
    out = {}
    for r in rows:
        if len(r) < 4:
            continue
        reg = r[1].strip()
        if re.fullmatch(r"\d{5}", reg):
            ags = reg
        elif re.fullmatch(r"\d{2}", reg) and reg not in laender_with_kreise:
            ags = reg + "000"                           # Stadtstaat: the Land IS the Kreis
        else:
            continue                                    # DG / Land aggregate / 8-digit Bezirk
        # placeholder Kreise (74) carry "-" in every column incl. Insgesamt; a real Kreis
        # always has positive total employment. Drop by the raw Insgesamt cell (its "-" would
        # otherwise read as an exact 0 and survive as an all-zero Kreis).
        if r[3].strip() in _ZERO or r[3].strip() in _SUPPRESSED:
            continue
        vals = {code: _val(r[3 + j]) for j, code in enumerate(sec_codes) if 3 + j < len(r)}
        total = vals.get("Insg", np.nan)
        if not np.isfinite(total) or total <= 0:
            continue

        buckets = {}
        for key, codes in JOBS_BUCKETS.items():
            present = [vals.get(c, np.nan) for c in codes]
            s = np.nansum(present)
            buckets[key] = s if np.isfinite(s) and any(np.isfinite(present)) else np.nan

        # Industrie residual-fill: C suppressed → total − Σ(present non-industrie sections)
        if not np.isfinite(vals.get("C", np.nan)):
            nonind = [v for c, v in vals.items()
                      if c not in industrie and c != "Insg" and np.isfinite(v)]
            buckets["industrie"] = max(0.0, float(total) - float(np.sum(nonind)))
        out[ags] = buckets

    df = pd.DataFrame.from_dict(out, orient="index").reindex(columns=list(JOBS_BUCKETS))
    df.index.name = "kreis_ags"
    return df


def fetch_jobs_kreis() -> pd.DataFrame:
    """Per-Kreis sector employment (wide, bucket columns). Empty DataFrame + warning if
    credentials are unset (assemble then omits the Branche targets, like kreis_pay)."""
    try:
        csv_text = _fetch_table_csv(JOBS_TABLE)
    except RuntimeError as e:
        print(f"WARNING: GENESIS jobs skipped ({e})")
        return pd.DataFrame()
    df = parse_jobs_table(csv_text)
    print(f"genesis jobs: {len(df)} Kreise × {df.shape[1]} buckets "
          f"(Q non-null {df['gesundheit'].notna().mean():.0%}, "
          f"C-residual where suppressed)")
    return df


if __name__ == "__main__":
    # offline check: `python -m wohnen.genesis [path/to/cached.csv]` (no credentials
    # needed if a path is given; else fetches via the API).
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path:
        df = parse_jobs_table(Path(path).read_text(encoding="utf-8"))
    else:
        df = fetch_jobs_kreis()
    pd.set_option("display.width", 160)
    print(df.shape)
    print(df.describe().round(0).T[["count", "min", "50%", "max"]])
    for ags, lab in [("03103", "Wolfsburg"), ("09162", "München St"),
                     ("09662", "Schweinfurt St"), ("11000", "Berlin")]:
        if ags in df.index:
            print(lab, df.loc[ags].round(0).to_dict())
