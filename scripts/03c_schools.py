"""Authoritative school typing from JedeSchule (CC0), with OSM filling the gaps.

JedeSchule has the authoritative Schulart for ~34k schools but coords for only 81 %.
OSM has the LOCATIONS (~36k, more than JedeSchule) but we can only TYPE ~half by name.
So per track (grund/gym/real/mittel): take EVERY JedeSchule school that delivers it
(authoritative, located), then add an OSM school of that track only where JedeSchule has
NO school of that track nearby. That means JedeSchule wins where it has coords, while OSM
fills individual schools JedeSchule misses — INCLUDING inside a Schulzentrum, where several
schools sit within metres: the dedup is per-track, so a JedeSchule Gymnasium never suppresses
an OSM Realschule (or vice-versa) at the same campus. The JedeSchule schools that ship
address-only (the no-coord Länder: NI 100 %, SH 100 %, SL 49 %, HE/TH ~9 %) are placed by
03d_geocode against the official Land cadastre (OpenAddresses) and merged in below (~95 % at
~31 m); the residual ~5 % fall to OSM name-typing.
Writes the per-school track membership 12 consumes.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from wohnen.config import INTERIM, RAW
from wohnen.h3util import utm32_transformer
from wohnen.io import write_parquet_if_changed
from wohnen.schools import TRACKS, js_buckets, osm_buckets

MATCH_M = 150.0  # a JedeSchule and an OSM school of the same track within this = same place
_T = utm32_transformer()


def _xy(lon, lat):
    x, y = _T.transform(np.asarray(lon), np.asarray(lat))
    return np.c_[x, y]


def main():
    # OSM schools, typed by name (the gap-fill source)
    pois = pd.read_parquet(INTERIM / "pois.parquet")
    osm = pois[pois["category"] == "school"][["lat", "lon", "subcategory"]].reset_index(drop=True)
    osm_bk = [osm_buckets(s) for s in osm["subcategory"]]

    # JedeSchule, authoritative type — keep coord-bearing, typeable rows
    js = pd.read_csv(RAW / "jedeschule.csv", low_memory=False)
    js["lat"] = pd.to_numeric(js["latitude"], errors="coerce")
    js["lon"] = pd.to_numeric(js["longitude"], errors="coerce")
    js["bk"] = [js_buckets(t) for t in js["school_type"]]
    n_raw = len(js)

    # Fill the coord-less schools (the no-coord Länder) from the official cadastre
    # placement (03d_geocode): ~95 % of the ~3.7k that JedeSchule ships address-only,
    # at ~31 m median accuracy. Recover BEFORE the coord filter below so they survive.
    geo_path = INTERIM / "schools_geocoded.parquet"
    n_geo = 0
    if geo_path.exists():
        geo = pd.read_parquet(geo_path).set_index("id")
        miss = js["lat"].isna() & js["id"].isin(geo.index)
        js.loc[miss, "lat"] = js.loc[miss, "id"].map(geo["lat"])
        js.loc[miss, "lon"] = js.loc[miss, "id"].map(geo["lon"])
        n_geo = int(miss.sum())

    # JedeSchule often splits a school across two rows (esp. the no-coord states SH/ST/SL,
    # where Schulart and geo-coords come from different sources): one row carries the type
    # but no coords, the other coords but no type. Re-join by (name, zip) so a coord-bearing
    # row with no usable type borrows its typed twin's Schulart (e.g. Marion-Dönhoff-
    # Gymnasium Mölln) — otherwise the type-less + coord-less halves both get dropped below.
    def nkey(name, zp):
        n = re.sub(r"\s+", " ", name.strip().lower()) if isinstance(name, str) else ""
        z = re.sub(r"\D", "", str(zp))[:5]
        return (n, z) if n else None

    typed = {}
    for name, zp, bk in zip(js["name"], js["zip"], js["bk"]):
        k = nkey(name, zp)
        if k and bk:
            typed[k] = typed.get(k, frozenset()) | bk
    recovered = 0
    bks = list(js["bk"])
    for i, (name, zp, bk, lat) in enumerate(zip(js["name"], js["zip"], js["bk"], js["lat"])):
        if not bk and pd.notna(lat):
            t = typed.get(nkey(name, zp))
            if t:
                bks[i] = t
                recovered += 1
    js["bk"] = bks

    js = js[js["lat"].notna() & js["lon"].notna() & (js["bk"].map(len) > 0)].reset_index(drop=True)

    # JedeSchule duplicates some schools across 2–3 rows with different scrambled `id`s
    # (same name + address + zip; worst in SL 16 % / HH 8 % / BW 5 %). Dedup so one school
    # = one point. Key on (name, zip), NOT coords: a Schulzentrum geocodes several DISTINCT
    # schools to one point, which coord-dedup would wrongly merge. Unnamed rows kept as-is.
    js["_k"] = [nkey(n, z) for n, z in zip(js["name"], js["zip"])]
    keyed = js["_k"].notna()
    n_dup = int(keyed.sum() - js.loc[keyed, "_k"].nunique())
    js = (pd.concat([js[keyed].drop_duplicates("_k"), js[~keyed]])
          .drop(columns="_k").reset_index(drop=True))

    js_xy = _xy(js["lon"].values, js["lat"].values)
    osm_xy = _xy(osm["lon"].values, osm["lat"].values)

    # per track, a KDTree of the JedeSchule schools delivering it
    trees = {}
    for t in TRACKS:
        m = np.array([t in b for b in js["bk"]], dtype=bool)
        trees[t] = cKDTree(js_xy[m]) if m.any() else None

    # an OSM school contributes a track only where JedeSchule has none of that track nearby
    osm_keep = [set() for _ in range(len(osm))]
    for t in TRACKS:
        idx = [i for i in range(len(osm)) if t in osm_bk[i]]
        if not idx:
            continue
        tr = trees[t]
        if tr is None:                       # JedeSchule has no school of this track at all
            for i in idx:
                osm_keep[i].add(t)
            continue
        d, _ = tr.query(osm_xy[idx], distance_upper_bound=MATCH_M)
        d = np.atleast_1d(d)
        for k, i in enumerate(idx):
            if not np.isfinite(d[k]):        # no JedeSchule of track t within R → OSM fills it
                osm_keep[i].add(t)

    rows = {"lat": [], "lon": [], **{t: [] for t in TRACKS}}

    def emit(lat, lon, tracks):
        rows["lat"].append(lat)
        rows["lon"].append(lon)
        for t in TRACKS:
            rows[t].append(t in tracks)

    for i in range(len(js)):                 # all JedeSchule points (authoritative)
        emit(js["lat"].iat[i], js["lon"].iat[i], js["bk"].iat[i])
    osm_kept = 0
    for i in range(len(osm)):                # OSM gap-fill (only its uncovered tracks)
        if osm_keep[i]:
            emit(osm["lat"].iat[i], osm["lon"].iat[i], osm_keep[i])
            osm_kept += 1

    out = pd.DataFrame(rows)
    changed = write_parquet_if_changed(out, INTERIM / "schools_points.parquet",
                                       sort_cols=["lat", "lon"])

    print(f"JedeSchule: {n_raw} rows → {len(js)} typeable located points "
          f"({n_geo} coord-less placed via cadastre, {recovered} re-typed from a split twin, "
          f"{n_dup} duplicate rows dropped)")
    print(f"OSM gap-fill schools kept (a track JedeSchule lacks nearby): {osm_kept}")
    print(f"schools_points.parquet: {len(out)} points "
          f"({'written' if changed else 'unchanged'})")
    print(out[list(TRACKS)].sum().to_string())


if __name__ == "__main__":
    main()
