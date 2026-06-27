"""Place the coord-less JedeSchule schools against the official cadastre (OpenAddresses).

JedeSchule has a Schulart + street address but NO coords for ~3.7k schools, all in the
no-coord Länder (NI 100 %, SH 100 %, SL 49 %, HE/TH ~9 %). 03c therefore dropped them and
those areas fell back to OSM name-typing. Here we recover their coordinates by matching the
JedeSchule address to the official Land cadastre address points (Hauskoordinaten, shipped
uniformly by OpenAddresses) — measured ~95 % placed at ~31 m median accuracy, vs 43 % at
150 m–1 km for an OSM-address geocode. Writes (id, lat, lon, match) for the placed schools;
03c merges these in to fill the missing coords before its per-track placement.

The index is built FILTERED to the target schools' (street, hnr) keys while streaming the
gz files, so memory stays bounded by the few thousand schools, not the millions of addresses
(the states we DON'T need are never downloaded — see config.OA_STATES). Logic: wohnen/geocode.
"""

import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from wohnen.config import INTERIM, RAW
from wohnen.geocode import GOOD, geocode, house_num, norm_city, norm_street, split_address
from wohnen.io import write_parquet_if_changed
from wohnen.schools import js_buckets


def main():
    js = pd.read_csv(RAW / "jedeschule.csv", low_memory=False)
    has_coord = (js["latitude"].notna() & js["longitude"].notna()
                 & (js["latitude"].astype(str).str.strip() != ""))
    typeable = js["school_type"].map(lambda s: len(js_buckets(s)) > 0)
    street_hnr = js["address"].map(split_address)
    tgt = js[~has_coord & typeable].copy()
    tgt["street"] = [street_hnr[i][0] for i in tgt.index]
    tgt["hnr"] = [street_hnr[i][1] for i in tgt.index]
    tgt["zip5"] = tgt["zip"].astype(str).str.extract(r"(\d{5})")[0]

    # target (norm_street, house_num) keys — the index is filtered to these
    want = set()
    for s, h in zip(tgt["street"], tgt["hnr"]):
        ns, hn = norm_street(s), house_num(h)
        if ns and hn is not None:
            want.add((ns, hn))
    print(f"coord-less typeable schools: {len(tgt)} ({len(want)} distinct street/no. keys)")

    # stream the official cadastre files, keep only target-matching address points
    idx = defaultdict(list)
    files = sorted((RAW / "openaddresses").glob("de_*.geojson.gz"))
    for fp in files:
        n = 0
        with gzip.open(fp, "rt", encoding="utf-8") as f:
            for line in f:
                try:
                    o = json.loads(line)
                except ValueError:
                    continue
                p = o.get("properties") or {}
                hn = house_num(p.get("number"))
                ns = norm_street(p.get("street"))
                if hn is None or not ns or (ns, hn) not in want:
                    continue
                g = o.get("geometry") or {}
                c = g.get("coordinates")
                if not c:
                    continue
                idx[(ns, hn)].append((c[1], c[0], (p.get("postcode") or "")[:5],
                                      norm_city(p.get("city"))))
                n += 1
        print(f"  {fp.name}: {n} matching address points")

    out = {"id": [], "lat": [], "lon": [], "match": []}
    counts = defaultdict(int)
    for r in tgt.itertuples():
        lat, lon, mt = geocode(idx, r.zip5, r.city, r.street, r.hnr)
        counts[mt] += 1
        if mt in GOOD:
            out["id"].append(r.id)
            out["lat"].append(lat)
            out["lon"].append(lon)
            out["match"].append(mt)
    placed = len(out["id"])
    changed = write_parquet_if_changed(pd.DataFrame(out),
                                       INTERIM / "schools_geocoded.parquet",
                                       sort_cols=["id"])
    print(f"placed {placed}/{len(tgt)} ({100 * placed / len(tgt):.1f}%) — "
          f"{dict(sorted(counts.items(), key=lambda x: -x[1]))}")
    print(f"schools_geocoded.parquet: {placed} rows ({'written' if changed else 'unchanged'})")


if __name__ == "__main__":
    main()
