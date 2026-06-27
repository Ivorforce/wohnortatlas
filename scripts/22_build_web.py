"""Emit web/data.bin: a gzip-compressed binary payload + folded manifests.

Wire format (parsed by web/decode.js — keep the two in sync). The file on disk
is gzip(container); decode.js fetches it and inflates with DecompressionStream.
The container is:
  "WHN1" | uint32 LE header length | header JSON (utf-8) | column blocks,
  one per header "cols" entry, little-endian, each 4-byte aligned.
  header: {"n": <cells>, "labels": [<string table>],
           "cols": [{"name", "kind", "dtype", <encoding>}, ...],
           "layers": [...], "targets": [...], "centers": [...]}
  kinds: h3hi/h3lo (h3 id as two uint32), label (string-table index),
  inhabited (0/1), score (0-100), raw (nullable, all u8). A raw column's
  <encoding> (see encode_raw) is one of: {"dec"} -> value = code / 10^dec,
  {"lo","q"} -> lo + code*q, or {"log","lo","q"} -> expm1(lo + code*q).
  Codes clamp to [0,254]; 255 is the NaN sentinel. (Percentile-rank "pct"
  columns were dropped — the client computes per-target CDF ranks itself.)
Raw binary fetched over http(s) (the map is served, not opened from file://),
so no base64 wrapper — the layer/target/center manifests that used to ride in
a base64 JS blob are folded into the header instead, leaving one fetch + one gzip.
"""

import gzip
import json
import struct
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import geopandas as gpd
import numpy as np
import pandas as pd

from wohnen.config import LAYERS, WEB, DATA, BBOX, JOBS_BUCKET_LABELS

# Score layers shipped to the web, in display order. This list is PURELY data:
# which scores.parquet columns are factor layers and in what order they appear.
# All presentation — German label, description, default weight — lives in the
# frontend (SCORE_META in index.html), alongside the 6 client-side layers'
# metadata, so all 14 factors are described in one place and the weights aren't
# duplicated across the Python/JS boundary. The manifest ships only the keys.
#
# Not in this list (client-side layers, defined entirely in index.html):
#   s_rent / s_anbindung — recomputed client-side from raw columns + user controls;
#   s_access / s_freizeit — computed client-side from raw t_*/reach_* columns;
#   s_density / s_age     — preference pseudo-layers over raw pop_dens / avg_age.
# The old bundled pipeline scores (s_rent / s_daily / s_famedu / s_leisure) were
# dropped 2026-06 — the web recomputes Miete / In der Nähe / Freizeit client-side
# from the raw columns, so they are neither written nor shipped.
# Panel order (client-side s_rent/s_anbindung/s_access/s_freizeit/s_density/s_age
# splice in RELATIVE to these in index.html). Narrative: the place & its green
# run Ortsbild → Grünes Viertel → Natur erreichbar (green bridges built-character
# and reachable-nature), then ambient/risk (Ruhe, Hochwasser, Klima), then infra.
SCORE_LAYERS = [
    "s_oepnv", "s_character", "s_green", "s_nature",
    "s_quiet", "s_flood", "s_vacancy", "s_climate", "s_broadband",
]
RAW_COLS = {
    # The 04f commute aggregates any_*/gross_* are intentionally absent here: emit_branche
    # packs them into on-demand chunks (web/reach/branche-{any,gross}.bin) like the job
    # sectors, so the client loads every field target through one path. They live in
    # scores.parquet for that emitter; this list governs only the eager data.bin columns.
    "deps_per_day": 0, "n_lines_w": 0,
    "rent_cal": 1, "kreis_pay": 0, "avg_age": 0,  # avg_age: integer years (1-yr steps)
    # age-band shares (fraction of residents) — the life-stage signal s_age scores on
    # alongside avg_age. u18 separates families (kids) from students/WGs (both young by
    # mean), 65+ pins senior enclaves. dec=3 → 0.1% precision.
    "share_u18": 2, "share_65plus": 2,  # 1pp steps — soft life-stage signal, plenty
    "tree_share": 2, "noise_penalty": 2,
    # noise sub-sources (noise = max of these) — power the compare accordion
    "road_penalty": 2, "rail_penalty": 2, "urban_penalty": 2, "airport_penalty": 2,
    "water_share": 2, "grass_share": 2,  # s_green sub-drivers beyond trees
    "park_share": 2,  # tree_canopy feeds the score but isn't shipped (not a
    "waterway_share": 2,  # breakdown row — duplicates Baumanteil); river/Bach ambience
    "flood_depth_hq100": 1,
    "vacancy_pct": 0,  # s_vacancy tooltip: Wohnungsleerstand % (whole %, fits u8)
    # relief_m feeds s_nature inside 13 (source amplifier) but isn't shipped — it's
    # not a home-cell driver (see RAW_LABELS note), so not a breakdown row.
    "t_lake_min": 0, "t_forest_min": 0, "t_river_min": 0,
    "sight_score": 2,  # s_nature sub-driver (viewpoints/peaks/waterfalls density)
    "t_grundschule_min": 0, "t_gymnasium_min": 0,
    "t_realschule_min": 0, "t_mittelschule_min": 0,  # Sek-I nearest-times (web "Für Jugend")
    "t_kita_min": 0,  # KiTa nearest-time (s_access KiTa need)
    "t_doctor_min": 0, "t_dentist_min": 0,  # Nahversorgung Arzt + Zahnarzt
    "t_pharmacy_min": 0,  # Nahversorgung Apotheke
    # Nahversorgung food, two tiers: Vollsortiment (full shop) vs Frische (capped gap-fill)
    "t_vollsort_min": 0, "t_frische_min": 0,
    "kita_crowd": 2,  # internal: Kita-spot crowding (client multiplies kita score)
    # s_access "In der Nähe" (essentials) is computed client-side from the raw
    # t_*_min above. s_freizeit "Freizeit & Kultur" reads these precomputed per-mode
    # gravity reachability surfaces (all derived by 04e_freizeit), max-over-modes
    # client-side. dec=2 (they go through the geo-mean / weighted mean). reach_activity
    # = the always-on broad baseline; the rest are checkbox sources.
    "reach_activity_transit": 2, "reach_activity_bike": 2, "reach_activity_foot": 2, "reach_activity_car": 2,
    "reach_kultur_transit": 2, "reach_kultur_bike": 2, "reach_kultur_foot": 2, "reach_kultur_car": 2,
    "reach_swim_transit": 2, "reach_swim_bike": 2, "reach_swim_foot": 2, "reach_swim_car": 2,
    "reach_kino_transit": 2, "reach_kino_bike": 2, "reach_kino_foot": 2, "reach_kino_car": 2,
    "reach_klettern_transit": 2, "reach_klettern_bike": 2, "reach_klettern_foot": 2, "reach_klettern_car": 2,
    "reach_golf_transit": 2, "reach_golf_bike": 2, "reach_golf_foot": 2, "reach_golf_car": 2,
    # s_character/Ortsbild drivers (historic=* + public art + street grain)
    "hist_density": 0, "art_density": 0, "street_grain": 2,  # int density (display only)
    "ftth_share": 2,
    "rain_mm": 0, "sun_h": 0, "snow_days": 0,
    "population": 0,
    "pop_inhabited_dens": 0,  # Einw./km² of inhabited land — drives s_density
}


DTYPES = {"u8": np.uint8, "u16": np.uint16, "u32": np.uint32}


# Raw columns whose range can't fit a u8 at any integer `dec` and so use a
# per-column affine encoding instead. LINEAR: v = lo + code*q (income/weather,
# narrow-ish but offset far from 0). LOG: v = expm1(lo + code*q) — population &
# built-up density span >2 decades and are perceptually multiplicative (density
# bands are ratios), so a log grid spends the 254 levels where they matter.
LINEAR_COLS = {"kreis_pay", "rain_mm", "sun_h", "snow_days"}
LOG_COLS = {"population", "pop_inhabited_dens"}


def encode_raw(name, vals, dec):
    """Pack a raw column into u8 (255 = NaN sentinel). Encoding chosen per column;
    the header carries enough to invert it in decode.js:
      LOG    -> {"log": True, "lo", "q"}  v = expm1(lo + code*q)
      LINEAR -> {"lo", "q"}               v = lo + code*q
      else   -> {"dec"}                   v = code / 10^dec
    Codes clamp to [0, 254] — 254 means "≥ this", which for minute/density columns
    is a cell already past every reachable-distance threshold (identical score) and
    for the rest is the true max. 255 is reserved for NaN. Per-element round(x, dec)
    matches CPython's correct rounding (np.round on v*10^dec inherits the multiply's
    .5-boundary representation error)."""
    v = np.asarray(vals, dtype=float)
    ok = ~np.isnan(v)
    if ok.any() and (v[ok] < 0).any():
        raise ValueError(f"{name}: negative values cannot be packed unsigned")
    out = np.full(len(v), 255, np.uint8)
    if not ok.any():
        return {"dec": dec}, out
    if name in LOG_COLS:
        lo = float(np.log1p(v[ok].min()))
        q = (float(np.log1p(v[ok].max())) - lo) / 254 or 1.0
        code = (np.log1p(v[ok]) - lo) / q
        meta = {"log": True, "lo": round(lo, 6), "q": round(q, 8)}
    elif name in LINEAR_COLS:
        lo = float(v[ok].min())
        q = (float(v[ok].max()) - lo) / 254 or 1.0
        code = (v[ok] - lo) / q
        meta = {"lo": round(lo, 4), "q": round(q, 8)}
    else:
        code = np.array([round(float(x), dec) for x in v[ok]]) * 10**dec
        meta = {"dec": dec}
    out[ok] = np.clip(np.round(code), 0, 254).astype(np.uint8)
    return meta, out


# walk_min LAST: a chunk built before foot existed had 4 modes ending in car_hbf_min;
# keeping the first four in place + appending walk lets the client decode old and new
# chunks the same way (decodeTarget infers the count). Keep in sync with web/decode.js.
REACH_MODES = ["transit_hbf_min", "transit_bike_min", "bike_hbf_min", "car_hbf_min", "walk_min"]


def _colmap(npz, order):
    """Map each base-payload cell (in `order`) to its column in the reach npz; `fill` flags
    cells absent from the npz (written as the 255 no-reach sentinel)."""
    pos = {h: i for i, h in enumerate(npz["cell_ids"])}
    colmap = np.fromiter((pos.get(h, 0) for h in order), np.int64, len(order))
    fill = np.fromiter((h not in pos for h in order), bool, len(order))
    return colmap, fill


def _write_chunk(rdir, cid, mats, row, colmap, fill):
    """Encode one target's per-mode reach → web/reach/<cid>.bin (gzip u8 minutes, 255 =
    no-reach), reordered to the base payload's cell order. mats: {mode: (T, N) uint8} from the
    reach npz (already uint8 — no rounding); row = this target's matrix index."""
    buf = bytearray()
    for m in REACH_MODES:
        u = mats[m][row][colmap]
        if fill.any():
            u = u.copy()
            u[fill] = 255
        buf += u.tobytes()
    (rdir / f"{cid}.bin").write_bytes(gzip.compress(bytes(buf), 9))


def gemeinde_pop(df) -> dict:
    """Total population per municipality (VG250 GEN → Σ hex population), keyed the
    same way as a city's name. The map declutters candidate markers on zoom-out by
    this — the town's OWN size, so a real but isolated city (Landshut) outranks a
    Munich suburb (Unterhaching), unlike catchment_pop which counts the neighbours'
    people and floats suburbs up to Großstadt level."""
    vg = gpd.read_file(f"zip://{DATA / 'raw' / 'vg250.zip'}!"
                       + next(n for n in zipfile.ZipFile(DATA / "raw" / "vg250.zip").namelist()
                              if n.endswith("VG250_GEM.shp"))).to_crs(4326)
    vg = vg[vg["GF"] == 4][["GEN", "geometry"]].cx[BBOX[0]:BBOX[2], BBOX[1]:BBOX[3]].to_crs(25832)
    pts = gpd.GeoDataFrame(df[["population"]],
                           geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs=4326).to_crs(25832)
    j = gpd.sjoin(pts, vg, how="left", predicate="within")
    return j.groupby("GEN")["population"].sum().to_dict()


def emit_targets(df):
    """Commute-target chunks + manifests for the Lebensmittelpunkt picker.

    Whole-CITY targets (the catchment-weighted percentile over a Gemeinde's centers, 04f)
    ship as web/reach/<id>.bin — München included now (no base-col special-case). Multi-center
    cities ALSO ship per-PART chunks (one center each), so a job in a specific corner scores
    honestly instead of via the whole-city percentile. CENTERS carries the part id/name for
    snapping."""
    cpath = LAYERS / "cities.parquet"
    if not cpath.exists():
        print("NOTE: cities.parquet missing — no commute targets emitted")
        return [], []
    cities = pd.read_parquet(cpath).sort_values("catchment_pop", ascending=False)
    gpop = gemeinde_pop(df)
    rc = np.load(LAYERS / "reach_cities.npz", allow_pickle=False)
    order = df["h3"].values
    city_mats = {m: rc[m] for m in REACH_MODES}
    city_row = {t: i for i, t in enumerate(rc["target_ids"])}
    ccolmap, cfill = _colmap(rc, order)
    rdir = WEB / "reach"
    if rdir.exists():
        for f in rdir.glob("*.bin"):
            f.unlink()                       # drop stale chunks (centers change on rebuild)
    rdir.mkdir(exist_ok=True)

    # multi-center cities = those with selectable PARTS (≥2 reference points). Only they
    # need the "ganz …" prefix to disambiguate the whole-city percentile from a corner;
    # a single-center town is just its name.
    ppath = LAYERS / "parts.parquet"
    parts = pd.read_parquet(ppath) if ppath.exists() else pd.DataFrame(
        columns=["id", "name", "city", "lat", "lon", "center_h3"])
    multi_city_ids = set(parts["city"])

    manifest, n_chunks = [], 0
    for _, c in cities.iterrows():
        # "o" = the town's own Gemeinde population, the marker's size/prominence —
        # the web declutters candidate markers on zoom-out by hiding those below a
        # zoom-dependent o floor (catchment_pop floated suburbs up — see gemeinde_pop).
        entry = {"id": c["id"], "name": c["name"],
                 "lat": round(float(c["lat"]), 4), "lon": round(float(c["lon"]), 4),
                 "o": int(round(float(gpop.get(c["name"], c["catchment_pop"]))))}
        if c["id"] in multi_city_ids:
            entry["multi"] = 1
        # every city (incl. München) ships a real per-city reach chunk = the catchment-
        # weighted percentile over its centers (04f). München no longer reuses the base
        # central-Hbf columns as its target — picking "München" now means "reach most of
        # München", not "reach the Hbf" (the base cols stay in the payload for the cell
        # inspector's raw time-to-centre readout).
        if c["id"] not in city_row:
            continue
        _write_chunk(rdir, c["id"], city_mats, city_row[c["id"]], ccolmap, cfill)
        manifest.append(entry)
        n_chunks += 1

    # per-PART chunks (multi-center cities only — see 04c). Each selectable corner of a
    # big city; surfaced through search, not the city chips/dropdown.
    if ppath.exists() and len(parts):
        rp = np.load(LAYERS / "reach_parts.npz", allow_pickle=False)
        part_mats = {m: rp[m] for m in REACH_MODES}
        part_row = {t: i for i, t in enumerate(rp["target_ids"])}
        pcolmap, pfill = _colmap(rp, order)
        for pid in parts["id"]:
            if pid in part_row:
                _write_chunk(rdir, pid, part_mats, part_row[pid], pcolmap, pfill)
    print(f"web/reach/: {n_chunks} city + {len(parts)} part chunks "
          f"({len(REACH_MODES)} modes × {len(order)} u8)")

    # per-CENTER coords for client-side nearest-center snapping (search). [lat4, lon4,
    # cityIdx] — and, when the center is a selectable PART of a multi-center city,
    # + [partId, partName] so search can offer that specific corner. cityIdx indexes
    # TARGETS (by Gemeinde name; every center's city is in the manifest).
    name_to_idx = {e["name"]: i for i, e in enumerate(manifest)}
    part_by_h3 = {r["center_h3"]: (r["id"], r["name"]) for _, r in parts.iterrows()}
    centers = pd.read_parquet(LAYERS / "centers.parquet")
    centers_out = []
    for _, r in centers.iterrows():
        if r["name"] not in name_to_idx:
            continue
        e = [round(float(r["lat"]), 4), round(float(r["lon"]), 4), name_to_idx[r["name"]]]
        p = part_by_h3.get(r["id"])
        if p:
            e += [p[0], p[1]]
        centers_out.append(e)
    print(f"CENTERS: {len(centers_out)} snap points, {len(part_by_h3)} with a part")

    # per-tier OUTLINE sets for the active aggregate target: the centers whose cityness
    # weight FEEDS the aggregate score — o_any / o_gross straight from centers.parquet (04b),
    # the SAME weights 04f scores from, so the green outline can never drift from the score.
    # ABSOLUTE and per-center (o > 0 ⟺ catchment over the tier band): a low-density corner of
    # a big Gemeinde (o_gross 0) is NOT drawn, but a real Großstadt shadowed by a bigger
    # neighbour still is (no argmax/percentile — those would be relative). Strongest-first.
    from wohnen.cityness import cityness_o          # recompute o from catchment (matches 04f)
    outline = {}
    for tier in ("any", "gross"):
        o = cityness_o(centers["catchment_pop"].to_numpy(float), tier)
        sel = centers.iloc[np.argsort(-o)[: int((o > 0).sum())]]   # o>0 centres, strongest-first
        outline[tier] = [[round(float(la), 4), round(float(lo), 4)]
                         for la, lo in zip(sel["lat"], sel["lon"])]
    print(f"OUTLINE: any={len(outline['any'])} gross={len(outline['gross'])} centers")
    return manifest, centers_out, outline


BRANCHE_O_MODES = ["transit", "bike", "walk", "car"]   # the 4 web modes (= 04h/04f O_MODES)
AGG_FIELDS = [("any", "Alle Branchen (Stadt)"), ("gross", "Alle Branchen (Großstadt)")]
# EVERY field target (the cityness baselines AND the job sectors) ships the SAME wire format:
# per mode an M/B/O triple (12 u8 arrays, per-mode grouped m,b,o), gate "mbo". The client scores
# the budget interp B + (T−M)/(2h−M)·(O−B), best-wins over modes (recomputeAnbindungMBO).
MBO_KEYS = [f"{p}_{mode}" for mode in BRANCHE_O_MODES for p in ("m", "b", "o")]


def _mbo_chunk(arrays, colmap, fill):
    """Pack one mbo field chunk: 12 u8 arrays (per mode M, B, O) reordered to base cell order,
    255 where the base cell is absent from the routed grid."""
    buf = bytearray()
    for u in arrays:
        u = u[colmap]
        if fill.any():
            u = u.copy(); u[fill] = 255
        buf += np.asarray(u, dtype=np.uint8).tobytes()
    return bytes(buf)


def emit_branche(df):
    """Field (Branche) Anbindung targets — lazy M/B/O chunks + one manifest:
      • "Alle Branchen" any/gross — cityness M/B/O from reach_cityness.npz (04f).
      • the 9 job sectors — sector-jobs M/B/O from reach_branche.npz (04h).
    Same 12-u8 wire format + client path for both; only the per-centre opportunity differs."""
    rdir = WEB / "reach"
    rdir.mkdir(exist_ok=True)               # emit_targets already cleared stale *.bin
    order = df["h3"].values
    manifest = []

    cpath = LAYERS / "reach_cityness.npz"
    if cpath.exists():
        rc = np.load(cpath, allow_pickle=False)
        colmap, fill = _colmap(rc, order)
        for tier, label in AGG_FIELDS:
            arrays = [rc[f"{tier}_{k}"] for k in MBO_KEYS]
            (rdir / f"branche-{tier}.bin").write_bytes(gzip.compress(_mbo_chunk(arrays, colmap, fill), 9))
            manifest.append({"id": tier, "label": label, "gate": "mbo"})
    else:
        print("NOTE: reach_cityness.npz missing — no Alle-Branchen baselines emitted")

    bpath = LAYERS / "reach_branche.npz"
    if bpath.exists():
        rb = np.load(bpath, allow_pickle=False)
        colmap, fill = _colmap(rb, order)
        for row, key in enumerate(rb["bucket_ids"]):
            arrays = [rb[k][row] for k in MBO_KEYS]
            (rdir / f"branche-{key}.bin").write_bytes(gzip.compress(_mbo_chunk(arrays, colmap, fill), 9))
            manifest.append({"id": f"branche:{key}", "label": JOBS_BUCKET_LABELS[key], "gate": "mbo"})
    else:
        print("NOTE: reach_branche.npz missing — no Branche sector targets emitted")

    print(f"web/reach/branche-*.bin: {len(manifest)} field targets (mbo {len(MBO_KEYS)} u8 × {len(df)})")
    return manifest


def main():
    df = pd.read_parquet(LAYERS / "scores.parquet")
    n = len(df)

    layers = [{"key": k} for k in SCORE_LAYERS if k in df.columns]

    cols = []  # ({header entry}, ndarray) in wire order

    ids = np.array([int(h, 16) for h in df["h3"]], dtype=np.uint64)
    cols.append(({"name": "h3hi", "kind": "h3hi", "dtype": "u32"},
                 (ids >> 32).astype(np.uint32)))
    cols.append(({"name": "h3lo", "kind": "h3lo", "dtype": "u32"},
                 (ids & 0xFFFFFFFF).astype(np.uint32)))

    from wohnen.labels import labels_for_points
    labels = labels_for_points(df["lat"].values, df["lon"].values)
    codes, table = pd.factorize(labels)
    dt = "u16" if len(table) <= 0xFFFF else "u32"
    cols.append(({"name": "label", "kind": "label", "dtype": dt},
                 codes.astype(DTYPES[dt])))

    inhabited = df.get("inhabited", pd.Series(True, index=df.index)).fillna(False)
    cols.append(({"name": "inhabited", "kind": "inhabited", "dtype": "u8"},
                 inhabited.values.astype(np.uint8)))

    for l in layers:
        cols.append(({"name": l["key"], "kind": "score", "dtype": "u8"},
                     np.round(df[l["key"]].fillna(0).values * 100)
                     .astype(np.uint8)))

    raw_cols = {c: d for c, d in RAW_COLS.items() if c in df.columns}
    for c, dec in raw_cols.items():
        meta, arr = encode_raw(c, df[c].values, dec)
        cols.append(({"name": c, "kind": "raw", "dtype": "u8", **meta}, arr))

    # No "pct" columns: the population-weighted percentile ranks they carried were
    # dead weight (~35% of data.bin) — the client computes its own per-target CDF
    # ranks (arrCDF/cdfPct in index.html), which the old whole-region/München-base
    # pct couldn't do anyway. Dropped; index.html never read CELLS.pct.

    # the layer/target/center manifests fold into the header (they used to be separate
    # globals in a base64 JS blob); emit_targets also writes the per-target reach chunks.
    targets, centers, outline = emit_targets(df)
    branches = emit_branche(df)            # after emit_targets (which clears stale chunks)

    head = json.dumps(
        {"n": n, "labels": [str(s) for s in table],
         "cols": [m for m, _ in cols],
         "layers": layers, "targets": targets, "centers": centers, "outline": outline,
         "branches": branches},
        ensure_ascii=False, separators=(",", ":")).encode()
    blob = bytearray(b"WHN1") + struct.pack("<I", len(head)) + head
    for _, arr in cols:
        blob += b"\0" * (-len(blob) % 4)
        blob += arr.astype(arr.dtype.newbyteorder("<")).tobytes()

    WEB.mkdir(exist_ok=True)
    gz = gzip.compress(bytes(blob), 9)
    (WEB / "data.bin").write_bytes(gz)
    print(f"web/data.bin: {len(gz)/1e6:.1f} MB gz (binary {len(blob)/1e6:.1f} MB), "
          f"{n} cells, {len(layers)} layers, {len(targets)} targets, "
          f"{len(table)} label strings")


if __name__ == "__main__":
    main()
