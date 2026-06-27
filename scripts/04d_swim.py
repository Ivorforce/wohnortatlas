"""Reverse-route s_freizeit point SOURCES (swim/Kino/Klettern/Golf) to all
cells, per mode, and persist two arrays per source×mode to reach_spots.npz:

    reach_spots.npz["<src>_<mode>"][cell]      = Σ_spot mass·point_decay(t)  (gravity sum)
    reach_spots.npz["<src>_<mode>_tmin"][cell] = nearest reachable spot time (min; inf=none)

04e_freizeit combines these into the [0,1] s_freizeit surfaces (the "need just one" model
in wohnen.freizeit: nearest sets a distance ceiling, variety fills it). Splitting route
(here, expensive) from derive (04e, cheap) keeps POINT_B/POINT_K + the ceiling tunable
without re-routing; changing the spot SET / decay SHAPE / spot mass means re-running THIS.

WEIGHT-only (no veto): "can I get to a pool/cinema" is a weekend question, a ~1 h
trip is fine. Reverse-origins symmetry (evening return leg ≈ round-trip). Spots are
snapped to their grid cell and deduped so the origin count stays small. Add a lumpy
activity by extending SOURCES (each routes independently → one surface).

Reuses the wohnen.r5_routing scaffold via wohnen.reach.load_r5.

Usage: 04d_swim.py [--sample N]   (route to a cell sample; no write)
"""

import concurrent.futures
import datetime as dt
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# capture our flag BEFORE importing 04 — its module body rewrites sys.argv.
_SAMPLE = int(sys.argv[sys.argv.index("--sample") + 1]) if "--sample" in sys.argv else None
# REACH_SPOT_LIMIT=N: route only N (spread) spots/source to the FULL grid, print per-mode
# sanity, DON'T write — a fast national value/rate check before the multi-hour real run.
_LIMIT = int(os.environ["REACH_SPOT_LIMIT"]) if os.environ.get("REACH_SPOT_LIMIT") else None

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree

from wohnen.config import LAYERS, INTERIM, CAR_CONGESTION, CAR_PARK_MIN
from wohnen.freizeit import SOURCES, point_decay
from wohnen.reach import (
    door_percentile, gather_window, honest_stop_times, load_egress, morton_order,
    setup_decomp_net)

# Per-spot pull = mass·point_decay(t) (flat to ~5 min, generalized-Gaussian tail; the
# shared "need just one" model in wohnen.freizeit). Both the gravity SUM and the per-cell
# NEAREST-spot time are persisted — 04e combines them into the [0,1] surface (the ceiling
# re-tunes route-free from the nearest time; the decay SHAPE is baked here → re-route).
HORIZON = 60.0   # leisure reach horizon (min): transit-decompose cap + gravity cutoff
_WORKERS = int(os.environ.get("REACH_TRANSIT_WORKERS", str(max(4, (os.cpu_count() or 8) - 2))))
# Spots are Morton-sorted into local batches. DIRECT modes route each batch only to
# cells within that mode's FULL 60-min reach (MODE_WINDOW_KM) — walk/bike reach ~5/15 km
# in 60 min, so a single 140-km-for-everything window made walk's window ~28× its real
# reach (the cost was the per-origin extraction over those cells). Each window covers
# the WHOLE 60-min reach, so no reachable cell is clipped — lumpy hobby amenities mean a
# place's only pool by some mode may be a far one, and dropping it would wrongly
# disqualify that place. TRANSIT is NOT distance-windowed (its 60-min reach is unbounded
# — ICE ~250 km); it decomposes to ALL cells. Gravity accumulates into the full grid.
MODE_WINDOW_KM = {"foot": 8.0, "bike": 25.0, "car": 150.0}   # ≥ each mode's 60-min reach
SPOT_BATCH = int(os.environ.get("REACH_SPOT_BATCH", "64"))
_T25832 = Transformer.from_crs(4326, 25832, always_xy=True)  # metres, good over DE

r5py = None  # set in main from the decomposition context (matrix60 reads it)

# SOURCES (category, subcategory, mass per leisure source) lives in wohnen/freizeit.py —
# shared with 03b_freizeit_spots, which pre-extracts exactly these rows from pois.parquet
# into freizeit_spots.parquet (the minimal POI input this script routes from).

# NB: a teen's school bus ("Für Jugend" ÖPNV) is NOT routed here — public GTFS
# lacks dedicated Schülerbeförderung, so the web models it as a coach-speed proxy
# from the straight-line school distance instead (see KID_OEPNV_* in index.html).


def matrix60(network, origins, dests, departure, modes) -> pd.DataFrame:
    """Like 04's matrix() but capped at the ~1 h leisure horizon. Returns the LONG
    (from_id, to_id, travel_time) frame — gravity_accumulate consumes it directly, so
    no dense [origins × dests] pivot/reindex is materialised (that was the bottleneck)."""
    ttm = r5py.TravelTimeMatrix(
        network, origins=origins, destinations=dests, departure=departure,
        departure_time_window=dt.timedelta(hours=1), percentiles=[50],
        max_time=dt.timedelta(minutes=HORIZON), transport_modes=modes)
    return pd.DataFrame(ttm)[["from_id", "to_id", "travel_time"]]


def load_spots(pois: pd.DataFrame, spec) -> tuple[list, np.ndarray, list, list]:
    """Snap each spot to its res-8 cell and sum mass per cell (dedupe origins)."""
    agg: dict[str, float] = {}
    for cat, sub, mass in spec:
        sel = pois[(pois["category"] == cat) & (pois["subcategory"] == sub)]
        for la, lo in zip(sel["lat"].values, sel["lon"].values):
            c = h3.latlng_to_cell(float(la), float(lo), 8)
            agg[c] = agg.get(c, 0.0) + mass
    cells = list(agg)
    latlng = [h3.cell_to_latlng(c) for c in cells]
    return cells, np.array([agg[c] for c in cells]), [p[0] for p in latlng], [p[1] for p in latlng]


def gravity_accumulate(frm, to, tt, mass_b, cell_index, into, into_tmin):
    """Scatter-add Σ mass·point_decay(t) into into[] AND min-accumulate the nearest spot
    time into into_tmin[] (reachable, t < HORIZON) straight from the long-format routing
    result — no dense pivot/reindex. frm/to are spot-cell / dest-cell ids; mass_b maps spot
    cell → mass; cell_index maps dest cell → grid row."""
    keep = np.isfinite(tt) & (tt < HORIZON)
    if not keep.any():
        return
    idx = cell_index.get_indexer(to[keep])
    g = mass_b.reindex(frm[keep]).to_numpy() * point_decay(tt[keep])
    np.add.at(into, idx, g)
    np.minimum.at(into_tmin, idx, tt[keep])


def route_source(ctx, g_walk_all, cell_index, cells_gdf, cell_ids, tree,
                 s_cells, mass, s_lat, s_lon, departure, evening, modes) -> dict:
    """Route ONE source's spots → {mode: gravity[n_cells]}. Morton-sorted into local
    SPOT_BATCHes. DIRECT modes (walk/bike/car) route to a per-mode window = their full
    60-min reach (MODE_WINDOW_KM); TRANSIT decomposes to ALL cells (unbounded reach).
    Gravity scatter-accumulates into the full grid → exact (no reachable cell clipped)."""
    B, W, C = modes
    s_cells = np.asarray(s_cells); mass = np.asarray(mass)
    s_lat = np.asarray(s_lat); s_lon = np.asarray(s_lon)
    sx, sy = _T25832.transform(s_lon, s_lat)
    order = morton_order(sx, sy)
    s_cells, mass, s_lat, s_lon, sx, sy = (a[order] for a in (s_cells, mass, s_lat, s_lon, sx, sy))
    n = len(cell_ids)
    grav = {m: np.zeros(n) for m in ("transit", "bike", "foot", "car")}
    tmin = {m: np.full(n, np.inf) for m in ("transit", "bike", "foot", "car")}  # nearest spot time
    direct = [("foot", [W], evening), ("bike", [B], evening), ("car", [C], departure)]
    nb = (len(s_cells) + SPOT_BATCH - 1) // SPOT_BATCH
    for bi, b0 in enumerate(range(0, len(s_cells), SPOT_BATCH)):
        sl = slice(b0, b0 + SPOT_BATCH)
        b_cells, b_mass = s_cells[sl], mass[sl]
        b_lat, b_lon, b_sx, b_sy = s_lat[sl], s_lon[sl], sx[sl], sy[sl]
        bspots = gpd.GeoDataFrame({"id": b_cells},
            geometry=gpd.points_from_xy(b_lon, b_lat), crs="EPSG:4326")
        mass_b = pd.Series(b_mass, index=b_cells)

        # TRANSIT: full grid, per-spot SPARSE gravity (no dense matrix, no distance clip)
        def one(i):
            t2s = honest_stop_times(ctx, b_lat[i], b_lon[i], HORIZON, evening)
            if t2s is None:
                return None
            pidx = int(np.ceil(0.5 * t2s.shape[0]) - 1)       # R5 findPercentileIndex(nIter, 50)
            door = door_percentile(t2s, g_walk_all, n, pidx, HORIZON)
            m = np.isfinite(door)
            return (np.nonzero(m)[0], b_mass[i] * point_decay(door[m]), door[m]) if m.any() else None
        with concurrent.futures.ThreadPoolExecutor(max_workers=_WORKERS) as ex:
            for r in ex.map(one, range(len(b_cells))):
                if r is not None:
                    np.add.at(grav["transit"], r[0], r[1])
                    np.minimum.at(tmin["transit"], r[0], r[2])

        # DIRECT modes: per-mode 60-min reach window, gravity straight from the long frame
        for key, mode, dep_t in direct:
            hit = tree.query_ball_point(np.c_[b_sx, b_sy], r=MODE_WINDOW_KM[key] * 1000.0)
            win = np.unique(np.concatenate([np.asarray(h, int) for h in hit] + [np.empty(0, int)]))
            if not len(win):
                continue
            ttm = matrix60(ctx.net, bspots, cells_gdf.iloc[win], dep_t, mode)
            tt = ttm["travel_time"].to_numpy(dtype=float)
            if key == "car":
                tt = tt * CAR_CONGESTION + CAR_PARK_MIN
            gravity_accumulate(ttm["from_id"].to_numpy(), ttm["to_id"].to_numpy(),
                               tt, mass_b, cell_index, grav[key], tmin[key])
        print(f"    batch {bi + 1}/{nb} ({len(b_cells)} spots)", flush=True)
    return grav, tmin


RAW_NPZ = "reach_spots.npz"  # per-source×mode gravity sum + nearest-spot time (the routed result)


def main():
    departure = dt.datetime.fromisoformat(
        json.loads((INTERIM / "departure.json").read_text())["departure"])
    evening = departure.replace(hour=17, minute=0, second=0, microsecond=0)

    grid = pd.read_parquet(INTERIM / "grid.parquet")
    if _SAMPLE:
        grid = grid.sample(min(2000, len(grid)), random_state=42)
    cell_ids = grid["h3"].tolist()
    cells_gdf = gpd.GeoDataFrame(
        {"id": grid["h3"]},
        geometry=gpd.points_from_xy(grid["lon"], grid["lat"]), crs="EPSG:4326")

    spots = pd.read_parquet(INTERIM / "freizeit_spots.parquet")  # only the SOURCES rows (03b)
    global r5py
    ctx = setup_decomp_net()            # table-less net (cache + buildDistanceTables skip)
    r5py = ctx.r5py                     # matrix60 reads the module global
    egress = load_egress(ctx, cell_ids)
    g_walk_all = gather_window(egress["walk"], egress["erow"])   # walk egress, all cells (transit)
    cell_index = pd.Index(cell_ids)                              # dest cell h3 → grid row
    gx, gy = _T25832.transform(grid["lon"].to_numpy(), grid["lat"].to_numpy())
    tree = cKDTree(np.c_[gx, gy])
    modes = (ctx.TransportMode.BICYCLE, ctx.TransportMode.WALK, ctx.TransportMode.CAR)
    win = ", ".join(f"{k} {int(v)}km" for k, v in MODE_WINDOW_KM.items())

    raw = {}
    for src, spec in SOURCES.items():
        s_cells, mass, s_lat, s_lon = load_spots(spots, spec)
        if not s_cells:  # category absent from freizeit_spots → flat-zero (don't crash)
            print(f"{src}: no spots found — flat-zero surface")
            for mode in ("transit", "bike", "foot", "car"):
                raw[f"{src}_{mode}"] = np.zeros(len(cell_ids))
                raw[f"{src}_{mode}_tmin"] = np.full(len(cell_ids), np.inf)
            continue
        if _LIMIT and len(s_cells) > _LIMIT:   # value/rate check: a spread subset of spots
            idx = np.linspace(0, len(s_cells) - 1, _LIMIT).astype(int)
            s_cells = [s_cells[i] for i in idx]; mass = mass[idx]
            s_lat = [s_lat[i] for i in idx]; s_lon = [s_lon[i] for i in idx]
        print(f"{src}: {len(s_cells)} snapped spots -> {len(cell_ids)} cells "
              f"(transit all-cells; direct {win}; batch {SPOT_BATCH})", flush=True)
        grav, tmin = route_source(ctx, g_walk_all, cell_index, cells_gdf, cell_ids, tree,
                                  s_cells, mass, s_lat, s_lon, departure, evening, modes)
        for mode, v in grav.items():
            raw[f"{src}_{mode}"] = v
            raw[f"{src}_{mode}_tmin"] = tmin[mode]
        print("  " + src + ": " + " | ".join(
            f"{m} reach {100*(grav[m] > 0).mean():4.1f}% max {grav[m].max():6.2f}"
            for m in ("transit", "bike", "foot", "car")), flush=True)

    if _SAMPLE or _LIMIT:
        print("(value/smoke check — not writing)")
        return
    LAYERS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(LAYERS / RAW_NPZ, cell_ids=np.asarray(cell_ids, dtype=str), **raw)
    print(f"wrote {RAW_NPZ} ({len(cell_ids)} cells, {len(raw)} source×mode sums) "
          f"— 04e_freizeit derives the [0,1] surfaces")


if __name__ == "__main__":
    main()
