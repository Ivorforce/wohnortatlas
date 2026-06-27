"""Reverse-route every commute CENTER (from 04b) to all cells and persist the raw
per-center×cell minutes to reach_centers.npz (uint8). This is the ONLY expensive
step; ALL derived maps are computed route-free from the npz by other scripts:
  - 04e_freizeit : the s_freizeit going-out surfaces.
  - 04f_aggregate : the per-CITY / per-PART reach (the Lebensmittelpunkt picker's
    named targets, a catchment-weighted percentile over a city's centers) AND the
    "Irgendeine Stadt / Großstadt" aggregates.
Keeping the derivations out of here means re-tuning them (percentile, decay, τ) costs
seconds and needs no JDK/GTFS — only a real input change forces a re-route.

The two TRANSIT modes are computed by an egress DECOMPOSITION, not a door-to-door
TravelTimeMatrix: door-to-door rebuilds R5's egress linkage (alighting stop → home
cell) once per 64-center batch (~42×) — ~14 h, dominated by the bike-egress table.
That egress is origin-independent, so 04g precomputes it ONCE (data/layers/egress.npz,
cell↔stop CSR, walk + regular bike) and we reuse it. Per center we read R5's HONEST
per-stop arrival times (FastRaptorWorker, skipping its own egress propagation), then
min-plus those against the cached egress, per departure-minute, at R5's exact
percentile index. Validated within 1 min of R5 door-to-door (scripts/04c_proto.py;
StreetRouter platform-vertex egress is ~+1 min conservative). The three DIRECT modes
(bike/walk/car) stay plain TravelTimeMatrix reverse routes. Bike is now a REGULAR bike
(r5py default speed) everywhere — the old special e-bike handling is dropped.

Reuses the wohnen.r5_routing scaffold (parallel TravelTimeMatrix + matrix()) via
wohnen.reach.load_r5, so the r5py setup lives in exactly one place.

Usage: 04c_reach.py [--sample N]   (route only N centers to a cell sample; no write)
"""

import concurrent.futures
import datetime as dt
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# capture our flag BEFORE importing 04 — its module body rewrites sys.argv (strips
# everything but --max-memory for r5py), so reading --sample later would miss it.
_SAMPLE = int(sys.argv[sys.argv.index("--sample") + 1]) if "--sample" in sys.argv else None

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree

from wohnen.config import (
    LAYERS, INTERIM, CAR_CONGESTION, CAR_PARK_MIN, TRAVEL_SENTINEL_MIN,
)
from wohnen.reach import (
    door_percentile, gather_window, honest_stop_times, load_egress, morton_order,
    setup_decomp_net)

# r5py is pulled in only when routing (setup_decomp_net → the JDK-21 assertion); set in
# main. This script ONLY routes + persists the per-center×cell reach to reach_centers.npz;
# the derived maps (04e freizeit, 04f city/part + aggregates) come from the npz route-free.
_CTX = None      # decomposition context (net + r5py + matrix + jpype classes) — main
_EGRESS = None   # {"walk": csr, "bike": csr, "erow": int[ncells]} — main

MODES = ["transit_hbf_min", "transit_bike_min", "bike_hbf_min", "car_hbf_min", "walk_min"]
HORIZON_MIN = TRAVEL_SENTINEL_MIN           # transit reach horizon (== the 255 sentinel)
_TRANSIT_WORKERS = int(os.environ.get("REACH_TRANSIT_WORKERS", str(max(4, (os.cpu_count() or 8) - 2))))

# Centers are routed BATCH at a time so peak RAM is one batch's matrices
# (BATCH × n_cells × 5 modes), not the full n_centers × n_cells float (~11 GB/mode
# nationally). Each batch persists a fragment, so a crash/kill resumes mid-run.
BATCH = int(os.environ.get("REACH_BATCH", "64"))
# Geographic reach window: route each center only to cells within WINDOW_KM. The
# routing already caps reach at max_time=120 min, so cells past the window are the
# 255 sentinel anyway — this just stops paying to extract travel times to all 536k
# cells (~15 s/center) when only the local ~1/7 are reachable. Must comfortably
# exceed the 120 min reach of the fastest mode (autobahn car) so nothing real is
# dropped; ~Munich-map radius. Batches are spatially sorted so a batch's centers
# share one small window. Set REACH_WINDOW_KM=0 to disable (route all cells).
WINDOW_KM = float(os.environ.get("REACH_WINDOW_KM", "150"))
_T25832 = Transformer.from_crs(4326, 25832, always_xy=True)  # metres, good over DE


def _transit_decompose(centers, cells, dest_erows, evening, bike_hbf_df):
    """The two transit-mode DataFrames (index=center id, cols=cell h3) via the
    egress decomposition. Egress over the (shared) window is gathered ONCE; only the
    honest stop times differ per center. transit_bike folds in the direct bike."""
    cids = centers["id"].to_numpy()
    lat = centers["lat"].to_numpy(); lon = centers["lon"].to_numpy()
    dest_h3 = cells["id"].to_numpy()
    n = len(dest_h3)
    g_walk = gather_window(_EGRESS["walk"], dest_erows)
    g_bike = gather_window(_EGRESS["bike"], dest_erows)
    th = np.full((len(cids), n), np.nan, np.float32)
    tb = np.full((len(cids), n), np.nan, np.float32)

    def one(i):
        t2s = honest_stop_times(_CTX, lat[i], lon[i], HORIZON_MIN, evening)
        if t2s is None:
            return
        pidx = int(np.ceil(0.5 * t2s.shape[0]) - 1)             # R5 findPercentileIndex(nIter, 50)
        th[i] = door_percentile(t2s, g_walk, n, pidx, HORIZON_MIN)
        tb[i] = door_percentile(t2s, g_bike, n, pidx, HORIZON_MIN)

    with concurrent.futures.ThreadPoolExecutor(max_workers=_TRANSIT_WORKERS) as ex:
        list(ex.map(one, range(len(cids))))

    # fold the direct bike into transit_bike (ÖPNV+Rad = best of transit-with-bike-egress
    # and just-cycling); fmin treats NaN as "unreachable", so either side may win.
    bike = bike_hbf_df.reindex(index=cids, columns=dest_h3).to_numpy(dtype=np.float64)
    tb = np.fmin(tb.astype(np.float64), bike)
    return (pd.DataFrame(th, index=cids, columns=dest_h3),
            pd.DataFrame(tb, index=cids, columns=dest_h3))


def route_centers(centers, cells, dest_erows, departure, evening) -> dict:
    """{mode: DataFrame(index=center_id, columns=cell h3)} — reverse routing. Direct
    modes via TravelTimeMatrix; transit modes via the honest-stops + egress decomp."""
    net, matrix, TM = _CTX.net, _CTX.matrix, _CTX.TransportMode
    B, W, C = TM.BICYCLE, TM.WALK, TM.CAR
    out = {}
    print("  bike direct ...");  out["bike_hbf_min"] = matrix(net, centers, cells, evening, [B])
    print("  walk direct ...");  out["walk_min"]     = matrix(net, centers, cells, evening, [W])
    print("  car ...");          car = matrix(net, centers, cells, departure, [C])
    out["car_hbf_min"] = (car * CAR_CONGESTION + CAR_PARK_MIN).clip(upper=TRAVEL_SENTINEL_MIN)
    print("  transit (decompose) ...")
    th, tb = _transit_decompose(centers, cells, dest_erows, evening, out["bike_hbf_min"])
    out["transit_hbf_min"] = th
    out["transit_bike_min"] = tb
    return out


REACH_NPZ = "reach_centers.npz"   # the assembled per-center×cell minutes (04e/04f input)
PARTS_DIR = "reach_centers_parts"  # per-batch fragments — resumable scratch, safe to delete


def _to_uint8(reach: dict, cids, cell_ids) -> dict:
    """{mode: DataFrame} -> {mode: uint8 (len(cids) × len(cell_ids))}, 255=unreach.
    uint8 minutes keep the persisted reach ~5× smaller than float; 255 doubles as
    "beyond the TRAVEL_SENTINEL_MIN routing horizon" (the natural reach window).
    Every mode is capped at the horizon: r5py's street routing reports DIRECT
    walk/bike times slightly past max_time, so without this they'd leak values
    >120 (transit already caps in the decomposition, car clips in route_centers) —
    breaking the clean ≤horizon-reachable / 255-beyond invariant the npz relies on."""
    arrs = {}
    for m in MODES:
        a = reach[m].reindex(index=cids, columns=cell_ids).to_numpy(dtype=float)
        r = np.round(a)
        arrs[m] = np.where(np.isnan(a) | (r > TRAVEL_SENTINEL_MIN), 255, r).astype(np.uint8)
    return arrs


def _route_batch(batch, cells_gdf, cell_ids, erow, departure, evening, tree=None) -> dict:
    """Route this batch's centers -> {mode: uint8 (len(batch) × len(cell_ids))}.
    With a cell cKDTree (EPSG:25832), route only to cells within WINDOW_KM of the
    batch; _to_uint8 reindexes back to ALL cell_ids, so windowed-out cells become
    255 — they're beyond the 120 min horizon anyway, so the result is unchanged.
    `erow` maps each cell (cell_ids order) to its row in the egress CSR."""
    centers_gdf = gpd.GeoDataFrame(
        {"id": batch["id"], "lat": batch["lat"], "lon": batch["lon"]},
        geometry=gpd.points_from_xy(batch["lon"], batch["lat"]), crs="EPSG:4326")
    dest, dest_erows = cells_gdf, erow
    if tree is not None and WINDOW_KM > 0:
        bx, by = _T25832.transform(batch["lon"].to_numpy(), batch["lat"].to_numpy())
        hit = tree.query_ball_point(np.c_[bx, by], r=WINDOW_KM * 1000.0)
        win = (np.unique(np.concatenate([np.asarray(h, dtype=int) for h in hit]))
               if len(batch) else np.empty(0, int))
        dest = cells_gdf.iloc[win]
        dest_erows = erow[win]
    reach = route_centers(centers_gdf, dest, dest_erows, departure, evening)
    return _to_uint8(reach, batch["id"].to_numpy(), cell_ids)


def _assemble(parts: Path, cids_all, cell_ids):
    """Scatter the per-batch fragments into reach_centers.npz in the canonical
    centers.parquet order — by id, NOT fragment order, since batches are spatially
    sorted. Pre-allocate the output (255-filled) and fill each fragment's rows in
    place, so peak memory is one copy of the result, not fragments + a concat."""
    frags = sorted(parts.glob("part_*.npz"))
    expect = cids_all.astype(str)
    pos = {str(c): i for i, c in enumerate(expect)}
    merged = {m: np.full((len(expect), len(cell_ids)), 255, np.uint8) for m in MODES}
    seen = 0
    for f in frags:
        d = np.load(f, allow_pickle=False)
        fids = d["center_ids"]
        rows = np.fromiter((pos[str(c)] for c in fids), dtype=int, count=len(fids))
        for m in MODES:
            merged[m][rows] = d[m]
        seen += len(fids)
    assert seen == len(expect), \
        f"assembled {seen} center-rows != {len(expect)} centers — rm {parts}/ and rerun"
    np.savez_compressed(LAYERS / REACH_NPZ, center_ids=expect,
                        cell_ids=np.asarray(cell_ids, dtype=str), **merged)
    print(f"wrote {REACH_NPZ} ({len(expect)} centers × {len(cell_ids)} cells) from "
          f"{len(frags)} fragments — 04e derives freizeit, 04f the city/part + aggregates.\n"
          f"  ({parts.name}/ kept for resume; delete it once the build is confirmed)")


def main():
    sample = _SAMPLE
    departure = dt.datetime.fromisoformat(
        json.loads((INTERIM / "departure.json").read_text())["departure"])
    evening = departure.replace(hour=17, minute=0, second=0, microsecond=0)

    centers = pd.read_parquet(LAYERS / "centers.parquet")
    grid = pd.read_parquet(INTERIM / "grid.parquet")
    if sample:
        centers = centers.head(sample)
        grid = grid.sample(min(2000, len(grid)), random_state=42)
    cell_ids = grid["h3"].tolist()
    cells_gdf = gpd.GeoDataFrame(
        {"id": grid["h3"]},
        geometry=gpd.points_from_xy(grid["lon"], grid["lat"]), crs="EPSG:4326")

    global _CTX, _EGRESS  # routing path only — setup_decomp_net pulls in r5py + JDK-21 assertion
    _CTX = setup_decomp_net()           # table-less net (cache + buildDistanceTables skip)
    _EGRESS = load_egress(_CTX, cell_ids)
    erow = _EGRESS["erow"]              # egress CSR row per cell, in cell_ids order

    if sample:
        print(f"{len(centers)} centers -> {len(grid)} cells; "
              f"departure {departure} / return {evening}")
        _route_batch(centers, cells_gdf, cell_ids, erow, departure, evening)
        print("(smoke test — not writing)")
        return

    # Route BATCH centers at a time into per-batch npz fragments under
    # reach_centers_parts/. Peak RAM is one batch's matrices (not the full
    # 2653×536k float ≈ 11 GB/mode), and a crash/kill resumes by skipping the
    # fragments already on disk. The final pass concatenates them into the npz.
    parts = LAYERS / PARTS_DIR
    parts.mkdir(parents=True, exist_ok=True)
    # Spatially sort centers so each contiguous BATCH is geographically local (one
    # small window). centers stays the canonical order; _assemble restores it by id.
    cx, cy = _T25832.transform(centers["lon"].to_numpy(), centers["lat"].to_numpy())
    centers_spatial = centers.iloc[morton_order(cx, cy)].reset_index(drop=True)
    gx, gy = _T25832.transform(grid["lon"].to_numpy(), grid["lat"].to_numpy())
    tree = cKDTree(np.c_[gx, gy]) if WINDOW_KM > 0 else None
    n = len(centers)
    nb = (n + BATCH - 1) // BATCH
    win = f"window {WINDOW_KM:.0f} km" if tree is not None else "all cells (no window)"
    print(f"{n} centers -> {len(cell_ids)} cells; {nb} batches of {BATCH}; {win}; "
          f"departure {departure} / return {evening} (resumable: {parts.name}/)")
    for bi in range(nb):
        frag = parts / f"part_{bi:04d}.npz"
        if frag.exists() and frag.stat().st_size > 0:
            print(f"  batch {bi + 1}/{nb}: cached, skipping")
            continue
        batch = centers_spatial.iloc[bi * BATCH:(bi + 1) * BATCH]
        print(f"  batch {bi + 1}/{nb}: routing {len(batch)} centers ...", flush=True)
        arrs = _route_batch(batch, cells_gdf, cell_ids, erow, departure, evening, tree)
        # write to a "_"-prefixed temp (must end .npz so savez doesn't append it,
        # and must NOT match the part_*.npz glob in _assemble) then atomic-rename,
        # so a half-written fragment never looks complete to the resume skip.
        tmp = frag.with_name("_" + frag.name)
        np.savez_compressed(tmp, center_ids=batch["id"].to_numpy().astype(str), **arrs)
        tmp.rename(frag)
    _assemble(parts, centers["id"].to_numpy(), cell_ids)


if __name__ == "__main__":
    main()
