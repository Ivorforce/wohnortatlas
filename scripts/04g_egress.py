"""Precompute the national stop->cell egress tables (walk + regular bike), ONCE, cached.

Egress (alighting stop -> home cell) is origin-independent, so the transit decomposition (04c)
reuses this instead of R5 rebuilding the egress linkage per batch (the ~14 h cost).

Method: one capped StreetRouter per cell, read getReachedStops() -> reachable stops only
(sparse; parallelises cleanly in Java — the r5py TravelTimeMatrix path is GIL-bound for these
tiny routes and would take ~27 h). Egress time = walk/bike route TIME to the stop's PLATFORM
vertex (≈+1 min vs R5's coordinate-linking shortcut, conservative; arguably more realistic).
Regular bike at r5py default speed (special e-bike handling dropped); both capped EGRESS_CAP_MIN.
Output: data/layers/egress.npz, cell-major CSR keyed by GTFS stop_id.

Usage: JAVA_HOME=... poetry run python scripts/04g_egress.py [--sample N]
"""
import sys, os, datetime as dt, json, io, zipfile, concurrent.futures, threading
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
_SAMPLE = int(sys.argv[sys.argv.index("--sample") + 1]) if "--sample" in sys.argv else None

import numpy as np, pandas as pd
from wohnen.config import INTERIM, LAYERS
from wohnen.reach import load_r5, skip_egress_distance_tables, use_decomp_cache

EGRESS_CAP_MIN = 30
OUT = Path(os.environ.get("EGRESS_OUT", str(LAYERS / "egress.npz")))


def main():
    use_decomp_cache()                  # table-less net cache (before r5py loads)
    t04 = load_r5(); r5py = t04.r5py
    skip_egress_distance_tables()       # skip buildDistanceTables (egress.npz replaces it)
    from jpype import JClass
    StreetRouter = JClass("com.conveyal.r5.streets.StreetRouter")
    StreetMode = JClass("com.conveyal.r5.profile.StreetMode")
    DURATION = JClass("com.conveyal.r5.streets.StreetRouter$State$RoutingVariable").DURATION_SECONDS
    from r5py.r5.regional_task import RegionalTask
    from r5py import TransportMode

    osm = os.environ.get("EGRESS_OSM", str(INTERIM/"region-filtered.osm.pbf"))
    gtfs = os.environ.get("EGRESS_GTFS", str(INTERIM/"gtfs_region.zip"))
    net = r5py.TransportNetwork(osm, [gtfs]); print("net ready", flush=True)
    tn = net._transport_network; sl = tn.streetLayer; tl = tn.transitLayer
    nstops = int(tl.getStopCount()); sidx = tl.stopIdForIndex
    r5_to_gtfs = np.array([str(sidx.get(i)).split(":")[-1] for i in range(nstops)])

    # tasks just provide a ProfileRequest (speed/LTS); bike = r5py default speed (regular bike)
    twalk = RegionalTask(transport_network=net, transport_modes=[TransportMode.WALK],
                         access_modes=[TransportMode.WALK])._regional_task
    tbike = RegionalTask(transport_network=net, transport_modes=[TransportMode.BICYCLE],
                         access_modes=[TransportMode.BICYCLE])._regional_task
    cap_s = EGRESS_CAP_MIN * 60

    def reached(lat, lon, mode, jt):
        sr = StreetRouter(sl); sr.profileRequest = jt; sr.streetMode = mode
        if not sr.setOrigin(float(lat), float(lon)):
            return np.empty(0, np.int32), np.empty(0, np.float32)
        sr.timeLimitSeconds = cap_s; sr.quantityToMinimize = DURATION; sr.route()
        if mode == StreetMode.WALK:
            sr.keepRoutingOnFoot()
        m = sr.getReachedStops()
        keys = np.asarray(m.keys(), dtype=np.int32)
        vals = np.fromiter((m.get(int(k)) / 60.0 for k in keys), np.float32, len(keys))
        return keys, vals

    grid = pd.read_parquet(INTERIM/"grid.parquet")
    if _SAMPLE:
        grid = grid.sample(_SAMPLE, random_state=1).reset_index(drop=True)
    lat = grid.lat.to_numpy(); lon = grid.lon.to_numpy(); cell_ids = grid.h3.to_numpy()
    N = len(grid)
    workers = max(4, (os.cpu_count() or 8) - 2)
    print(f"{N} cells, {nstops} stops, {workers} workers, cap {EGRESS_CAP_MIN} min", flush=True)

    def build(mode, jt, label):
        idx_per = [None] * N; val_per = [None] * N; done = [0]; lock = threading.Lock()
        def one(i):
            idx_per[i], val_per[i] = reached(lat[i], lon[i], mode, jt)
            with lock:
                done[0] += 1
                if done[0] % 50000 == 0: print(f"  {label}: {done[0]}/{N}", flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(one, range(N)))
        lengths = np.fromiter((len(a) for a in idx_per), np.int64, N)
        indptr = np.zeros(N + 1, np.int64); np.cumsum(lengths, out=indptr[1:])
        indices = np.concatenate(idx_per).astype(np.int32) if N else np.empty(0, np.int32)
        data = np.concatenate(val_per).astype(np.float32) if N else np.empty(0, np.float32)
        print(f"  {label}: {len(data)} (cell,stop) pairs", flush=True)
        return indptr, indices, data

    wptr, widx, wdat = build(StreetMode.WALK, twalk, "walk")
    bptr, bidx, bdat = build(StreetMode.BICYCLE, tbike, "bike")
    np.savez_compressed(OUT, cell_ids=cell_ids.astype(str), stop_gtfs_ids=r5_to_gtfs.astype(str),
        walk_indptr=wptr, walk_indices=widx, walk_data=wdat,
        bike_indptr=bptr, bike_indices=bidx, bike_data=bdat, cap_min=np.int32(EGRESS_CAP_MIN))
    print(f"wrote {OUT} ({'SAMPLE ' if _SAMPLE else ''}{N} cells)", flush=True)


if __name__ == "__main__":
    main()
