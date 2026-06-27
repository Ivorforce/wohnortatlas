"""Shared helpers for the reverse-routing pipeline (04c centers, 04d POI spots)
and the route-free derivation steps (04e Freizeit).

Kept in one place so the routing/derivation scripts don't each re-implement them:
  - load_r5(): lazily import wohnen.r5_routing (the r5py setup + parallel
    TravelTimeMatrix monkeypatch + matrix()). Lazy so the pure-derivation steps never
    pull in r5py / the JDK-21 assertion.
  - gravity_sum(): the decayed-reachability kernel Σ_source mass·exp(-t/τ), shared by
    04d (POI spots) and 04e (going-out centers).
"""

import datetime as dt
import os
import threading
import types
from pathlib import Path

import numpy as np

from wohnen.config import INTERIM, LAYERS

# Per-mode reach columns of reach_centers.npz, in ship order. This list IS the npz
# schema — 04c writes it, 04e/04f/04h read it, 22 packs it; they must agree or the web
# binary silently desyncs. walk_min stays LAST so a 4-mode web chunk built before foot
# existed still decodes (decodeTarget infers the count); keep in sync with web/decode.js.
MODE_COLS = ["transit_hbf_min", "transit_bike_min", "bike_hbf_min", "car_hbf_min", "walk_min"]

# client mode -> the npz column(s) whose MIN drives that mode's decay (transit picks the
# better of walk-access / bike-egress). Shared by the M/B/O derivations (04f, 04h).
MODE_DECAY = {
    "bike": ["bike_hbf_min"],
    "car": ["car_hbf_min"],
    "walk": ["walk_min"],
    "transit": ["transit_hbf_min", "transit_bike_min"],
}

_t04 = None


def load_r5():
    """Return the wohnen.r5_routing module (exposes r5py, matrix, and applies the
    parallel-TravelTimeMatrix monkeypatch + r5py setup on first import). Imported
    lazily here so importing wohnen.reach alone never starts r5py / the JVM."""
    global _t04
    if _t04 is None:
        from wohnen import r5_routing
        _t04 = r5_routing
    return _t04


# --- egress-decomposition R5 setup (shared by 04c centers, 04d freizeit, 04g egress) ---
# The decomposition computes transit reach as honest_stop_times(point) min-plus the
# precomputed egress.npz (stop→cell walk/bike). It therefore does NOT need R5's
# stop→street-vertex egress distance tables (buildDistanceTables) — and skipping them
# is what lets the 688k-stop NATIONAL net build & load within the 22 GB heap
# (buildDistanceTables OOMs otherwise). FastRaptorWorker never reads those tables
# (validated bit-identical with them cleared), so this is lossless for the raptor.
#
# ALL transit reach decomposes (04c + 04d), so nothing needs the egress tables —
# setup_decomp_net applies the skip + the dedicated table-less cache uniformly for
# every routing script. The net is kept in a
# SEPARATE cache dir (not the default ~/.cache/r5py) so a table-less build never gets
# served to some future r5py consumer that *does* want full egress.
DECOMP_CACHE = Path.home() / ".cache" / "wohnortatlas-r5-noegress"

_HOUR = dt.timedelta(hours=1)             # range-raptor departure window
_EGRESS_CAP = dt.timedelta(minutes=30)    # walk access cap; egress.npz is capped to match
_tls = threading.local()                  # per-worker RegionalTask cache (see honest_stop_times)

_skip_registered = False


def use_decomp_cache():
    """Point r5py's network cache at the table-less DECOMP_CACHE. Call BEFORE load_r5
    (r5py reads XDG_CACHE_HOME when it extracts its jar / builds nets)."""
    os.environ["XDG_CACHE_HOME"] = str(DECOMP_CACHE)


def skip_egress_distance_tables():
    """Shadow TransitLayer.buildDistanceTables → a no-op leaving empty egress tables.
    Call AFTER load_r5 (JVM started) and BEFORE building the network. Idempotent."""
    global _skip_registered
    if _skip_registered:
        return
    import jpype

    @jpype.JImplementationFor("com.conveyal.r5.transit.TransitLayer")
    class _TransitLayerNoEgressTables:  # noqa: N801
        def buildDistanceTables(self, geometry):
            self.stopToVertexDistanceTables = jpype.JClass("java.util.ArrayList")()

    _skip_registered = True


def setup_decomp_net(osm=None, gtfs=None):
    """Build/load the table-less transit net and return a context for the egress
    decomposition: net + r5py + matrix() + the transit/street layers + the jpype
    classes honest_stop_times needs. Applies the no-egress-tables cache + skip, so the
    national net fits in heap. REACH_OSM / REACH_GTFS env vars override the inputs
    (regional testing). Shared by 04c (centers), 04d (freizeit spots), 04g (egress)."""
    use_decomp_cache()
    r5 = load_r5()
    skip_egress_distance_tables()
    osm = str(osm or os.environ.get("REACH_OSM") or INTERIM / "region-filtered.osm.pbf")
    gtfs = str(gtfs or os.environ.get("REACH_GTFS") or INTERIM / "gtfs_region.zip")
    net = r5.r5py.TransportNetwork(osm, [gtfs])
    from jpype import JClass
    from r5py.r5.regional_task import RegionalTask
    from r5py import TransportMode
    tn = net._transport_network
    return types.SimpleNamespace(
        net=net, r5py=r5.r5py, matrix=r5.matrix,
        tl=tn.transitLayer, sl=tn.streetLayer,
        RegionalTask=RegionalTask, TransportMode=TransportMode,
        StreetRouter=JClass("com.conveyal.r5.streets.StreetRouter"),
        FastRaptorWorker=JClass("com.conveyal.r5.profile.FastRaptorWorker"),
        StreetMode=JClass("com.conveyal.r5.profile.StreetMode"),
        DURATION=JClass("com.conveyal.r5.streets.StreetRouter$State$RoutingVariable").DURATION_SECONDS,
        UNREACHED=int(JClass("com.conveyal.r5.profile.FastRaptorWorker").UNREACHED),
    )


def load_egress(ctx, cell_ids):
    """Load egress.npz, align its stop columns to THIS net's R5 stop indexing, and map
    each cell (in cell_ids order) to its CSR row. Returns {"walk": csr, "bike": csr,
    "erow": int[len(cell_ids)]}, csr = (indptr, indices, data) with indices = R5 stop
    index, data = minutes. egress.npz is keyed by GTFS id, so it survives stop
    re-indexing between builds. REACH_EGRESS env overrides the path (regional testing)."""
    tl = ctx.tl
    nstops = int(tl.getStopCount()); sidx = tl.stopIdForIndex
    r5_to_gtfs = np.array([str(sidx.get(i)).split(":")[-1] for i in range(nstops)])
    z = np.load(os.environ.get("REACH_EGRESS", str(LAYERS / "egress.npz")), allow_pickle=False)
    egr_gtfs = z["stop_gtfs_ids"].astype(str)
    if np.array_equal(egr_gtfs, r5_to_gtfs):
        widx, bidx = z["walk_indices"], z["bike_indices"]          # same net build — identity
    else:
        g2r = {g: i for i, g in enumerate(r5_to_gtfs)}
        remap = np.array([g2r.get(g, -1) for g in egr_gtfs], dtype=np.int64)
        assert (remap >= 0).all(), "egress.npz references stops absent from the current net"
        widx, bidx = remap[z["walk_indices"]], remap[z["bike_indices"]]
    cap = int(z["cap_min"])
    assert cap >= _EGRESS_CAP.total_seconds() / 60, \
        f"egress.npz capped at {cap} min < required {_EGRESS_CAP}"
    erow_of = {h: i for i, h in enumerate(z["cell_ids"].astype(str))}
    missing = [h for h in cell_ids if h not in erow_of]
    assert not missing, f"{len(missing)} cells absent from egress.npz (e.g. {missing[:3]}) — rerun 04g"
    erow = np.fromiter((erow_of[h] for h in cell_ids), dtype=np.int64, count=len(cell_ids))
    return {
        "walk": (z["walk_indptr"], widx.astype(np.int32), z["walk_data"]),
        "bike": (z["bike_indptr"], bidx.astype(np.int32), z["bike_data"]),
        "erow": erow,
    }


def honest_stop_times(ctx, lat, lon, horizon_min, departure):
    """R5's HONEST per-stop arrival times for one origin: [nIter, nStops] minutes
    (inf=unreached). Walk access ≤30 min → access stops → FastRaptorWorker.route(),
    giving the true alighting time at every stop WITHOUT R5's egress propagation
    (routing to a stop's coordinate alights at a faster neighbour and walks in — the
    'contamination' that made the naive decomposition optimistic). The decomposition
    supplies egress from egress.npz instead. The RegionalTask is cached per worker
    thread (only fromLat/fromLon vary) — rebuilding it per call is the GIL bottleneck."""
    key = (horizon_min, departure)
    if getattr(_tls, "key", None) != key:
        task = ctx.RegionalTask(transport_network=ctx.net, origin=None, departure=departure,
            departure_time_window=_HOUR, percentiles=[50],
            max_time=dt.timedelta(minutes=horizon_min), max_time_walking=_EGRESS_CAP,
            transport_modes=[ctx.TransportMode.TRANSIT], access_modes=[ctx.TransportMode.WALK])
        _tls.task = task               # keep the wrapper alive (prevents GC of the Java task)
        _tls.jt = task._regional_task
        _tls.key = key
    jt = _tls.jt
    jt.fromLat = float(lat); jt.fromLon = float(lon)
    sr = ctx.StreetRouter(ctx.sl); sr.profileRequest = jt; sr.streetMode = ctx.StreetMode.WALK
    if not sr.setOrigin(float(lat), float(lon)):
        return None
    sr.timeLimitSeconds = jt.getMaxTimeSeconds(ctx.StreetMode.WALK)
    sr.quantityToMinimize = ctx.DURATION
    sr.route(); sr.keepRoutingOnFoot()
    w = ctx.FastRaptorWorker(ctx.tl, jt, sr.getReachedStops()); w.retainPaths = False
    st = np.asarray(w.route(), dtype=np.float32)                  # [nIter, nStops] sec (~165 MB nat.)
    st[st >= ctx.UNREACHED] = np.inf                             # UNREACHED ~2.1e9 → +inf in float32
    return st / 60.0                                            # minutes (inf = unreached)


def gather_window(csr, rows):
    """Ragged-gather the egress CSR rows for `rows` (cell indices) into flat arrays
    (cell_local, stop_idx, egress_min). cell_local (0..len(rows)-1, the position in
    `rows`) comes out ASCENDING by construction, so it is ready for reduceat."""
    indptr, indices, data = csr
    counts = (indptr[rows + 1] - indptr[rows]).astype(np.int64)
    total = int(counts.sum())
    if total == 0:
        return np.empty(0, np.int64), np.empty(0, np.int32), np.empty(0, np.float32)
    cell_local = np.repeat(np.arange(len(rows)), counts)
    starts = indptr[rows]
    off = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
    pos = np.repeat(starts, counts) + off
    return cell_local, indices[pos], data[pos]


def door_percentile(t2s, gathered, n_cells, pidx, horizon_min):
    """Per-cell door-to-door minutes from honest stop times + pre-gathered egress.
    door[iter] = min_stop( t2s[iter,stop] + egress ); value = R5's percentile index
    over the sorted per-iteration doors; NaN where unreachable or > horizon_min."""
    out = np.full(n_cells, np.nan, np.float32)
    cell_local, stop_idx, egr = gathered
    if len(stop_idx) == 0:
        return out
    keep = np.isfinite(t2s).any(axis=0)[stop_idx]     # only stops the raptor reached
    if not keep.any():
        return out
    cl, st, eg = cell_local[keep], stop_idx[keep], egr[keep]
    door = (t2s[:, st].astype(np.float32) + eg[None, :])          # [nIter, nKept]
    uniq, first = np.unique(cl, return_index=True)               # cl ascending → reduceat boundaries
    seg = np.minimum.reduceat(door, first, axis=1)               # [nIter, nUniq] per-cell min/iter
    seg.sort(axis=0)                                             # inf sinks to the bottom
    val = seg[pidx]
    out[uniq] = np.where(val > horizon_min, np.nan, val)
    return out


def morton_order(x, y):
    """Z-order (Morton) sort indices for points — consecutive indices are spatially
    near, so contiguous batch slices are geographically local (small shared routing
    window). Shared by 04c (centers) and 04d (freizeit spots)."""
    def _q(v):
        v = ((v - v.min()) / (np.ptp(v) + 1e-9) * 65535).astype(np.uint32)
        v = (v | (v << 8)) & 0x00FF00FF
        v = (v | (v << 4)) & 0x0F0F0F0F
        v = (v | (v << 2)) & 0x33333333
        v = (v | (v << 1)) & 0x55555555
        return v
    return np.argsort(_q(x) | (_q(y) << 1))


def gravity_sum(times, mass, tau, horizon, chunk=256):
    """Decayed reachability per cell: Σ_source mass·exp(-t/τ), ignoring t ≥ horizon.

    times: (S, N) minutes from each of S sources to N cells — uint8 with a ≥255
      sentinel OR float with NaN (both, and t ≥ horizon, count as unreachable).
    mass:  (S,) pull weight per source. Returns (N,) raw gravity (un-normalized).
    Summed in CHUNKS over the source axis so peak memory stays bounded to one
    (chunk, N) exp() regardless of S — keeps the derivation lean (no 5× float64
    center×cell copies) as the center count grows."""
    times = np.asarray(times)
    mass = np.asarray(mass, dtype=float)
    out = np.zeros(times.shape[1])
    for i in range(0, times.shape[0], chunk):
        t = times[i:i + chunk].astype(float)
        t[np.isnan(t) | (t >= horizon)] = np.inf
        out += (mass[i:i + chunk, None] * np.exp(-t / tau)).sum(axis=0)
    return out
