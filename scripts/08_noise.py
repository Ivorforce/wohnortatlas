"""Noise layer: official END Lden maps (UBA federal road + aircraft, EBA rail),
energy-summed in dB, plus an OSM minor-road proxy and a diffuse-urban term.

All three transport sources are real acoustic Lden (dB) on the same anchor
(55 dB -> 0, 70 dB -> 1), so they are directly comparable and combine by
energy sum (10*log10 of summed intensities) the way sound physically does —
not by max() of incomparable proxies.
"""

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from scipy.spatial import cKDTree

from wohnen.config import CACHE, EBA_WFS, INTERIM, LAYERS, UBA_NOISE_WMS
from wohnen.dl import cached_download, cached_get_json
from wohnen.norm import clip01

ROAD_W = {"motorway": 1.0, "trunk": 0.8, "primary": 0.5, "secondary": 0.25,
          "tertiary": 0.12}

# dB anchor shared by every source: Lden 55 -> 0 penalty, 70 -> 1.0.
DB_LO, DB_SPAN = 55.0, 15.0

_T25832 = Transformer.from_crs(4326, 25832, always_xy=True)


def to_25832(lon, lat):
    return _T25832.transform(lon, lat)


def anchor(db):
    """Lden dB -> 0..1 penalty; NaN (no band / quiet) -> 0."""
    return np.nan_to_num(clip01((np.asarray(db, float) - DB_LO) / DB_SPAN))


def energy_sum(*dbs):
    """Acoustic sum of Lden levels: 10*log10(Σ 10^(L/10)). Sources absent
    (NaN) contribute zero intensity; all-absent -> NaN."""
    lin = np.zeros_like(np.asarray(dbs[0], float))
    any_present = np.zeros(len(lin), bool)
    for db in dbs:
        db = np.asarray(db, float)
        present = ~np.isnan(db)
        lin[present] += 10.0 ** (db[present] / 10.0)
        any_present |= present
    out = np.full(len(lin), np.nan)
    out[any_present] = 10.0 * np.log10(lin[any_present])
    return out


# --- UBA federal official Lden raster (classified PNG over WMS) ------------
#
# The UBA noise WMS renders Lden as five fixed 5 dB band colors — verified
# byte-for-byte identical to the per-Land (LfU) palette, 2026-06. We map each
# color to its band-center dB; PNG alpha=0 marks "no band here" (<55 dB).
_BAND_RGB = np.array([[226, 242, 191], [243, 198, 131], [205, 70, 62],
                      [117, 8, 92], [67, 10, 74]], float)
_BAND_DB = np.array([57.5, 62.5, 67.5, 72.5, 77.5])
_COLOR_TOL2 = 1500.0  # max squared RGB dist to count a pixel as a band color

_TILE_M = 20_000.0    # tile span; at 10 m/px -> 2000 px (UBA ArcGIS caps ≤4096)
_TILE_PX = 2000
# UBA is dynamic-render-only (no WMTS/cache) and slow per tile (~50 s for a
# 2000 px render); a national grid needs ~hundreds of land tiles per layer, so
# we prefetch them concurrently — ~8x faster than serial. The bounded worker
# count IS the politeness throttle (each tile holds a connection for its whole
# render). Override with UBA_WORKERS to throttle mid-run: a kill + restart
# resumes from the on-disk tile cache, so the worker count is free to change.
_FETCH_WORKERS = int(os.environ.get("UBA_WORKERS", "8"))


def _tile_path(layer, ti, tj):
    return CACHE / "lden_uba" / f"{layer}_{ti}_{tj}.png"


def _fetch_tile(layer, x0, y0, x1, y1, dest):
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    url = (UBA_NOISE_WMS +
           "?service=WMS&version=1.3.0&request=GetMap&styles="
           f"&layers={layer}&crs=EPSG:25832"
           f"&bbox={x0:.1f},{y0:.1f},{x1:.1f},{y1:.1f}"
           f"&width={_TILE_PX}&height={_TILE_PX}"
           "&transparent=true&format=image/png")
    time.sleep(0.4)  # polite spacing for the UBA server
    return cached_download(url, dest, desc=dest.name, progress=False)


def _decode_tile(path):
    """PNG tile -> (H, W) Lden dB array, NaN where no band."""
    with rasterio.open(path) as ds:
        a = ds.read()  # (bands, H, W)
    h, w = a.shape[1], a.shape[2]
    rgb = np.stack([a[0], a[1], a[2]], -1).reshape(-1, 3).astype(float)
    alpha = (a[3].reshape(-1) if a.shape[0] >= 4
             else np.full(h * w, 255, np.uint8))
    best = np.full(len(rgb), np.inf)
    band = np.zeros(len(rgb), int)
    for k in range(len(_BAND_RGB)):
        dk = ((rgb - _BAND_RGB[k]) ** 2).sum(1)
        m = dk < best
        best[m], band[m] = dk[m], k
    db = _BAND_DB[band]
    db[(best > _COLOR_TOL2) | (alpha == 0)] = np.nan
    return db.reshape(h, w)


def _enumerate_tiles(pts, sources):
    """List of (layer, ti, tj, tx0, ty0, tx1, ty1, idx) for every fixed-grid
    tile that holds a point — idx is the compact int array of point rows in the
    tile, NOT a full-length boolean mask (at national scale a mask per tile
    would be ~thousands × millions of bytes). Tiles align to a global EPSG:25832
    grid (index = floor(coord / _TILE_M)) so the on-disk cache is geographically
    stable across calls; empty land/ocean/foreign tiles never appear."""
    ti = np.floor(pts[:, 0] / _TILE_M).astype(np.int64)
    tj = np.floor(pts[:, 1] / _TILE_M).astype(np.int64)
    groups = pd.DataFrame({"ti": ti, "tj": tj}).groupby(["ti", "tj"]).indices
    tiles = []
    for (gi, gj), idx in groups.items():
        tx0, ty0 = gi * _TILE_M, gj * _TILE_M
        for layer in sources:
            tiles.append((layer, int(gi), int(gj), tx0, ty0,
                          tx0 + _TILE_M, ty0 + _TILE_M, idx))
    return tiles


def _prefetch_tiles(tiles):
    """Fetch every uncached tile concurrently into the disk cache so the serial
    decode pass below reads from disk (no network). _fetch_tile is idempotent,
    so a failed prefetch is simply retried (blocking) by the decode pass."""
    todo = [(layer, tx0, ty0, tx1, ty1, _tile_path(layer, ti, tj))
            for (layer, ti, tj, tx0, ty0, tx1, ty1, _idx) in tiles
            if not (_tile_path(layer, ti, tj).exists()
                    and _tile_path(layer, ti, tj).stat().st_size > 0)]
    if not todo:
        return
    print(f"  prefetching {len(todo)} UBA tiles ({_FETCH_WORKERS} workers)...")
    done = errs = 0
    with ThreadPoolExecutor(max_workers=_FETCH_WORKERS) as ex:
        futs = [ex.submit(_fetch_tile, layer, x0, y0, x1, y1, dest)
                for (layer, x0, y0, x1, y1, dest) in todo]
        for fut in as_completed(futs):
            done += 1
            try:
                fut.result()
            except Exception as e:  # noqa: BLE001
                errs += 1
                if errs <= 5:
                    print(f"    WARNING: prefetch failed: {e}")
            if done % 50 == 0 or done == len(todo):
                print(f"    {done}/{len(todo)} fetched ({errs} errors)")


def sample_lden(pts_xy, sources):
    """Max official Lden (dB) at each EPSG:25832 point over the given UBA WMS
    layer ids. NaN where no source has a band there."""
    pts = np.asarray(pts_xy, float)
    out = np.full(len(pts), np.nan)
    res = _TILE_M / _TILE_PX
    tiles = _enumerate_tiles(pts, sources)
    _prefetch_tiles(tiles)  # parallel network; decode/assign below stays serial
    hits = {layer: 0 for layer in sources}
    for (layer, ti, tj, tx0, ty0, tx1, ty1, idx) in tiles:
        try:
            p = _fetch_tile(layer, tx0, ty0, tx1, ty1, _tile_path(layer, ti, tj))
            db = _decode_tile(p)
        except Exception as e:  # noqa: BLE001
            print(f"  WARNING: tile {layer} {ti},{tj}: {e}")
            continue
        col = ((pts[idx, 0] - tx0) / res).astype(int)
        row = ((ty1 - pts[idx, 1]) / res).astype(int)  # PNG row 0 = top
        np.clip(col, 0, _TILE_PX - 1, out=col)
        np.clip(row, 0, _TILE_PX - 1, out=row)
        v = db[row, col]
        cur = out[idx]
        out[idx] = np.where(np.isnan(cur), v,
                            np.where(np.isnan(v), cur,
                                     np.maximum(cur, v)))
        hits[layer] += int(np.isfinite(v).sum())
    for layer in sources:
        print(f"  layer {layer}: {hits[layer]} child-points in a band")
    return out


# --- EBA federal-rail Lden bands (vector WFS) ------------------------------
def fetch_eba_rail() -> gpd.GeoDataFrame | None:
    """Fetch Lden isophone bands for the bbox, paginated."""
    from wohnen.config import BBOX
    x0, y0 = to_25832(BBOX[0], BBOX[1])
    x1, y1 = to_25832(BBOX[2], BBOX[3])
    feats = []
    start = 0
    try:
        while True:
            page = cached_get_json(
                EBA_WFS,
                params={
                    "service": "WFS", "version": "2.0.0", "request": "GetFeature",
                    "typeNames": "app:Isophonenbaender_EK",
                    "outputFormat": "application/geo+json",
                    "count": "10000", "startIndex": str(start),
                    "bbox": f"{x0:.0f},{y0:.0f},{x1:.0f},{y1:.0f},"
                            "urn:ogc:def:crs:EPSG::25832",
                },
                cache_key=f"eba_{start}", rate_bucket="eba", rate_limit_s=1.0)
            got = page.get("features", [])
            feats.extend(got)
            print(f"  eba page @{start}: {len(got)} features")
            if len(got) < 10000:
                break
            start += 10000
    except Exception as e:
        print(f"WARNING: EBA WFS failed ({e})")
        if not feats:
            return None
    if not feats:
        return None
    return gpd.GeoDataFrame.from_features(feats, crs=4326)


def rail_db_at(child_pts) -> np.ndarray | None:
    """Per-child Lden dB from EBA bands (NaN outside any band)."""
    gdf = fetch_eba_rail()
    if gdf is None or "hh_measure" not in gdf.columns:
        return None
    if "gml_description" in gdf.columns:
        lden = gdf[gdf["gml_description"].astype(str)
                   .str.contains("Lden", case=False, na=False)]
        if len(lden):
            gdf = lden
    gdf["db"] = pd.to_numeric(gdf["hh_measure"], errors="coerce")
    gdf = gdf.dropna(subset=["db"])
    pts = gpd.GeoDataFrame(
        {"i": np.arange(len(child_pts))},
        geometry=gpd.points_from_xy(child_pts["lon"], child_pts["lat"]),
        crs=4326)
    j = gpd.sjoin(pts, gdf[["db", "geometry"]], how="left")
    db = j.groupby("i")["db"].max().reindex(range(len(child_pts)))
    print(f"  rail bands: {len(gdf)} polys, child-points in a band: "
          f"{int(db.notna().sum())}")
    return db.values


# --- OSM minor-road proxy (sub-threshold fallback) -------------------------
def proximity_decay(child_xy, samples, radius_m=600.0, scale_m=200.0):
    """Σ w · exp(−d/scale) at each child point."""
    gx, gy = child_xy[:, 0], child_xy[:, 1]
    sx, sy = to_25832(samples["lon"].values, samples["lat"].values)
    w = samples["w"].values
    tree = cKDTree(np.c_[sx, sy])
    out = np.zeros(len(gx))
    pairs = tree.query_ball_point(np.c_[gx, gy], r=radius_m)
    for i, idx in enumerate(pairs):
        if idx:
            d = np.hypot(sx[idx] - gx[i], sy[idx] - gy[i])
            out[i] = float(np.sum(w[idx] * np.exp(-d / scale_m)))
    return out


def hex_energy_mean(child_db, n):
    """Per-hex Lden (dB) = energy mean of its 7 res-9 child samples —
    the expected acoustic exposure of a uniform-in-cell resident. NaN
    children contribute zero intensity; all-NaN hex -> NaN."""
    child_db = np.asarray(child_db, float).reshape(n, 7)
    lin = np.where(np.isnan(child_db), 0.0, 10.0 ** (child_db / 10.0))
    s = lin.mean(1)
    out = np.full(n, np.nan)
    out[s > 0] = 10.0 * np.log10(s[s > 0])
    return out


def main():
    import h3

    grid = pd.read_parquet(INTERIM / "grid.parquet")
    roads = pd.read_parquet(INTERIM / "roads.parquet")
    n = len(grid)

    # 7 res-9 child centers per hex — sub-hex sampling for every source.
    child_ll = np.array([h3.cell_to_latlng(ch) for c in grid["h3"]
                         for ch in h3.cell_to_children(c, 9)])
    child = pd.DataFrame({"lat": child_ll[:, 0], "lon": child_ll[:, 1]})
    cx, cy = to_25832(child["lon"].values, child["lat"].values)
    child_xy = np.c_[cx, cy]

    # Official road Lden: major roads outside agglomerations (30) + all roads
    # inside agglomerations (27) (disjoint coverage -> max).
    road_off_db = sample_lden(child_xy, ["30", "27"])
    road_off_hex = hex_energy_mean(road_off_db, n)

    # OSM minor-road proxy as a sub-threshold fallback: roads below the
    # 8200 Kfz/day mapping cutoff aren't in the official maps. Calibrated to
    # a low ceiling (busiest proxy cell ~ DB_LO + PROXY_DB_SPAN dB), so it
    # only fills quiet gaps and never out-shouts a measured autobahn.
    PROXY_DB_SPAN = 13.0
    rd = roads[roads["cls"].isin(ROAD_W)].copy()
    rd["w"] = rd["cls"].map(ROAD_W)
    prox_child = proximity_decay(child_xy, rd)
    prox_hex = pd.Series(prox_child).groupby(np.arange(len(prox_child)) // 7) \
        .mean().values
    p95 = np.percentile(prox_hex[prox_hex > 0], 95) if (prox_hex > 0).any() else 1.0
    prox_norm = clip01(prox_hex / p95)
    proxy_db = np.where(prox_norm > 0, 50.0 + PROXY_DB_SPAN * prox_norm, np.nan)

    road_db = energy_sum(road_off_hex, proxy_db)
    road_penalty = anchor(road_db)

    # Official aircraft Lden: major airports (16) + agglomeration aircraft (13).
    air_db = hex_energy_mean(sample_lden(child_xy, ["16", "13"]), n)
    airport_penalty = anchor(air_db)

    # EBA federal-rail Lden, sampled per child then energy-meaned to hex.
    rail_child = rail_db_at(child)
    if rail_child is not None:
        rail_db = hex_energy_mean(rail_child, n)
        rail_src = "eba_wfs"
    else:
        print("WARNING: EBA rail unavailable -> rail term skipped")
        rail_db = np.full(n, np.nan)
        rail_src = "missing"
    rail_penalty = anchor(rail_db)

    # diffuse urban noise: dense population means traffic/activity on every
    # street, which major-road proximity misses entirely. ~10k residents/hex
    # (dense Munich block) saturates; exponent tapers mid-range towns.
    demo = LAYERS / "demographics.parquet"
    if demo.exists():
        pop = grid[["h3"]].merge(pd.read_parquet(demo)[["h3", "population"]],
                                 on="h3", how="left")["population"].fillna(0)
        idx = {c: i for i, c in enumerate(grid["h3"])}
        pv = pop.values.astype(float)
        smooth = pv.copy()
        cnt = np.ones(len(pv))
        for i, c in enumerate(grid["h3"]):
            for nb in h3.grid_ring(c, 1):
                j = idx.get(nb)
                if j is not None:
                    smooth[i] += pv[j]
                    cnt[i] += 1
        smooth /= cnt
        urban_penalty = clip01((smooth / 10_000) ** 0.7)
    else:
        print("NOTE: demographics layer missing, urban noise term skipped")
        urban_penalty = np.zeros(n)

    # Combine: energy-sum the three measured dB sources, anchor once, then
    # take the louder of (transport, diffuse-urban). Urban is a proxy, not
    # dB, so it stays outside the acoustic sum.
    transport_penalty = anchor(energy_sum(road_db, rail_db, air_db))
    noise_penalty = np.maximum(transport_penalty, urban_penalty)

    out = pd.DataFrame({
        "h3": grid["h3"],
        "road_penalty": road_penalty,
        "rail_penalty": rail_penalty,
        "urban_penalty": urban_penalty,
        "airport_penalty": airport_penalty,
        "noise_penalty": noise_penalty,
    })
    LAYERS.mkdir(parents=True, exist_ok=True)
    out.to_parquet(LAYERS / "noise.parquet", index=False)
    print(f"noise (rail source: {rail_src}):")
    print(out.drop(columns="h3").describe().round(3).to_string())


if __name__ == "__main__":
    main()
