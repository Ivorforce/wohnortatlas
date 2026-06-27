"""Greenness layer: ESA WorldCover 2021 class shares per hex (tree/grass/water/built-up)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h3
import numpy as np
import pandas as pd
import rasterio
import rasterio.errors
import rasterio.features
import rasterio.windows
from rasterio.transform import from_origin
from shapely import STRtree
from shapely import wkt as shapely_wkt
from shapely.geometry import box

from wohnen.config import BBOX, INTERIM, LAYERS, WORLDCOVER_S3
from wohnen.h3util import RES8_STEP_KM, disk_gaussian_mean, points_to_cells

PX = 1 / 12000  # 10 m in degrees (WorldCover grid)
CLASSES = {"tree": 10, "grass": 30, "crop": 40, "builtup": 50, "water": 80}
CROP = INTERIM / "worldcover_crop.tif"
BLOCK = 10  # aggregate 10x10 px (100 m) before h3 assignment
# street/park tree count that saturates the canopy proxy to 0.5 (n/(n+N0)) —
# OSM trees, so completeness-limited; a tunable anchor, not ground truth
TREE_N0 = 150.0


def worldcover_tiles():
    """3°x3° WorldCover tiles (named by their SW corner) covering BBOX —
    N/E only, like the GLO-30 naming in 13_nature.py."""
    lon0, lat0, lon1, lat1 = BBOX
    for la in range(int(lat0 // 3) * 3, int(-(-lat1 // 3)) * 3, 3):
        for lo in range(int(lon0 // 3) * 3, int(-(-lon1 // 3)) * 3, 3):
            yield f"N{la:02d}E{lo:03d}"


def build_crop():
    if CROP.exists():
        return
    lon0, lat0, lon1, lat1 = BBOX
    w = round((lon1 - lon0) / PX)
    h = round((lat1 - lat0) / PX)
    mosaic = np.zeros((h, w), dtype=np.uint8)
    for tile in worldcover_tiles():
        url = f"/vsicurl/{WORLDCOVER_S3}/ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
        try:
            src = rasterio.open(url)
        except rasterio.errors.RasterioIOError as e:
            # ESA publishes a tile only where the 3°×3° cell contains land, so an
            # ocean-only cell in the bbox genuinely 404s — skip it. But a network
            # failure must NOT be silently skipped (that silently yields an all-
            # zero crop), so swallow only a real 404 and re-raise anything else.
            if "404" not in str(e):
                raise
            print(f"  {tile}: no tile (ocean-only), skipped")
            continue
        with src:
            win = rasterio.windows.from_bounds(lon0, lat0, lon1, lat1,
                                               transform=src.transform)
            win = win.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
            if win.width <= 0 or win.height <= 0:
                print(f"  {tile}: outside bbox, skipped")
                continue
            data = src.read(1, window=win)
            # paste into mosaic by georeference
            x_off = round((src.window_bounds(win)[0] - lon0) / PX)
            y_off = round((lat1 - src.window_bounds(win)[3]) / PX)
            mosaic[y_off:y_off + data.shape[0], x_off:x_off + data.shape[1]] = data
            print(f"  {tile}: {data.shape} pasted at ({y_off},{x_off})")
    transform = from_origin(lon0, lat1, PX, PX)
    with rasterio.open(CROP, "w", driver="GTiff", height=h, width=w, count=1,
                       dtype="uint8", crs="EPSG:4326", transform=transform,
                       compress="deflate") as dst:
        dst.write(mosaic, 1)
    print(f"crop saved: {CROP} ({h}x{w} px)")


def main():
    build_crop()
    grid = pd.read_parquet(INTERIM / "grid.parquet")

    with rasterio.open(CROP) as src:
        transform = src.transform
        lon0, lat1 = transform.c, transform.f
        H, W = src.height, src.width
    h_, w_ = H // BLOCK * BLOCK, W // BLOCK * BLOCK
    hb, wb = h_ // BLOCK, w_ // BLOCK

    # OSM polygons burned onto the same 10 m grid (all_touched, so tiny ones
    # still register). park: designated green WorldCover lumps into built-up.
    # no_access: green you can't enter (military, fenced/private woods) —
    # masked out of s_nature's reachable area in 13. STRtree-indexed so each
    # strip rasterizes only the polygons it overlaps, not all 143k every strip.
    def load_geoms(name):
        df = pd.read_parquet(INTERIM / name)
        return [shapely_wkt.loads(s) for s in df["wkt"]] if len(df) else []
    park_geoms = load_geoms("green_areas.parquet")
    noacc_geoms = load_geoms("no_access.parquet")
    park_tree = STRtree(park_geoms) if park_geoms else None
    noacc_tree = STRtree(noacc_geoms) if noacc_geoms else None
    # engineered-water polygons → per-pixel lake-source PENALTY (1 − quality);
    # block-meaned, then divided by the cell's water share in 13 to demote the
    # satellite water that is actually a reservoir/basin/Klärbecken (see 03).
    waterq_df = pd.read_parquet(INTERIM / "water_quality.parquet")
    waterq_geoms = [shapely_wkt.loads(s) for s in waterq_df["wkt"]] if len(waterq_df) else []
    waterq_pen = (1.0 - waterq_df["quality"].values).astype("float32") if len(waterq_df) else np.array([])
    waterq_tree = STRtree(waterq_geoms) if waterq_geoms else None
    print(f"green polys: {len(park_geoms)}, no-access polys: {len(noacc_geoms)}, "
          f"engineered-water polys: {len(waterq_geoms)}")

    # The full 10 m crop is ~10.6 GB; the old code held it + two rasterised
    # masks + a 105 M-row frame at once, which OOMs a 32 GB box (zeros only fit
    # by compression). Stream it in horizontal strips, reduce each 10x10 px
    # block to a 100 m cell, assign to h3, and accumulate per-hex sums/counts —
    # identical result (per-hex mean of the 100 m cell shares), bounded memory.
    # wetland/moor (90) and heath (shrubland 20 + moss/lichen 100) are extra s_nature
    # (13) classes, not folded into s_green. BARE/SPARSE (60) is deliberately EXCLUDED:
    # the class lumps quarries/gravel-pits/construction with alpine scree, and crediting
    # a gravel pit as 0.42-nature is wrong; the rare scenic-scree loss is acceptable.
    share_cols = list(CLASSES) + ["park", "no_access", "wetland", "heath", "water_pen"]
    idx_of = {c: i for i, c in enumerate(grid["h3"])}
    sums = {name: np.zeros(len(grid)) for name in share_cols}
    counts = np.zeros(len(grid))
    lons = lon0 + (np.arange(wb) + 0.5) * PX * BLOCK

    def burn(geoms, tree, win_tf, ext, sh):
        """park/no-access wet-fraction per 100 m block over a strip window."""
        if tree is None:
            return np.zeros((sh, wb), np.float32)
        gs = [geoms[i] for i in tree.query(box(*ext))]
        if not gs:
            return np.zeros((sh, wb), np.float32)
        a = rasterio.features.rasterize(
            ((g, 1) for g in gs), out_shape=(sh * BLOCK, w_), transform=win_tf,
            fill=0, all_touched=True, dtype="uint8")
        return a.reshape(sh, BLOCK, wb, BLOCK).mean(axis=(1, 3))

    def burn_vals(geoms, vals, tree, win_tf, ext, sh):
        """Per-100m-block MEAN of a painted float value (e.g. water penalty) — like
        burn() but rasterizing each polygon's own value instead of a 0/1 mask."""
        if tree is None:
            return np.zeros((sh, wb), np.float32)
        idxs = tree.query(box(*ext))
        if not len(idxs):
            return np.zeros((sh, wb), np.float32)
        a = rasterio.features.rasterize(
            ((geoms[i], float(vals[i])) for i in idxs), out_shape=(sh * BLOCK, w_),
            transform=win_tf, fill=0.0, all_touched=True, dtype="float32")
        return a.reshape(sh, BLOCK, wb, BLOCK).mean(axis=(1, 3))

    STRIP = 256  # block-rows per strip (~2560 px tall window; <1 GB resident)
    print(f"streaming {hb}x{wb} 100m cells in strips of {STRIP} block-rows ...")
    with rasterio.open(CROP) as src:
        for rb0 in range(0, hb, STRIP):
            rb1 = min(rb0 + STRIP, hb)
            sh = rb1 - rb0
            win = rasterio.windows.Window(0, rb0 * BLOCK, w_, sh * BLOCK)
            sub = src.read(1, window=win)
            win_tf = src.window_transform(win)
            ext = (lon0, lat1 - rb1 * BLOCK * PX, lon0 + w_ * PX,
                   lat1 - rb0 * BLOCK * PX)
            blocks = sub.reshape(sh, BLOCK, wb, BLOCK)

            slats = lat1 - (np.arange(rb0, rb1) + 0.5) * PX * BLOCK
            scell = np.fromiter(
                (idx_of.get(h3.latlng_to_cell(la, lo, 8), -1)
                 for la in slats for lo in lons), dtype=np.int64, count=sh * wb)
            ok = scell >= 0
            si = scell[ok]
            np.add.at(counts, si, 1.0)
            for name, code in CLASSES.items():
                np.add.at(sums[name], si,
                          (blocks == code).mean(axis=(1, 3)).ravel()[ok])
            np.add.at(sums["wetland"], si,
                      (blocks == 90).mean(axis=(1, 3)).ravel()[ok])
            np.add.at(sums["heath"], si,
                      np.isin(blocks, (20, 100)).mean(axis=(1, 3)).ravel()[ok])
            np.add.at(sums["park"], si,
                      burn(park_geoms, park_tree, win_tf, ext, sh).ravel()[ok])
            np.add.at(sums["no_access"], si,
                      burn(noacc_geoms, noacc_tree, win_tf, ext, sh).ravel()[ok])
            np.add.at(sums["water_pen"], si,
                      burn_vals(waterq_geoms, waterq_pen, waterq_tree,
                                win_tf, ext, sh).ravel()[ok])

    cnt = np.where(counts > 0, counts, np.nan)
    shares = pd.DataFrame({name: sums[name] / cnt for name in share_cols},
                          index=pd.Index(grid["h3"], name="h3"))

    out = grid[["h3"]].copy()
    for name in CLASSES:
        out[f"{name}_share"] = out["h3"].map(shares[name]).fillna(0)
    out["park_share"] = out["h3"].map(shares["park"]).fillna(0)
    out["wetland_share"] = out["h3"].map(shares["wetland"]).fillna(0)
    out["heath_share"] = out["h3"].map(shares["heath"]).fillna(0)
    # fraction of the hex you can't enter (military/fenced) — a mask for 13,
    # not green ambience, so kept raw (not in the smoothing batch below)
    out["no_access_share"] = out["h3"].map(shares["no_access"]).fillna(0)

    # street/park tree density → saturating canopy proxy (OSM natural=tree
    # nodes + densified tree_row lines); the urban green WorldCover can't see
    trees = pd.read_parquet(INTERIM / "trees.parquet")
    if len(trees):
        tcount = pd.Series(points_to_cells(trees["lat"].values,
                                           trees["lon"].values)).value_counts()
        n = out["h3"].map(tcount).fillna(0).values.astype(float)
    else:
        n = np.zeros(len(out))
    out["tree_canopy"] = n / (n + TREE_N0)
    nz = n[n > 0]
    print(f"trees: {int(n.sum())} in {len(nz)} hexes; "
          f"per-hex p50/p90/p99 = {np.percentile(nz, [50, 90, 99]).round().tolist() if len(nz) else []}")

    # waterway ambience: a river/Bach you pass adds to the "grünes Viertel" feel
    # (commute scenery). WorldCover sees big-river water but misses narrow
    # rivers/streams, so credit OSM-line presence (river weighted 2× stream).
    roads = pd.read_parquet(INTERIM / "roads.parquet")
    ww = roads[roads["cls"].isin(["river", "stream"])]
    if len(ww):
        wt = np.where(ww["cls"].values == "river", 1.0, 0.5)
        hits = pd.Series(wt, index=points_to_cells(ww["lat"].values, ww["lon"].values)
                         ).groupby(level=0).sum()
        nw = out["h3"].map(hits).fillna(0).values
    else:
        nw = np.zeros(len(out))
    out["waterway_share"] = np.minimum(1.0, nw / 6.0)  # full river crossing ≈1, Bach ≈0.75

    # raw (unsmoothed) shares for lake/forest detection downstream — smoothing
    # dilutes small lakes and forest patches below any sensible threshold
    out["water_share_raw"] = out["water_share"]
    out["tree_share_raw"] = out["tree_share"]

    # lake-source quality multiplier for 13: water_pen is the hex-mean of (1−quality)
    # over engineered-water pixels (so pen = frac_of_hex_water_demoted × (1−q)). Divide
    # by the cell's raw water share to get the mean quality of the cell's water:
    # q_eff = 1 − pen/water_share. Cells with no engineered water → 1.0 (natural). The
    # 0.02 floor guards the divide where OSM marks water the satellite barely sees.
    water_pen = out["h3"].map(shares["water_pen"]).fillna(0.0).values
    out["water_recr_q"] = np.clip(
        1.0 - water_pen / np.maximum(out["water_share_raw"].values, 0.02), 0.0, 1.0)

    # walkable-neighbourhood smoothing: Gaussian-weighted mean (sigma 0.5 km, k=1)
    # so your own hex dominates (~48 %) and the 6 adjacent hexes feather in —
    # you walk around but spend most time at your doorstep. A flat k=1 mean
    # over-weighted the periphery (every neighbour 0.9 km out counted equally
    # with home); ring-2 weight is negligible at this sigma. Smooth from the
    # pre-smoothing snapshot so columns don't feed back into each other.
    smooth_cols = ([f"{n}_share" for n in CLASSES]
                   + ["park_share", "tree_canopy", "wetland_share", "heath_share",
                      "waterway_share"])
    base = out.set_index("h3")
    for col in smooth_cols:
        out[col] = disk_gaussian_mean(base[col], k=1, sigma_km=0.5).values

    LAYERS.mkdir(parents=True, exist_ok=True)
    out.to_parquet(LAYERS / "greenness.parquet", index=False)
    print(out.drop(columns="h3").describe().round(3).to_string())


if __name__ == "__main__":
    main()
