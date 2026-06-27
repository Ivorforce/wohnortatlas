"""Nature layer: can I get to a tranquil place by bike on the weekend/evening?
(NOT swim/sport access — that's the Freizeit layer.) The honest question is "is there A
good place I can reach", so the reach operator is a distance-discounted MAX
("dilate"), not a coverage mean — "how green are my surroundings" is s_green's job,
and a mean here would just duplicate it.

Each tile's nature-ness is computed ONCE, map-wide, then spread by the dilate so a
tile inherits the BEST reachable source, discounted by distance. Quality is a
property of the DESTINATION, so hilliness and crowding fold into the source tile
(not the home tile): a hilly forest is a better outing for everyone who can reach
it; an overrun park is a worse one.

Per SOURCE tile, per type, quality q ∈ [0,1]:
  q = sat(smoothed_share × mask) × relief_amp[land] × crowd
  - share: the Gaussian-smoothed (09, σ≈0.5km) share — already mass-consolidated,
    so a lone green sliver can't masquerade as a real destination under the max.
  - sat = cᵃ/(cᵃ+kᵃ): steep (a=2.5) for forest/grass/crop/heath, concave (a=1)
    for the sparse-but-valuable lake/wetland so a small-but-near body counts fairly.
  - mask = (1−no_access_share) × (1−NOISE_BITE·transport_noise): green you can't
    enter (military/fenced) or that an autobahn runs through is worth less.
    Transport noise only (road/rail/air); the diffuse-urban term overlaps crowd.
  - relief_amp = 1 + A_RELIEF·relief_score, LAND types only (a hilly forest is
    worth more; a flat lake/river is just as tranquil). relief = per-hex elevation
    std (GLO-30), disk-smoothed. No standalone bump — relief only amplifies nature.
  - crowd = 1 − CROWD_MAX·L/(L+CROWD_HALF), L = the population that can reach this
    destination for an outing = disk_weighted_SUM of population over the SAME reach
    kernel as the dilate (a load is additive → sum, not max). Wider than the old
    residential catchment, so it captures more day-tripper bleed — but still
    residential, so true S-Bahn day-tripper pressure remains under-counted.

  forest, grass, crop, wetland(class 90), heath (OSM natural=heath polygons from 09 —
  WorldCover calls Calluna heath grassland, so its share is reclaimed OUT of grass here
  to avoid double-counting the same ground), lake(water cells
  ≥0.04 RAW share so a small Weiher still registers, river-adjacent excluded;
  saturated quality scaled by water_recr_q from 09 — engineered water (reservoir/
  basin/Klärbecken) is demoted as it's not a swim/picnic outing, swim-tagged kept
  full, so the Ismaninger Speichersee reads partial, not a premier lake).
  river  — best reachable GREEN riverbank, dilated like the area types: per river
           cell the bank quality = green cover (tree+grass+park+wetland) minus its
           built-up share, so a meadow/forest river reads ~1 and a canalized concrete
           channel ~0. Crowd- AND noise-discounted at the source (a loud, overrun
           urban riverbank — the Isar through Munich — is a lesser outing).
  stream — same bank model for OSM waterway=stream (Bäche): a smaller draw.
  sights — best reachable notable sight (viewpoint/peak/waterfall; urban
           viewpoints builtup>0.2 dropped), crowd-discounted, dilated like the rest.

River/stream are NOT best-eligible types: green carries the bulk, water only sweetens.
Each spends a capped slice of the REMAINING noisy-OR headroom (RIVER_BONUS/STREAM_BONUS,
cf. s_green's WATER_STANDALONE), scaled by its reachable green-bank quality — so a leafy
river lifts an already-green cell but can't carry a barren one, and a concrete channel
adds ~nothing. Swimming is the Freizeit layer's job; here water is aesthetics.

Type ceilings W (best-eligible): lake .90, forest .82, heath .80, wetland .65,
sights .55, grass .48, crop .15. crop barely counts.

  R_t   = disk_weighted_max(q_t) — best reachable source, distance-discounted.
  base  = noisy-OR over W[t]·R_t: best type at full weight, the rest tempered by
          BETA — monotype-excellent stays high (no diversity gate) and variety
          lifts toward 1 with diminishing returns (diversity is intrinsic).
  score = clip((base-NATURE_FLOOR)/(NATURE_TOP-NATURE_FLOOR))**GAMMA: band-rescale +
          mild curve. The dilate reaches *some* green everywhere (hard ~0.15 floor) and
          the type ceilings W cap the best at ~0.91 — both fatal for a geo-mean
          PREFERENCE layer (it could never reach 0 OR 100). Rescaling [FLOOR,TOP]→[0,1]
          lets the worst farmland/urban cells reach 0 and premier nature saturate ~100.
Calibration anchors live in scripts/90_validate.py (Herrsching/Schwabing/Geretsried).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h3
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from scipy.spatial import cKDTree

from wohnen.config import (BBOX, BIKE_KMH, CELL_RMS_M, DETOUR_FACTOR, INTERIM,
                           LAYERS)
from wohnen.h3util import disk_weighted_max, disk_weighted_sum
from wohnen.norm import clip01

DEM_S3 = "https://copernicus-dem-30m.s3.amazonaws.com"
DEM_CROP = INTERIM / "dem_crop.tif"
DEM_RES = 1 / 1200  # 3 arcsec ≈ 90 m at this latitude? GLO-30 is 1/3600°; we resample
_T = Transformer.from_crs(4326, 25832, always_xy=True)

# --- s_nature scoring (anchors in scripts/90_validate.py) ---
RK, RSCALE = 8, 4.0  # reach DILATE kernel: best reachable nature over grid_disk(8),
#                      exp(-d / 4 km) — gentle, "a comfortable bike outing" (a great
#                      lake 15 bike-min away is still a fine weekend destination)
W = {  # monotype ceiling: score if your reach were purely this best-eligible type.
    # river/stream are NOT here — they are capped bonuses (RIVER_BONUS/STREAM_BONUS),
    # not best-eligible, so the surroundings always decide a riverside hex.
    # heath sits at forest tier: an open Calluna heath (the Lüneburger Heide) is a
    # premier tranquil-nature outing, not a sub-meadow afterthought — the old 0.42
    # (below grass) predated heath ever having real data (09 now sources it from OSM).
    "lake": 0.90, "forest": 0.82, "heath": 0.80, "wetland": 0.65, "sights": 0.55,
    "grass": 0.48, "crop": 0.15,
}
KSAT = {  # saturation midpoint per type: the per-source SMOOTHED share where p hits 0.5
    # forest/grass raised (was .50/.38): the dilate is a MAX, so a small park's
    # dense core cell used to read like a big forest — a higher midpoint demands a
    # genuinely large contiguous share, so Nymphenburg/urban parks count less while
    # rural forests (share→0.9) stay near 1. lake/wetland keep low midpoints (concave
    # curve) so a small near Weiher still counts.
    "forest": 0.57, "grass": 0.45, "crop": 0.50, "lake": 0.045,
    "wetland": 0.04, "heath": 0.30,
}
# per-type curve steepness c^a/(c^a+k^a): area types use a steep S-curve (a=2.5)
# to spread their compressed-but-substantial share; lake/wetland are sparse-
# but-valuable, so a concave curve (a=1) credits a small-but-near body fairly
# instead of crushing it to ~0.
SATEXP = {"forest": 2.5, "grass": 2.5, "crop": 2.5, "heath": 2.5,
          "lake": 1.0, "wetland": 1.0}
LAKE_THR = 0.04    # detect small lakes (a ~0.03 km² Weiher), not just big ones
BETA = 0.35        # diversity temper on the non-best types (lower → leans toward
#                    the "is there ONE good place" max; β=0.45 let ubiquitous
#                    low-value types — pasture, a Bach, a monotone field — stack a
#                    ~0.15 floor onto an always-reachable weak forest, AND squashed
#                    very different cells onto a ~0.68 top plateau)
# The noisy-OR base is compressed at BOTH ends. FLOOR: the wide bike-dilate reaches
# *some* green almost everywhere, so the worst farmland/urban cell still scores ~0.15 —
# nothing reached 0. TOP: the type ceilings W cap at 0.90 (lake), so the best lake/forest
# reach maxed at base ~0.91 — nothing reached 1. As a PREFERENCE in the geo-mean composite
# both ends matter: a place with no nature outing must read ~0 to bite, and a premier
# lake/forest spot should saturate ~100. So band-rescale base from [NATURE_FLOOR, NATURE_TOP]
# onto [0,1], then curve. "Best reachable nature this weak → 0; a great multi-type reach
# → 100." Monotone, so the good distribution and the relative anchors
# (Herrsching>Schwabing, Geretsried>=Schwabing) survive; the absolute bands in
# 90_validate.py track it. NATURE_TOP ~ p99.5 of base so the genuine top ~0.5% saturates.
NATURE_FLOOR, NATURE_TOP = 0.20, 0.89
GAMMA = 1.4        # final contrast: AFTER the rescale, pull the (now thin) bottom band
#                    further toward 0 so the worst rural-Augsburg/Munich-centre cells
#                    genuinely zero out, without reordering
A_RELIEF = 0.55    # hilly-natural amplifier on the SOURCE (a hilly forest is worth more)
SIGHT_K = 1.0      # per-cell sight count at half-intensity (one notable sight registers)
# river/stream are BONUS terms (not best-eligible): each spends up to this slice of
# the remaining noisy-OR headroom, scaled by reachable green-bank quality — so green
# carries the bulk and water only sweetens (cf. s_green's WATER_STANDALONE=0.20).
RIVER_BONUS, STREAM_BONUS = 0.30, 0.12
# bank quality per river cell = green cover / BANK_FULL − BANK_BUILT·builtup, clipped:
# ~50% green banks already reads as a full green riverbank; built-up directly negates
# (a concrete channel is not an outing, however quiet the neighbourhood).
BANK_FULL, BANK_BUILT = 0.50, 1.0
# crowd LOAD = pop SUM over the reach kernel (not catchment_wide); CROWD_HALF
# re-seated for that magnitude. Discount bottoms at 1-MAX, never negates.
# Strong + early-biting (was .55/300k): dense Munich green (Isar/Nymphenburg/EG,
# load ~375k) is a crowded outing, not tranquil nature — the city's reach-load is
# ~30x Herrsching's (~12k), so this hits the city hard and the countryside barely.
CROWD_MAX, CROWD_HALF = 0.58, 220_000
NOISE_BITE = 0.38  # a fully-noisy reachable area (Lden>=70) keeps 1-BITE of its value
#                    (was .30): a traffic-lined riverbank/park is a louder outing


def dem_tiles():
    lats = range(int(BBOX[1]), int(BBOX[3]) + 1)
    lons = range(int(BBOX[0]), int(BBOX[2]) + 1)
    for la in lats:
        for lo in lons:
            name = f"Copernicus_DSM_COG_10_N{la:02d}_00_E{lo:03d}_00_DEM"
            yield f"/vsicurl/{DEM_S3}/{name}/{name}.tif"


def build_dem_crop():
    """Mosaic the bbox from GLO-30 tiles, downsampled ~3x to ~90 m."""
    if DEM_CROP.exists():
        return
    import rasterio.windows
    from rasterio.enums import Resampling
    from rasterio.transform import from_origin

    step = DEM_RES
    w = round((BBOX[2] - BBOX[0]) / step)
    h = round((BBOX[3] - BBOX[1]) / step)
    mosaic = np.full((h, w), np.nan, dtype=np.float32)
    for url in dem_tiles():
        try:
            with rasterio.open(url) as src:
                win = rasterio.windows.from_bounds(*BBOX[:2], *BBOX[2:],
                                                   transform=src.transform)
                win = win.intersection(
                    rasterio.windows.Window(0, 0, src.width, src.height))
                if win.width <= 0 or win.height <= 0:
                    continue
                wb = src.window_bounds(win)
                oh = round((wb[3] - wb[1]) / step)
                ow = round((wb[2] - wb[0]) / step)
                data = src.read(1, window=win, out_shape=(oh, ow),
                                resampling=Resampling.average)
                y0 = round((BBOX[3] - wb[3]) / step)
                x0 = round((wb[0] - BBOX[0]) / step)
                mosaic[y0:y0 + oh, x0:x0 + ow] = data
                print(f"  {url.rsplit('/', 1)[-1]}: {data.shape}")
        except rasterio.errors.RasterioIOError as e:
            print(f"  WARNING tile failed: {e}")
    with rasterio.open(DEM_CROP, "w", driver="GTiff", height=h, width=w,
                       count=1, dtype="float32", crs="EPSG:4326",
                       transform=from_origin(BBOX[0], BBOX[3], step, step),
                       compress="deflate") as dst:
        dst.write(mosaic, 1)
    print(f"DEM crop: {DEM_CROP} ({h}x{w})")


def relief_per_hex(grid: pd.DataFrame) -> pd.Series:
    with rasterio.open(DEM_CROP) as src:
        arr = src.read(1)
        lon0, lat1 = src.transform.c, src.transform.f
    h, w = arr.shape
    lats = lat1 - (np.arange(h) + 0.5) * DEM_RES
    lons = lon0 + (np.arange(w) + 0.5) * DEM_RES
    cells = []
    for la in lats:
        cells.append([h3.latlng_to_cell(la, lo, 8) for lo in lons])
    df = pd.DataFrame({"h3": np.array(cells).ravel(), "z": arr.ravel()})
    df = df.dropna()
    std = df.groupby("h3")["z"].std().rename("relief")

    rel = grid["h3"].map(std).fillna(0).values
    # smooth over k=1 disk: "hilly area", not just within-cell variance
    idx = {c: i for i, c in enumerate(grid["h3"])}
    sm = rel.copy()
    cnt = np.ones(len(rel))
    for i, c in enumerate(grid["h3"]):
        for nb in h3.grid_ring(c, 1):
            j = idx.get(nb)
            if j is not None:
                sm[i] += rel[j]
                cnt[i] += 1
    return pd.Series(sm / cnt, index=grid.index)


def main():
    grid = pd.read_parquet(INTERIM / "grid.parquet")
    build_dem_crop()
    relief = relief_per_hex(grid)
    relief_score = clip01((relief / 50.0) ** 0.7)

    pois = pd.read_parquet(INTERIM / "pois.parquet")
    sights = pois[pois["category"] == "sight"]
    if len(sights):
        from wohnen.h3util import points_to_cells
        s = sights.copy()
        s["h3"] = points_to_cells(s["lat"], s["lon"])
        # urban viewpoints (towers, monuments) are not nature outings —
        # only count viewpoints outside built-up cells; peaks/waterfalls always
        green_lk = pd.read_parquet(LAYERS / "greenness.parquet").set_index("h3")
        built = s["h3"].map(green_lk["builtup_share"]).fillna(0)
        s = s[(s["subcategory"] != "viewpoint") | (built < 0.2)]
        counts = s.groupby("h3").size().astype(float)
        # per-cell intensity (saturating: one notable sight already registers);
        # spread into "best reachable sight" by the dilate-max below, like areas.
        gc = grid["h3"].map(counts).fillna(0.0).values
        sight_intensity = pd.Series(gc / (gc + SIGHT_K), index=grid["h3"])
    else:
        print("WARNING: no sights extracted (re-run 03_pois on refiltered PBF)")
        sight_intensity = pd.Series(np.zeros(len(grid)), index=grid["h3"])

    green = pd.read_parquet(LAYERS / "greenness.parquet")
    g = grid[["h3"]].merge(green, on="h3", how="left")
    gx, gy = _T.transform(grid["lon"].values, grid["lat"].values)
    cells_list = grid["h3"].tolist()

    def bike_min_to(lats, lons) -> np.ndarray:
        if not len(lats):
            return np.full(len(grid), 120.0)
        lx, ly = _T.transform(lons, lats)
        d, _i = cKDTree(np.c_[lx, ly]).query(np.c_[gx, gy])
        # expected distance for a uniform in-cell resident, not the center
        return np.hypot(d, CELL_RMS_M) * DETOUR_FACTOR / 1000 / BIKE_KMH * 60

    def cell_latlons(cells):
        ll = [h3.cell_to_latlng(c) for c in cells]
        return [p[0] for p in ll], [p[1] for p in ll]

    # smoothed shares (09 Gaussian σ0.5) are the per-source "mass" field — already
    # consolidated, so a lone sliver can't masquerade as a real destination under
    # the max. Lake DETECTION still uses the RAW water share (sharp, so a small
    # Weiher isn't smeared below LAKE_THR); its mass uses the smoothed share.
    w_raw = "water_share_raw" if "water_share_raw" in g.columns else "water_share"
    t_share, w_share = "tree_share", "water_share"

    # rivers/streams from OSM waterway lines; exclude river water from lake mass
    roads = pd.read_parquet(INTERIM / "roads.parquet")
    rivers = roads[roads["cls"] == "river"]
    streams = roads[roads["cls"] == "stream"]
    t_river = bike_min_to(rivers["lat"].values, rivers["lon"].values)  # display only

    def waterway_near(ways):
        cells = set(ways["lat"].combine(ways["lon"],
                    lambda la, lo: h3.latlng_to_cell(la, lo, 8)))
        near = set()
        for c in cells:
            near.update(h3.grid_disk(c, 1))
        return cells, near

    river_cells, river_near = waterway_near(rivers)
    _stream_cells, stream_near = waterway_near(streams)

    lake_mask = (g[w_raw] >= LAKE_THR) & ~g["h3"].isin(river_near)

    # accessibility mask: green you can't enter (military, fenced/private woods)
    # is visible but not reachable nature, so drop it from the source quality.
    acc = (1 - g["no_access_share"].fillna(0)).values

    # quiet mask: a reachable area degraded by transport noise (an autobahn
    # through a forest) is worth less. Transport sources only (road/rail/air) —
    # the diffuse-urban noise term is a population proxy that would double-count
    # with the crowd discount below. Applied per cell BEFORE the reach kernel,
    # like acc, so a noisy patch contributes less of its type to your reach.
    noise_f = LAYERS / "noise.parquet"
    if noise_f.exists():
        nz = pd.read_parquet(noise_f).set_index("h3")
        ntrans = np.maximum.reduce([
            g["h3"].map(nz["road_penalty"]).fillna(0).values,
            g["h3"].map(nz["rail_penalty"]).fillna(0).values,
            g["h3"].map(nz["airport_penalty"]).fillna(0).values,
        ])
        quiet = 1 - NOISE_BITE * ntrans
        print(f"quiet mask: median {np.median(quiet):.2f}, min {quiet.min():.2f}")
    else:
        print("NOTE: noise layer missing, nature noise discount skipped")
        quiet = np.ones(len(grid))
    mask = acc * quiet

    # crowd LOAD on a destination = the population that can reach it for an outing,
    # over the SAME kernel as the dilate ("if you can reach j, you add to j's
    # crowd"). A load is additive, so it's a SUM (not the max used for nature).
    # Wider than residential catchment → more day-tripper bleed, but still
    # residential, so true S-Bahn day-tripper pressure stays under-counted.
    demo_f = LAYERS / "demographics.parquet"
    if demo_f.exists():
        pop = g["h3"].map(pd.read_parquet(demo_f).set_index("h3")["population"]).fillna(0.0)
        load = disk_weighted_sum(pd.Series(pop.values, index=g["h3"]),
                                 cells_list, k=RK, scale_km=RSCALE).values
        crowd = 1 - CROWD_MAX * load / (load + CROWD_HALF)
        print(f"crowd discount: median {np.median(crowd):.2f}, min {crowd.min():.2f}")
    else:
        print("NOTE: demographics layer missing, nature crowd discount skipped")
        crowd = np.ones(len(grid))

    # per-SOURCE quality q ∈ [0,1]: saturated smoothed-share × mask, amplified by
    # hilliness (LAND only — a flat lake is just as tranquil), discounted by crowd.
    # Quality lives on the DESTINATION; the dilate-max below spreads the best
    # reachable one to each home tile.
    def sat(c, t):
        k, a = KSAT[t], SATEXP[t]
        return c**a / (c**a + k**a)

    relief_amp = 1 + A_RELIEF * np.asarray(relief_score, dtype=float)
    LAND = {"forest", "grass", "crop", "wetland", "heath"}

    def reach(share_vals, t, qfactor=None):
        q = sat(np.clip(share_vals * mask, 0.0, None), t)
        if t in LAND:
            q = q * relief_amp
        # post-saturation quality factor (e.g. engineered-water demotion): MUST apply
        # after sat — the lake curve is concave with a tiny midpoint (KSAT 0.045), so
        # scaling the share pre-sat barely moves a large body (0.9→0.45 share still
        # saturates to ~0.9). Scaling the saturated quality bites linearly.
        if qfactor is not None:
            q = q * np.asarray(qfactor, dtype=float)
        q = pd.Series(clip01(np.asarray(q) * crowd), index=g["h3"])
        return disk_weighted_max(q, cells_list, k=RK, scale_km=RSCALE).values

    # engineered water (reservoir/basin/Klärbecken) is real water the satellite sees
    # but not a tranquil swim/picnic outing, so demote its lake-source QUALITY by the
    # OSM-derived factor (09): the Ismaninger Speichersee keeps a partial bird-water
    # signal (~0.45), a sewage basin almost none. Swim-tagged water was never demoted.
    recr_q = g["water_recr_q"].fillna(1.0).values if "water_recr_q" in g.columns else None
    lake_vals = g[w_share].where(lake_mask, 0.0).fillna(0).values

    # per-cell GREEN-BANK quality: a river/stream is an outing only where its banks are
    # green and open, not a concrete channel. green cover minus built-up share, rescaled
    # so ~half-green banks already read full; mask+crowd discount the source exactly like
    # the area types (an inaccessible/loud/overrun bank counts less). waterway_reach then
    # spreads the best reachable green bank to each home tile via the same dilate.
    def gcol(name):
        return g[name].fillna(0).values if name in g.columns else np.zeros(len(g))
    green_cover = (gcol("tree_share") + gcol("grass_share")
                   + gcol("park_share") + gcol("wetland_share"))
    bank_green = clip01(green_cover / BANK_FULL - BANK_BUILT * gcol("builtup_share"))
    bank_q = clip01(bank_green * mask * crowd)

    def waterway_reach(near):
        present = g["h3"].isin(near).values
        q = pd.Series(np.where(present, bank_q, 0.0), index=g["h3"])
        return disk_weighted_max(q, cells_list, k=RK, scale_km=RSCALE).values

    # WorldCover labels Calluna heath as grassland(30), so heath ground sits inside
    # grass_share. Now that heath is a scored type (sourced from OSM in 09), feeding the
    # SAME ground to both grass and heath would double-count it in the noisy-OR (a fake
    # diversity bonus for one feature seen by two datasets). Reclassify it out of grass
    # here — grass keeps only genuine meadow; the heath ground scores once, as heath.
    # s_green is untouched (it uses the raw grass_share, where heath IS green ambience).
    grass_for_nature = clip01(g["grass_share"].fillna(0).values
                              - g["heath_share"].fillna(0).values)
    R = {
        "forest":  reach(g[t_share].fillna(0).values,        "forest"),
        "grass":   reach(grass_for_nature,                    "grass"),
        "crop":    reach(g["crop_share"].fillna(0).values,    "crop"),
        "lake":    reach(lake_vals, "lake", qfactor=recr_q),
        "wetland": reach(g["wetland_share"].fillna(0).values, "wetland"),
        "heath":   reach(g["heath_share"].fillna(0).values,   "heath"),
        # river/stream: best reachable GREEN bank (see bank_q), dilated like the areas.
        # Applied below as a capped BONUS, not a best-eligible type.
        "river":   waterway_reach(river_near),
        "stream":  waterway_reach(stream_near),
        # sights: best reachable notable sight, crowd-discounted at the source
        "sights":  disk_weighted_max(
            pd.Series(clip01(sight_intensity.values * crowd), index=g["h3"]),
            cells_list, k=RK, scale_km=RSCALE).values,
    }
    sight_score = R["sights"]  # the reachable-sights field is the output sub-driver

    # combine: best type at full weight, the rest tempered (noisy-OR). A single
    # excellent type already scores well (no diversity gate), and each added type
    # lifts toward 1 with diminishing returns — so DIVERSITY is intrinsic.
    types = list(W)  # best-eligible only — river/stream fold in as a bonus below
    C = np.column_stack([W[t] * R[t] for t in types])
    best = C.max(axis=1)
    rest = np.prod(1 - BETA * C, axis=1) / (1 - BETA * best)  # drop best's own factor
    base = 1 - (1 - best) * rest

    # river/stream are BONUS terms (green carries the bulk): each spends a capped slice
    # of the REMAINING headroom, scaled by its reachable green-bank quality. A leafy
    # river sweetens an already-green cell; a canalized concrete channel (bank_green~0)
    # adds ~nothing, and a barren-setting river can't carry a cell on water alone.
    water_bonus = clip01(RIVER_BONUS * R["river"] + STREAM_BONUS * R["stream"])
    base = 1 - (1 - base) * (1 - water_bonus)

    # relief and crowd are already folded into the source quality (see reach()), so the
    # noisy-OR base, re-anchored above its universal floor then mildly curved, IS the
    # score. The re-anchor is what lets the worst cells reach 0 (see NATURE_FLOOR).
    nature = clip01((base - NATURE_FLOOR) / (NATURE_TOP - NATURE_FLOOR)) ** GAMMA

    # display-only nearest distances
    t_lake = bike_min_to(*cell_latlons(g.loc[lake_mask, "h3"]))
    t_forest = bike_min_to(*cell_latlons(g.loc[g[t_share] >= 0.55, "h3"]))

    out = pd.DataFrame({
        "h3": grid["h3"],
        "relief_m": relief.values,
        "sight_score": sight_score,
        "t_lake_min": t_lake,
        "t_forest_min": t_forest,
        "t_river_min": t_river,
        "nature_score": nature,
    })
    LAYERS.mkdir(parents=True, exist_ok=True)
    out.to_parquet(LAYERS / "nature.parquet", index=False)
    print(out[["relief_m", "t_lake_min", "nature_score"]].describe().round(2).to_string())


if __name__ == "__main__":
    main()
