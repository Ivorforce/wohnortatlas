"""Select commute-target CENTER hexes for the configurable Lebensmittelpunkt.

Centers are the population peaks themselves, found by greedy non-max suppression
on a lightly-smoothed (SMOOTH_KM) residential-population surface: repeatedly take
the densest remaining neighbourhood, fix a center on its raw-population peak hex,
suppress everything within KERNEL_KM ("same destination", ~15 min e-bike), repeat
while a neighbourhood still holds >= POP_SEED people. This puts a dot on every
town's actual core — one for a small town, several across a big city (they fall
out of the fixed suppression radius) — and never on the empty land between
clusters. (The earlier catchment-coverage surface peaked on that empty land and,
worse, gated candidacy on catchment size, so it tagged tiny well-embedded places
while missing isolated real towns like Dorfen — high population, low catchment.)

The catchment population O(h) is no longer used for *placement*, only for the
opportunity *weight* read at each chosen center: O_any (saturating) and O_gross
(Großstadt smoothstep) drive the two aggregate reach tiers. Per-center
reachability is routed in 04c; aggregates + per-city reach chunks derive there
and in 22_build_web.

Emits data/layers/centers.parquet: one row per center
(id, name, district, kind, h3, lat, lon, catchment_pop, o_any, o_gross), and
prints the largest selected centers for a sanity check.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from wohnen.config import LAYERS, INTERIM, DATA, BBOX
from wohnen.cityness import C_HALF, GROSS_HI, GROSS_LO, smoothstep
from wohnen.labels import labels_for_points

RAW = DATA / "raw"

# --- tunable knobs (calibrated on the Munich bbox) ---
KERNEL_KM = 4.5         # NMS suppression radius = min spacing between centers ("same destination", ~15 min e-bike)
SMOOTH_KM = 1.2         # neighbourhood radius the population surface is summed over before peak-finding
POP_SEED = 3_500        # min neighbourhood population (within SMOOTH_KM) to seed a center — a town core, not a hamlet
# C_HALF / GROSS_LO / GROSS_HI / smoothstep: the cityness O definition (wohnen/cityness.py),
# shared with 04f's per-cell native-value term.


def to_km(lat, lon):
    """Equirectangular projection to km around the bbox center — fine at this scale."""
    lat0 = np.deg2rad((BBOX[1] + BBOX[3]) / 2)
    x = np.deg2rad(lon) * np.cos(lat0) * 6371.0
    y = np.deg2rad(lat) * 6371.0
    return np.column_stack([x, y])


def select_centers(df: pd.DataFrame) -> list[int]:
    """Place centers on population peaks by greedy non-max suppression; returns
    row indices of the selected centers.

    Peaks are found on a lightly-smoothed (SMOOTH_KM) population surface so a
    town's *mass* wins over a lone dense outlier block, then each dot snaps to the
    raw-population peak hex of that neighbourhood (so it sits on real built-up
    land, never the smoothed centroid). Suppression at KERNEL_KM enforces minimum
    spacing — a small town yields one center, a big city several."""
    pop = df["population"].fillna(0).values
    xy = to_km(df["lat"].values, df["lon"].values)
    tree = cKDTree(xy)

    # smoothed surface: total population within SMOOTH_KM of each hex
    neigh = tree.query_ball_tree(tree, SMOOTH_KM)
    pop_s = np.array([pop[n].sum() for n in neigh])

    selected = []
    suppressed = np.zeros(len(pop), bool)
    for i in np.argsort(-pop_s):
        if pop_s[i] < POP_SEED:
            break          # sorted descending → every remaining neighbourhood is too small
        if suppressed[i]:
            continue
        local = np.array(neigh[i])
        center = int(local[np.argmax(pop[local])])   # the dot sits on the densest actual hex
        selected.append(center)
        suppressed[tree.query_ball_point(xy[center], KERNEL_KM)] = True
    return selected


def gemeinde_names(lat, lon) -> np.ndarray:
    """Municipality (Gemeinde) name per point via VG250 — the city you'd search for
    (labels_for_points gives neighbourhood-level names, wrong for a city target).
    Excludes gemeindefreie Gebiete (forests/lakes — e.g. a Munich-edge center sitting
    in 'Perlacher Forst') and snaps to the NEAREST real municipality, so a center in
    such a polygon takes the adjacent city's name."""
    zf = zipfile.ZipFile(RAW / "vg250.zip")
    shp = next(n for n in zf.namelist() if n.endswith("VG250_GEM.shp"))
    vg = gpd.read_file(f"zip://{RAW / 'vg250.zip'}!{shp}").to_crs(4326)
    vg = vg[(vg["GF"] == 4) & (vg["BEZ"] != "Gemeindefreies Gebiet")]
    vg = vg[["GEN", "geometry"]].cx[BBOX[0]:BBOX[2], BBOX[1]:BBOX[3]].to_crs(25832)
    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(lon, lat), crs=4326).to_crs(25832)
    j = gpd.sjoin_nearest(pts, vg, how="left")
    j = j[~j.index.duplicated(keep="first")]  # equidistant points can match 2 polys
    return j["GEN"].reindex(range(len(lat))).values


def main():
    grid = pd.read_parquet(INTERIM / "grid.parquet")
    # only the headcount-grid anchors, from population.parquet (07a) — NOT demographics.parquet,
    # so editing the age/life-stage fields never invalidates center placement or the routing.
    pop = pd.read_parquet(LAYERS / "population.parquet")[["h3", "catchment_pop", "population"]]
    df = grid.merge(pop, on="h3", how="left")

    idx = select_centers(df)
    sel = df.iloc[idx].copy().reset_index(drop=True)
    cat = sel["catchment_pop"].fillna(0).values
    sel["o_any"] = cat / (cat + C_HALF)
    sel["o_gross"] = smoothstep(cat, GROSS_LO, GROSS_HI)
    sel["name"] = gemeinde_names(sel["lat"].values, sel["lon"].values)
    sel["district"] = labels_for_points(sel["lat"].values, sel["lon"].values)
    # fall back to the neighbourhood label if a center sits outside any Gemeinde polygon
    sel["name"] = sel["name"].fillna(sel["district"])
    sel["kind"] = "city"
    # center id = full h3 (unique + stable across rebuilds; centers are internal —
    # the web targets cities, keyed by name-slug, grouped (by name + spatial cluster) in 04f)
    sel["id"] = sel["h3"]

    out = sel[["id", "name", "district", "kind", "h3", "lat", "lon",
               "catchment_pop", "o_any", "o_gross"]]
    LAYERS.mkdir(parents=True, exist_ok=True)
    out.to_parquet(LAYERS / "centers.parquet", index=False)
    print(f"{len(out)} centers, {out['name'].nunique()} distinct cities "
          f"-> {LAYERS / 'centers.parquet'}")
    show = out.sort_values("catchment_pop", ascending=False).head(40)
    print(show[["name", "district", "catchment_pop", "o_any", "o_gross"]]
          .round(2).to_string(index=False))


if __name__ == "__main__":
    main()
