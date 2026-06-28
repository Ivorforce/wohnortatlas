"""School and family-infrastructure layers: nearest-POI times via KDTree."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h3
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from wohnen.config import (BIKE_KMH, CELL_RMS_M, DETOUR_FACTOR, INTERIM,
                           LAYERS, WALK_KMH)
from wohnen.h3util import disk_weighted_sum, points_to_cells, utm32_transformer

_T = utm32_transformer()


def cell_samples(cells, gx, gy) -> np.ndarray:
    """Per H3 hexagon, 6 in-cell sample points: the boundary vertices scaled so
    their RMS offset from the centre == CELL_RMS_M. Averaging the nearest-POI
    distance over these (instead of one centre query + a hypot(·, CELL_RMS)
    de-bias) captures that with denser supply a resident's NEAREST option sits
    closer than the single nearest-to-centre — different residents have different
    nearest shops. So walking becomes viable in well-supplied areas, while
    sparse/rural cells (one far option) are essentially unchanged. It's the
    numerical evaluation of E_cell[dist to nearest], which has no closed form
    (depends on the amenities' Voronoi split of the cell). Returns (N, 6, 2) in
    EPSG:25832; the 6 symmetric points keep the same average offset as before."""
    n = len(cells)
    lat = np.empty((n, 6))
    lng = np.empty((n, 6))
    for i, c in enumerate(cells):
        b = list(h3.cell_to_boundary(c))[:6]
        while len(b) < 6:  # pentagons (none in-bbox) — pad defensively
            b.append(b[-1])
        for j in range(6):
            lat[i, j], lng[i, j] = b[j]
    vx, vy = _T.transform(lng.ravel(), lat.ravel())
    vx = np.asarray(vx).reshape(n, 6)
    vy = np.asarray(vy).reshape(n, 6)
    ox, oy = vx - gx[:, None], vy - gy[:, None]
    s = CELL_RMS_M / np.sqrt((ox**2 + oy**2).mean(axis=1, keepdims=True))
    return np.stack([gx[:, None] + ox * s, gy[:, None] + oy * s], axis=2)


def _mean_nearest_m(tree, samples) -> np.ndarray:
    """Mean over each cell's sample points of the distance to the nearest POI."""
    d, _ = tree.query(samples.reshape(-1, 2))
    return d.reshape(samples.shape[0], samples.shape[1]).mean(axis=1)


def nearest_min(samples, pois: pd.DataFrame, speed_kmh: float) -> np.ndarray:
    if not len(pois):
        return np.full(len(samples), np.inf)
    px, py = _T.transform(pois["lon"].values, pois["lat"].values)
    tree = cKDTree(np.c_[px, py])
    return _mean_nearest_m(tree, samples) * DETOUR_FACTOR / 1000 / speed_kmh * 60


def main():
    grid = pd.read_parquet(INTERIM / "grid.parquet")
    pois = pd.read_parquet(INTERIM / "pois.parquet")
    gx, gy = _T.transform(grid["lon"].values, grid["lat"].values)
    # 6 in-cell sample points per cell (density-aware nearest, see cell_samples)
    grid_samples = cell_samples(grid["h3"].tolist(), gx, gy)

    def sel(cat, sub=None):
        m = pois["category"] == cat
        if sub:
            m &= pois["subcategory"] == sub
        return pois[m]

    # Per-track school points come from 03c_schools (OSM locations authoritatively
    # typed from JedeSchule, with OSM name-typing as fallback) — NOT from the OSM
    # subcategory here. A school is a member of every track it delivers (combined
    # forms join several), so the boolean columns overlap; the web's "Für Jugend"
    # diversity then credits each track once.
    sp = pd.read_parquet(INTERIM / "schools_points.parquet")
    grund = sp[sp["grund"]]
    gym = sp[sp["gym"]]
    real = sp[sp["real"]]
    mittel = sp[sp["mittel"]]
    t_grund = nearest_min(grid_samples, grund, BIKE_KMH)
    t_gym = nearest_min(grid_samples, gym, BIKE_KMH)
    t_real = nearest_min(grid_samples, real, BIKE_KMH)
    t_mittel = nearest_min(grid_samples, mittel, BIKE_KMH)

    doctors = pois[(pois["category"] == "family")
                   & pois["subcategory"].isin(["doctor", "pediatrician"])]
    dentists = pois[(pois["category"] == "family")
                    & (pois["subcategory"] == "dentist")]

    # Two food tiers: VOLLSORTIMENT = the full weekly shop (supermarket +
    # small grocery + rural general store); FRISCHE = daily fresh-food gap-fillers
    # (Bäcker/Metzger/Obst-Gemüse/Hofladen/Bioladen), which the web scores at a cap
    # below a real grocery. Kiosk is NOT food (split out in 03 — newspaper/cigarette
    # booths don't belong in the convenience tier).
    vollsort = pois[(pois["category"] == "family")
                    & pois["subcategory"].isin(["supermarket", "convenience", "general"])]
    frische = pois[(pois["category"] == "family")
                   & pois["subcategory"].isin(
                       ["bakery", "butcher", "greengrocer", "farm", "health_food"])]

    # NB: the web "In der Nähe" (s_access) layer is computed CLIENT-SIDE from the
    # shipped t_*_min nearest-times (see index.html); no baked acc_* scores remain.

    # supply density of the locally-rationed child facilities (Kita +
    # Grundschule), same 3 km kernel as catchment_pop, so 20 can form a
    # per-capita supply ratio. These scale ~1:1 with population (corr 0.93,
    # facilities-per-1000 flat rural→core), so a raw-headcount crowding discount
    # would just penalize dense areas; the ratio flags genuine under-supply
    # (fast-growth suburbs where building lagged) instead. Cf. leisure.
    kg = pd.concat([sel("family", "kita"), grund])
    kg_counts = pd.Series(points_to_cells(kg["lat"], kg["lon"])).value_counts()
    kita_supply = disk_weighted_sum(kg_counts, grid["h3"].tolist(),
                                    k=4, scale_km=3.0)

    # nearest-times SHIPPED for the web "In der Nähe" client-side scoring.
    # Dentist = Nahversorgung AND-component, bike ref like Arzt.
    t_voll = nearest_min(grid_samples, vollsort, WALK_KMH)
    t_frische = nearest_min(grid_samples, frische, WALK_KMH)
    t_pharm = nearest_min(grid_samples, sel("family", "pharmacy"), WALK_KMH)
    t_doc = nearest_min(grid_samples, doctors, BIKE_KMH)
    t_kita = nearest_min(grid_samples, sel("family", "kita"), WALK_KMH)
    t_dentist = nearest_min(grid_samples, dentists, BIKE_KMH)

    LAYERS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "h3": grid["h3"], "t_grundschule_min": t_grund, "t_gymnasium_min": t_gym,
        "t_realschule_min": t_real, "t_mittelschule_min": t_mittel,
    }).to_parquet(LAYERS / "schools.parquet", index=False)
    pd.DataFrame({
        "h3": grid["h3"], "t_vollsort_min": t_voll, "t_frische_min": t_frische,
        "t_pharmacy_min": t_pharm, "t_doctor_min": t_doc, "t_kita_min": t_kita,
        "t_dentist_min": t_dentist,
        "kita_supply": kita_supply.values,
    }).to_parquet(LAYERS / "family.parquet", index=False)


if __name__ == "__main__":
    main()
