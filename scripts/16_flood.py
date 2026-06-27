"""Flood layer: JRC/Copernicus river-flood hazard maps (CEMS-EFAS, Dottori 2022).

EU-wide inundation-DEPTH GeoTIFFs (metres) per return period (10..500 yr),
100 m, CC-BY-4.0. We turn depth-per-return-period into one interpretable
safety score via the standard expected-annual-damage integral, with a
normalized depth-severity in place of euro damage:

  per hex, per RP:  sev = frac_flooded · sat(mean_depth),  sat(d)=d/(d+D0)
  flood_eas        = ∫ sev d(annual prob)        (trapezoid over p = 1/RP)
  flood_score      = 1 − clip(flood_eas / EAS_ANCHOR)

Frequent + deep → high EAS → low score (the old "häufige Überflutung ≈
Ausschluss" veto, now continuous and depth-aware). Permanent water bodies are
excluded — a lake is not a flood risk. Ships interpretable tooltip values:
flood_depth_hq100 (depth at the 1-in-100 yr reference flood) and flood_rp_first
("floods ~every N years", first RP with ≥ FRAC_FIRST of the hex wet).

Sampling is memory-light: flood extent is monotone in return period, so the
RP500 raster is the union of all wet pixels — we resolve those pixels to h3
once, then read each RP only at them.

Limitations (continental model — uniform national coverage at the cost of local
precision; documented on the method page): models river basins > 150 km², so
small creeks and pluvial/Starkregen are unmodeled (read as safe); and it is
DEFENSE-AGNOSTIC — levees/embankments aren't modelled, so protected river
corridors (e.g. embanked inner-city reaches) show residual hazard. Validated on
Bavaria: the Danube/Lech/Isar/Inn/Amper floodplains are correctly captured.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h3
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import xy
from rasterio.windows import from_bounds

from wohnen.config import BBOX, INTERIM, LAYERS, RAW

JRC = RAW / "jrc_flood"
RPS = [10, 20, 30, 40, 50, 75, 100, 200, 500]   # return periods (years)
P = np.array([1.0 / rp for rp in RPS])           # annual exceedance probability

D0 = 1.0          # depth (m) at which severity = 0.5; sat(d) = d/(d+D0)
EAS_ANCHOR = 0.04  # expected-annual-severity mapping to flood_score 0 (PROVISIONAL
#                    — tune against the national distribution once built)
FRAC_FIRST = 0.10  # hex-wet fraction counting as "floods here" for flood_rp_first
HEX_AREA_M2 = 737327.0  # mean res-8 hexagon area (denominator for wet fraction)
M_PER_DEG = 111_320.0


def read_window(path):
    """Read the BBOX window of a JRC raster -> (array, transform, nodata)."""
    with rasterio.open(path) as ds:
        win = from_bounds(BBOX[0], BBOX[1], BBOX[2], BBOX[3], ds.transform)
        a = ds.read(1, window=win)
        return a, ds.window_transform(win), ds.nodata


def main():
    grid = pd.read_parquet(INTERIM / "grid.parquet")

    # Union extent = RP500 (flood extent is monotone in return period).
    d500, t, nod = read_window(JRC / "Europe_RP500_filled_depth.tif")
    wet = (d500 != nod) & (d500 > 0)

    # Exclude permanent water bodies (same grid) — a lake is not flood risk.
    wb, _, wbnod = read_window(JRC / "Europe_permanent_water_bodies.tif")
    if wb.shape == wet.shape:
        wet &= ~((wb != wbnod) & (wb > 0))

    rows, cols = np.where(wet)
    if len(rows) == 0:
        print("no flooded pixels in bbox")
        out = grid[["h3"]].copy()
        for c in ["flood_depth_hq100", "flood_rp_first"]:
            out[c] = 0.0
        out["flood_score"] = 1.0
        LAYERS.mkdir(parents=True, exist_ok=True)
        out.to_parquet(LAYERS / "flood.parquet", index=False)
        return

    lons, lats = xy(t, rows, cols)            # pixel centers (EPSG:4326)
    lons, lats = np.asarray(lons), np.asarray(lats)
    print(f"{len(rows):,} flooded pixels (≤RP500, land); assigning to h3 ...")
    cells = np.array([h3.latlng_to_cell(la, lo, 8) for la, lo in zip(lats, lons)])

    # 4326 pixel area at each pixel's latitude (for the wet-fraction denominator)
    deg = abs(t.a)
    px_area = (deg * M_PER_DEG) * (deg * M_PER_DEG * np.cos(np.radians(lats)))

    # Read each RP at the flooded pixels; accumulate per-hex wet area + Σ(area·depth).
    base = pd.DataFrame({"h3": cells, "a": px_area})
    for rp in RPS:
        d, _, nd = read_window(JRC / f"Europe_RP{rp}_filled_depth.tif")
        dep = d[rows, cols].astype(float)
        dep[(d[rows, cols] == nd) | (dep < 0)] = 0.0
        base[f"d{rp}"] = dep

    idx = grid["h3"]
    frac = pd.DataFrame(index=idx)
    mdepth = pd.DataFrame(index=idx)
    for rp in RPS:
        d = base[f"d{rp}"].values
        a = base["a"].values
        wetm = d > 0
        fa = pd.Series(np.where(wetm, a, 0.0), index=base["h3"]).groupby(level=0).sum()
        da = pd.Series(np.where(wetm, a * d, 0.0), index=base["h3"]).groupby(level=0).sum()
        frac[rp] = (fa / HEX_AREA_M2).reindex(idx).fillna(0.0).clip(0, 1).values
        mdepth[rp] = (da / fa.replace(0, np.nan)).reindex(idx).fillna(0.0).values

    # severity per RP and the expected-annual-severity integral over probability
    F = frac.values                                   # hexes × RP (wet fraction)
    DM = mdepth.values                                # hexes × RP (mean wet depth)
    sev = F * (DM / (DM + D0))                        # exposure × depth-severity
    # trapezoid of sev over annual probability p (P descending, RP10 first)
    dP = -np.diff(P)                                  # positive prob increments
    eas = np.sum(0.5 * (sev[:, :-1] + sev[:, 1:]) * dP, axis=1)
    flood_score = 1.0 - np.clip(eas / EAS_ANCHOR, 0.0, 1.0)

    # interpretable tooltip values
    first = np.full(len(idx), 0.0)
    for rp in RPS:                                    # RPs ascending: first wet
        hit = (first == 0) & (frac[rp].values >= FRAC_FIRST)
        first[hit] = rp
    # report HQ100 depth only where the cell meaningfully floods (else the mean
    # depth of a tiny sliver overstates the risk a reader sees)
    depth_hq100 = np.where(first > 0, mdepth[100].values, 0.0)

    out = grid[["h3"]].copy()
    out["flood_depth_hq100"] = depth_hq100.round(2)
    out["flood_rp_first"] = first
    out["flood_score"] = flood_score
    LAYERS.mkdir(parents=True, exist_ok=True)
    out.to_parquet(LAYERS / "flood.parquet", index=False)

    touched = (eas > 0)
    print(f"flood: {touched.mean():.1%} of hexes touched; "
          f"score min {flood_score.min():.3f} "
          f"(< 0.5: {(flood_score < 0.5).sum()} hexes, "
          f"< 0.9: {(flood_score < 0.9).sum()})")
    print(out[out["flood_rp_first"] > 0][["flood_depth_hq100", "flood_rp_first",
          "flood_score"]].describe().round(3).to_string())


if __name__ == "__main__":
    main()
