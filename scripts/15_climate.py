"""Climate layer: DWD multi-annual 1 km grids (1991-2020) per hex.

rain_mm, sun_h, tmean_c, hot_days, snow_days, martonne: annual values per hex
climate_score: 0.6*sunshine + 0.4*rain comfort, where rain comfort punishes
both extremes: too wet via the mm ramp, too dry via the de Martonne aridity
index. Anchored to the GERMANY-wide range (not the bbox) so the score stays
meaningful when the map grows. Temperature/snow ship raw only — genuine
preferences, not qualities.
"""

import gzip
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

from wohnen.config import INTERIM, LAYERS, RAW
from wohnen.dl import cached_download
from wohnen.h3util import disk_median
from wohnen.norm import anchor_norm

DWD = "https://opendata.dwd.de/climate_environment/CDC/grids_germany/multi_annual"
# suffix _17 = annual aggregate (01-12 months, 13-16 seasons); the
# air_temp_mean files use underscores in the period, the others hyphens
GRIDS = {
    "rain_mm": "precipitation/grids_germany_multi_annual_precipitation_1991-2020_17.asc.gz",
    "sun_h": "sunshine_duration/grids_germany_multi_annual_sunshine_duration_1991-2020_17.asc.gz",
    "tmean_c": "air_temperature_mean/grids_germany_multi_annual_air_temp_mean_1991_2020_17.asc.gz",
    "hot_days": "hot_days/grids_germany_multi_annual_hot_days_1991-2020_17.asc.gz",
    "snow_days": "snowcover_days/grids_germany_multi_annual_snowcover_days_1991-2020_17.asc.gz",
    "martonne": "drought_index/grids_germany_multi_annual_drought_index_1991-2020_17.asc.gz",
}
# DWD grids_germany are 1 km cells in Gauss-Krüger zone 3, no CRS in the header
DWD_CRS = "EPSG:31467"
# air_temp_mean ships as 1/10 °C integers
SCALE = {"tmean_c": 0.1}

# Germany-wide absolute anchors (1991-2020 annual): sunshine ~1350 h
# (Sauerland/Erzgebirge) .. ~1950 h (Ostseeküste/Oberrhein); precipitation
# ~500 mm (mitteldeutsches Trockengebiet) .. >2000 mm (Alpenrand/Schwarzwald)
SUN_ANCHORS = dict(worst=1400.0, best=1900.0)
WET_ANCHORS = dict(worst=1800.0, best=600.0)
# too-dry side via de Martonne (DWD drought_index, P/(T+10)): classical bands
# put semi-arid <20, semi-humid 24-30, humid >30 — full comfort >=30, ramp to
# 0 at 15 (does not occur in Germany; driest ~24 around Halle -> mild penalty)
DRY_ANCHORS = dict(worst=15.0, best=30.0)


def sample_grid(name: str, fname: str, lat: np.ndarray, lon: np.ndarray,
                tf: Transformer) -> np.ndarray:
    f = cached_download(f"{DWD}/{fname}", RAW / "dwd" / Path(fname).name)
    with rasterio.open(f"/vsigzip/{f}") as src:
        arr = src.read(1).astype(float)
        arr[arr == src.nodata] = np.nan
        x, y = tf.transform(lon, lat)  # always_xy: in (lon,lat), out (E,N)
        rows, cols = rasterio.transform.rowcol(src.transform, x, y)
    rows = np.clip(rows, 0, arr.shape[0] - 1)
    cols = np.clip(cols, 0, arr.shape[1] - 1)
    vals = arr[rows, cols] * SCALE.get(name, 1.0)
    print(f"{name}: {np.isnan(vals).sum()} hexes outside grid, "
          f"range {np.nanmin(vals):.0f}..{np.nanmax(vals):.0f}")
    return vals


def main():
    grid = pd.read_parquet(INTERIM / "grid.parquet")
    lat, lon = grid["lat"].values, grid["lon"].values
    tf = Transformer.from_crs("EPSG:4326", DWD_CRS, always_xy=True)

    out = grid[["h3"]].copy()
    for name, fname in GRIDS.items():
        out[name] = sample_grid(name, fname, lat, lon, tf)
        # k=1 median smoothing: softens the 1 km Lego pattern at hex scale
        # and fills border hexes (Austria edge is NODATA in the DWD grids);
        # remaining gaps fall to the bbox median (only affects the far edge)
        s = disk_median(out.set_index("h3")[name], k=1)
        out[name] = s.fillna(float(s.median())).values

    sun = anchor_norm(out["sun_h"], **SUN_ANCHORS)
    # rain comfort punishes BOTH ends: too wet (mm ramp) and too dry
    # (de Martonne aridity, which prices hot-dry harder than cool-dry);
    # min() because a place can only fail on one side at a time
    wet_ok = anchor_norm(out["rain_mm"], **WET_ANCHORS)
    dry_ok = anchor_norm(out["martonne"], **DRY_ANCHORS)
    rain_comfort = np.minimum(wet_ok, dry_ok)
    # sunshine is the felt "nice weather" driver; rain partly overlaps with it
    out["climate_score"] = 0.6 * sun + 0.4 * rain_comfort

    LAYERS.mkdir(parents=True, exist_ok=True)
    out.to_parquet(LAYERS / "climate.parquet", index=False)
    print(out.drop(columns="h3").describe().round(2).to_string())


if __name__ == "__main__":
    main()
