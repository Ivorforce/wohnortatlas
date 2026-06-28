"""Beach areas for the Nature layer: per-hex share of OSM natural=beach.

The sea itself reaches Nature as lake-equivalent water (WorldCover water →
13_nature's lake type), but that FAILS on the North Sea: at the satellite
snapshot the Wattenmeer tidal flats read as bare land (water_share≈0), so a
premier beach town like St-Peter-Ording earns nothing. The visitable SAND —
natural=beach — is the real outing spot (and, unlike the protected mudflat, the
place you can actually go), so it feeds Nature as its own best-eligible type.

Point-samples each beach polygon at 50 m in EPSG:3035 (equal-area) and maps the
in-polygon points to res-8 H3 → beach_share = sampled area / hex area. Reads the
small region-filtered.osm.pbf (keeps natural=*); a cheap osmium pre-filter keeps
the geopandas read tiny. Output: data/interim/beach.parquet (h3, beach_share).
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer

from wohnen.config import INTERIM

SRC = INTERIM / "region-filtered.osm.pbf"
BEACH_PBF = INTERIM / "beach.osm.pbf"
HEX_AREA_M2 = 737327.0   # mean res-8 hexagon area
STEP = 50.0              # sample spacing (m); area per in-polygon point = STEP^2


def main():
    print("osmium tags-filter natural=beach ...")
    subprocess.run(["osmium", "tags-filter", "--overwrite", "-o", str(BEACH_PBF),
                    str(SRC), "nwr/natural=beach"], check=True)
    poly = gpd.read_file(BEACH_PBF, layer="multipolygons")
    beach = poly[poly["natural"] == "beach"].to_crs(3035)
    print(f"  {len(beach)} beach polygons, {beach.area.sum()/1e6:.0f} km²")

    to4326 = Transformer.from_crs(3035, 4326, always_xy=True)
    area: dict[str, float] = {}
    for geom in beach.geometry.values:
        if geom is None or geom.is_empty:
            continue
        minx, miny, maxx, maxy = geom.bounds
        gx, gy = np.meshgrid(np.arange(minx, maxx + STEP, STEP),
                             np.arange(miny, maxy + STEP, STEP))
        gx, gy = gx.ravel(), gy.ravel()
        if not len(gx):
            continue
        inside = shapely.within(shapely.points(gx, gy), geom)
        if not inside.any():
            rp = geom.representative_point()      # thin sliver: keep its presence
            gx, gy, inside = np.array([rp.x]), np.array([rp.y]), np.array([True])
        lon, lat = to4326.transform(gx[inside], gy[inside])
        for la, lo in zip(lat, lon):
            c = h3.latlng_to_cell(la, lo, 8)
            area[c] = area.get(c, 0.0) + STEP * STEP

    out = pd.DataFrame({"h3": list(area),
                        "beach_share": np.minimum(1.0, np.array(list(area.values()))
                                                  / HEX_AREA_M2)})
    INTERIM.mkdir(parents=True, exist_ok=True)
    out.to_parquet(INTERIM / "beach.parquet", index=False)
    print(f"  {len(out)} cells carry beach; share p50/p90/max = "
          f"{np.percentile(out['beach_share'], [50, 90]).round(3).tolist()}"
          f"/{out['beach_share'].max():.2f}")


if __name__ == "__main__":
    main()
