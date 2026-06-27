"""Build the H3 res-8 grid over the study bbox, clipped to German territory."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import geopandas as gpd
import pandas as pd
import shapely

from wohnen.config import INTERIM, RAW
from wohnen.h3util import bbox_cells, cells_latlng

_VG250_STA = ("zip://{zip}!vg250_01-01.utm32s.shape.ebenen/"
              "vg250_ebenen_0101/VG250_STA.shp")


def germany_land():
    """Dissolved German land territory (VG250 Staatsgebiet, GF=4 = Festland;
    GF 1/2/3 are territorial/coastal water), EPSG:4326, lightly simplified
    (~500 m, well under the res-8 edge) for fast point-in-polygon."""
    g = gpd.read_file(_VG250_STA.format(zip=RAW / "vg250.zip"))
    land = g[g["GF"] == 4].to_crs(4326)
    return land.geometry.union_all().simplify(0.005)


def main():
    INTERIM.mkdir(parents=True, exist_ok=True)
    cells = bbox_cells()
    lat, lon = cells_latlng(cells)

    # Clip the raw bbox rectangle to German territory: the Germany bbox spans
    # the North/Baltic Sea and slices of neighbouring countries, which we must
    # neither score nor fetch noise/other tiles for.
    poly = germany_land()
    shapely.prepare(poly)
    inside = shapely.contains(poly, shapely.points(lon, lat))

    cells = [c for c, k in zip(cells, inside) if k]
    lat, lon = lat[inside], lon[inside]
    df = pd.DataFrame({"h3": cells, "lat": lat, "lon": lon})
    df.to_parquet(INTERIM / "grid.parquet", index=False)
    print(f"grid: {len(df)} cells (clipped to DE land), "
          f"lat {lat.min():.3f}..{lat.max():.3f}, "
          f"lon {lon.min():.3f}..{lon.max():.3f}")


if __name__ == "__main__":
    main()
