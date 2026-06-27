"""INSPIRE ETRS89-LAEA grid cell ids -> WGS84 cell centers -> h3 cells.

Ids look like 'CRS3035RES1000mN2769000E4341000' or '1kmN2769E4341' /
'100mN27690E43410' (Zensus CSV style). N/E are in meters (full form) or in
units of the resolution (short form), referring to the lower-left corner.
"""

import re

import h3
import numpy as np
from pyproj import Transformer

from .config import H3_RES

_T3035 = Transformer.from_crs(3035, 4326, always_xy=True)

_FULL = re.compile(r"CRS3035RES(\d+)mN(\d+)E(\d+)")
_SHORT = re.compile(r"(\d+)(km|m)N(\d+)E(\d+)")


def parse_inspire_id(cell_id: str) -> tuple[float, float, int]:
    """Return (easting_center, northing_center, resolution_m)."""
    m = _FULL.match(cell_id)
    if m:
        res, n, e = int(m[1]), int(m[2]), int(m[3])
        return e + res / 2, n + res / 2, res
    m = _SHORT.match(cell_id)
    if m:
        res = int(m[1]) * (1000 if m[2] == "km" else 1)
        n, e = int(m[3]) * res, int(m[4]) * res
        return e + res / 2, n + res / 2, res
    raise ValueError(f"unparseable INSPIRE id: {cell_id!r}")


def inspire_ids_to_h3(cell_ids, res: int = H3_RES) -> list[str]:
    """Vectorized-ish conversion of INSPIRE ids to h3 cells (at cell centers)."""
    parsed = [parse_inspire_id(c) for c in cell_ids]
    e = np.array([p[0] for p in parsed])
    n = np.array([p[1] for p in parsed])
    lon, lat = _T3035.transform(e, n)
    return [h3.latlng_to_cell(la, lo, res) for la, lo in zip(lat, lon)]
