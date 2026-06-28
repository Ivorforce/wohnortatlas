"""Nearest-OSM-place labels for points (offline, no geocoder)."""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .config import INTERIM
from .h3util import utm32_transformer

FINE = ["suburb", "neighbourhood", "village", "town", "city"]
FINE_RADIUS_M = 1500.0

_T = utm32_transformer()


def labels_for_points(lat, lon) -> np.ndarray:
    """Nearest fine-grained place within 1.5 km, else nearest place of any kind."""
    places = pd.read_parquet(INTERIM / "places.parquet")
    px, py = _T.transform(places["lon"].values, places["lat"].values)
    names = places["name"].values
    fine = places["place"].isin(FINE).values

    qx, qy = _T.transform(np.asarray(lon), np.asarray(lat))
    xy = np.c_[qx, qy]
    d_f, i_f = cKDTree(np.c_[px, py][fine]).query(xy)
    _, i_a = cKDTree(np.c_[px, py]).query(xy)
    return np.where(d_f <= FINE_RADIUS_M, names[fine][i_f], names[i_a])
