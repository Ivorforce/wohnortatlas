"""H3 helpers: bbox fill, distance-discounted disk aggregation, point assignment."""

import h3
import numpy as np
import pandas as pd
import shapely.geometry

from .config import BBOX, CELL_RMS_M, H3_RES

# mean center-to-center distance between adjacent res-8 cells, km
RES8_STEP_KM = 0.93
# both endpoints of a cell-to-cell distance are spread within their cells,
# so the variances add: E[dist] ~= sqrt(d_center^2 + 2 * CELL_RMS^2)
_PAIR_RMS_KM = CELL_RMS_M * np.sqrt(2) / 1000


def bbox_cells(bbox=BBOX, res=H3_RES) -> list[str]:
    poly = shapely.geometry.box(*bbox)
    return sorted(h3.geo_to_cells(poly, res))


def cells_latlng(cells) -> tuple[np.ndarray, np.ndarray]:
    pts = [h3.cell_to_latlng(c) for c in cells]
    return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])


def points_to_cells(lat, lon, res=H3_RES) -> list[str]:
    return [h3.latlng_to_cell(la, lo, res) for la, lo in zip(lat, lon)]


def disk_median(values: pd.Series, k: int = 1, min_count: int = 1) -> pd.Series:
    """Median over grid_disk(k) per cell — robust to single-cell outliers.

    values: Series indexed by h3 cell; cells missing from the index are
    ignored in each neighbourhood. Cells with fewer than min_count data
    cells in their disk become NaN ("insufficient evidence").
    """
    vdict = values.dropna().to_dict()
    out = {}
    for cell in values.index:
        nb = [vdict[c] for c in h3.grid_disk(cell, k) if c in vdict]
        out[cell] = float(np.median(nb)) if len(nb) >= min_count else np.nan
    return pd.Series(out).reindex(values.index)


def disk_weighted_sum(
    counts: pd.Series, grid: list[str], k: int, scale_km: float
) -> pd.Series:
    """For each cell in `grid`, sum counts over grid_disk(k) with exp(-d/scale) decay.

    counts: Series indexed by h3 cell (any cells, e.g. POI counts).
    Distance d = grid ring index * RES8_STEP_KM, de-biased by the in-cell
    spread of both endpoints (ring 0 is ~480 m apart on average, not 0).
    """
    counts = counts[counts > 0]
    cdict = counts.to_dict()
    ring_w = [np.exp(-np.hypot(ring * RES8_STEP_KM, _PAIR_RMS_KM) / scale_km)
              for ring in range(k + 1)]
    out = np.zeros(len(grid))
    for i, cell in enumerate(grid):
        acc = 0.0
        for ring in range(k + 1):
            cells = h3.grid_ring(cell, ring) if ring else [cell]
            w = ring_w[ring]
            for c in cells:
                v = cdict.get(c)
                if v:
                    acc += v * w
        out[i] = acc
    return pd.Series(out, index=grid)


def disk_weighted_max(
    values: pd.Series, grid: list[str], k: int, scale_km: float
) -> pd.Series:
    """For each cell in `grid`, the max over grid_disk(k) of value * exp(-d/scale).

    Distance-discounted *max* ("dilate") instead of disk_weighted_sum's sum:
    the answer to "what is the best place I can reach", not "how much is around
    me". Ring 0 (self) is undiscounted (w=1.0) — being on it = full credit —
    unlike disk_weighted_sum's _PAIR_RMS_KM de-bias; ring r decays by
    exp(-r * RES8_STEP_KM / scale). The early break assumes values <= 1: once a
    ring's weight can't beat the running best, no farther ring can either.
    """
    vdict = values[values > 0].to_dict()
    ring_w = [1.0] + [np.exp(-(r * RES8_STEP_KM) / scale_km) for r in range(1, k + 1)]
    out = np.zeros(len(grid))
    for i, cell in enumerate(grid):
        best = 0.0
        for ring in range(k + 1):
            if ring_w[ring] <= best:  # values <= 1 => no farther ring can beat best
                break
            cells = h3.grid_ring(cell, ring) if ring else [cell]
            w = ring_w[ring]
            for c in cells:
                v = vdict.get(c)
                if v and v * w > best:
                    best = v * w
        out[i] = best
    return pd.Series(out, index=grid)


def disk_gaussian_mean(values: pd.Series, k: int, sigma_km: float) -> pd.Series:
    """Normalized Gaussian-weighted mean over grid_disk(k) per cell.

    Unlike disk_weighted_sum, the weights are renormalized per cell, so this
    is an average rather than a sum — the right tool for feathering a
    step-valued surface (e.g. Kreis-level income) across its boundaries: cell
    interiors stay flat (all neighbours share the value), borders blend
    smoothly. Gaussian decay matches the rent surface's kernel_shrink. Cells
    absent from `values` don't contribute and don't count toward the weight;
    a cell whose disk holds no data becomes NaN.
    """
    vdict = values.dropna().to_dict()
    ring_w = [np.exp(-np.hypot(ring * RES8_STEP_KM, _PAIR_RMS_KM) ** 2
                     / (2 * sigma_km ** 2)) for ring in range(k + 1)]
    out = {}
    for cell in values.index:
        num = den = 0.0
        for ring in range(k + 1):
            cells = h3.grid_ring(cell, ring) if ring else [cell]
            w = ring_w[ring]
            for c in cells:
                v = vdict.get(c)
                if v is not None:
                    num += v * w
                    den += w
        out[cell] = num / den if den else np.nan
    return pd.Series(out).reindex(values.index)
