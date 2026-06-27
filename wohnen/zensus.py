"""Readers for Zensus 2022 grid CSVs (semicolon, decimal comma, EPSG:3035 centers)."""

import zipfile
from pathlib import Path

import h3
import numpy as np
import pandas as pd
from pyproj import Transformer

from .config import BBOX, H3_RES

_T = Transformer.from_crs(3035, 4326, always_xy=True)


def bbox_3035() -> tuple[float, float, float, float]:
    t = Transformer.from_crs(4326, 3035, always_xy=True)
    # transform all 4 corners; LAEA is rotated relative to lon/lat
    xs, ys = zip(*(t.transform(lo, la)
                   for lo in (BBOX[0], BBOX[2]) for la in (BBOX[1], BBOX[3])))
    return min(xs), min(ys), max(xs), max(ys)


def read_grid_csv(
    zip_path: Path,
    csv_name: str,
    value_col: str,
    usecols: list[str] | None = None,
    chunksize: int = 2_000_000,
) -> pd.DataFrame:
    """Read a Zensus grid CSV from a zip, clip to bbox, return df with x, y, value (+usecols)."""
    x0, y0, x1, y1 = bbox_3035()
    zf = zipfile.ZipFile(zip_path)
    chunks = []
    cols = None
    with zf.open(csv_name) as f:
        for chunk in pd.read_csv(f, sep=";", chunksize=chunksize, dtype=str):
            if cols is None:
                cols = chunk.columns
                xcol = next(c for c in cols if c.startswith("x_mp"))
                ycol = next(c for c in cols if c.startswith("y_mp"))
            x = pd.to_numeric(chunk[xcol], errors="coerce")
            y = pd.to_numeric(chunk[ycol], errors="coerce")
            m = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
            if not m.any():
                continue
            sub = chunk[m].copy()
            sub["x"], sub["y"] = x[m], y[m]
            sub["value"] = pd.to_numeric(
                sub[value_col].str.replace(",", ".", regex=False), errors="coerce")
            keep = ["x", "y", "value"] + (usecols or [])
            chunks.append(sub[keep])
    df = pd.concat(chunks, ignore_index=True).dropna(subset=["value"])
    return df


def add_h3(df: pd.DataFrame, res: int = H3_RES) -> pd.DataFrame:
    lon, lat = _T.transform(df["x"].values, df["y"].values)
    df = df.copy()
    df["h3"] = [h3.latlng_to_cell(la, lo, res) for la, lo in zip(lat, lon)]
    return df
