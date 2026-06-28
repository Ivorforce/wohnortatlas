"""Broadband layer: speed-ladder availability per hex (Breitbandatlas 100m cells).

bb_score = 0.5*(>=100 Mbit) + 0.3*(>=1 Gbit) + 0.2*fibre, all "any technology"
household shares. An adequate floor plus a future-proof bonus, rather than
crediting only the patchy gigabit/fibre top rungs.
"""

import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h3
import numpy as np
import pandas as pd
import pyogrio

from wohnen.config import INTERIM, LAYERS, RAW
from wohnen.h3util import points_to_cells
from wohnen.zensus import bbox_3035

ZIP = RAW / "breitband_gitterzellen.gpkg.zip"


def extract_gpkg() -> Path:
    dest_dir = RAW / "breitband"
    existing = list(dest_dir.glob("*.gpkg"))
    if existing:
        return existing[0]
    dest_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(ZIP) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".gpkg"))
        zf.extract(name, dest_dir)
    return next(dest_dir.glob("**/*.gpkg"))


def main():
    gpkg = extract_gpkg()
    layers = pyogrio.list_layers(gpkg)
    print("layers:", layers)
    layer = layers[0][0]
    info = pyogrio.read_info(gpkg, layer=layer)
    print("fields:", info["fields"])
    crs = str(info["crs"])

    bbox = bbox_3035() if "3035" in crs else None
    gdf = pyogrio.read_dataframe(gpkg, layer=layer, bbox=bbox)
    print(f"cells in bbox: {len(gdf)}")

    cols = {c.lower(): c for c in gdf.columns}
    def col(name: str) -> str:
        if name not in cols:
            raise KeyError(f"expected Breitbandatlas column {name!r}; "
                           f"have e.g. {list(cols)[:8]}…")
        return cols[name]

    # BNetzA Breitbandatlas schema: down_fn_hh_<tech>_<min Mbit>, where the
    # value is the share of households reaching that downstream speed with that
    # technology. We ride the speed ladder rather than only the gigabit/fiber
    # top rungs: an "adequate" floor (>=100 Mbit, fine for any household) plus a
    # diminishing bonus for gigabit and fiber. "alle" = any technology;
    # "ftthb" = fibre to the building/home (its >=10 rung is just fiber presence,
    # since fiber is never the bottleneck).
    SHARES = {
        "share_100": col("down_fn_hh_alle_100"),    # >=100 Mbit, any tech
        "share_1000": col("down_fn_hh_alle_1000"),  # >=1 Gbit, any tech
        "ftth_share": col("down_fn_hh_ftthb_10"),   # fibre to building/home
    }
    print(f"using columns: {SHARES}")

    cent = gdf.geometry.representative_point().to_crs(4326)
    gdf["h3"] = points_to_cells(cent.y.values, cent.x.values)

    agg = {}
    for label, src in SHARES.items():
        v = pd.to_numeric(gdf[src], errors="coerce")
        if v.max() is not None and v.max() > 1.5:
            v = v / 100.0
        agg[label] = v.groupby(gdf["h3"]).mean()

    grid = pd.read_parquet(INTERIM / "grid.parquet")
    out = grid[["h3"]].copy()
    for label, series in agg.items():
        out[label] = out["h3"].map(series)

    # A populated hex can read worst-in-nation broadband two ways: its representative
    # point misses the served-cell aggregation (NaN), or the Breitbandatlas reports an
    # isolated 0 while every inhabited neighbour is well served. Both are false "internet
    # deserts" inside covered areas — broadband rolls out street-by-street, so a hex fully
    # enclosed by served hexes cannot really be unserved. Repair from inhabited served
    # neighbours: a NaN cell fills from any >=2 of them; a genuine-0 cell is repaired only
    # when strongly contradicted (>=4 neighbours, well-served median, large gap), so real
    # coverage edges and true low-coverage clusters are left untouched.
    cols = list(SHARES)
    W = np.array([0.5, 0.3, 0.2])              # share weights -> bb_score
    cells = out["h3"].to_numpy()
    pos = {c: i for i, c in enumerate(cells)}
    V = out[cols].to_numpy(dtype=float)

    pop = pd.read_parquet(LAYERS / "population.parquet",
                          columns=["h3", "population"]).set_index("h3")["population"]
    inhab = (out["h3"].map(pop).fillna(0) >= 10).to_numpy()

    for _ in range(3):
        bb = np.where(np.isnan(V[:, 0]), np.nan, V @ W)
        cand = np.where(inhab & (np.isnan(bb) | (bb <= 0.1)))[0]
        fills = []
        for i in cand:
            nb = [V[pos[r]] for r in h3.grid_disk(cells[i], 1)
                  if r != cells[i] and r in pos and inhab[pos[r]]
                  and not np.isnan(V[pos[r], 0])]
            if len(nb) < 2:
                continue
            nb = np.array(nb)
            nmed = float(np.median(nb @ W))
            outlier = len(nb) >= 4 and nmed >= 0.5 and bb[i] <= nmed - 0.4
            if np.isnan(bb[i]) or outlier:
                fills.append((i, nb.mean(axis=0)))
        if not fills:
            break
        for i, vec in fills:
            V[i] = vec
    out[cols] = V

    s100 = out["share_100"].fillna(0)
    s1000 = out["share_1000"].fillna(0)
    ftth = out["ftth_share"].fillna(0)
    # adequate-floor + future-proof bonus: 100 Mbit covers any real need, gigabit
    # and fiber add headroom. Weights sum to 1 so a fully-fibered hex scores 1.
    out["bb_score"] = 0.5 * s100 + 0.3 * s1000 + 0.2 * ftth

    LAYERS.mkdir(parents=True, exist_ok=True)
    out.to_parquet(LAYERS / "broadband.parquet", index=False)
    print(out.drop(columns="h3").describe().round(3).to_string())


if __name__ == "__main__":
    main()
