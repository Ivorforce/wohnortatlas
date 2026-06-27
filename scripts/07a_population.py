"""Population layer: Zensus headcount grid → per-hex population, inhabited-area density,
and distance-decayed catchment sums.

Split out of 07_demographics so the expensive reverse-routing is not invalidated by
demographic-character edits. The center placement (04b) + the reach routing (04c) read ONLY
population + catchment_pop from here; the age / life-stage share fields live
in 07_demographics, which reads THIS file and enriches it. Written content-aware
(write_parquet_if_changed): regenerating it leaves the file — and therefore the routed
reach_centers.npz — untouched unless the headcounts actually move."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from wohnen.config import INTERIM, LAYERS, RAW
from wohnen.h3util import disk_median, disk_weighted_sum
from wohnen.io import write_parquet_if_changed
from wohnen.zensus import add_h3, read_grid_csv


def main():
    grid = pd.read_parquet(INTERIM / "grid.parquet")
    out = grid[["h3"]].copy()

    # population per hex from the 100m grid (sums cleanly into res-8 cells)
    pop = read_grid_csv(RAW / "zensus_pop.zip",
                        "Zensus2022_Bevoelkerungszahl_100m-Gitter.csv",
                        "Einwohner")
    pop = add_h3(pop)
    g = pop.groupby("h3")["value"]
    out["population"] = out["h3"].map(g.sum()).fillna(0)
    # Inhabited-area density: residents per km² of *populated* land, not per
    # whole hex. The 100m grid lists only inhabited cells (each = 0.01 km²), so
    # sum / (0.01 * n_inhabited_subcells) is density over the built footprint.
    # A hex straddling a city's hard edge is half empty field; raw pop/hex then
    # reads rural and falsely flags the urban edge as "Land". This reads the
    # built half at its true city density, while genuine countryside (sparse
    # everywhere) stays low. Drives s_density (the Stadt-oder-Land preference);
    # the consequences of density — Anbindung, Einkaufen, Freizeit, Grün — are
    # scored by their own layers, so this is form only, not embedding/size.
    dens = g.sum() / (g.size() * 0.01)
    # Despeckle: a single hex dominated by non-residential land (a Klinikum,
    # cemetery, sports ground) has few inhabited subcells and reads far below
    # its dense surroundings — e.g. a Großhadern hex at ~1150 amid 6–10k
    # neighbours. A k=1 median over *inhabited* neighbours snaps such lone
    # outliers back to the local level while leaving smooth areas and genuine
    # edges intact. `dens` holds only inhabited cells, so uninhabited hexes
    # neither enter the median nor get lifted to urban — the edge fix stays.
    dens = disk_median(dens, k=1)
    out["pop_inhabited_dens"] = out["h3"].map(dens)
    print(f"population: {out['population'].sum():,.0f} total, "
          f"{(out['population'] > 0).mean():.1%} of hexes populated; "
          f"inhabited-dens median {out['pop_inhabited_dens'].median():,.0f}/km²")

    # catchment populations (distance-decayed): how many people share this
    # area's amenities — crowding discounts downstream. Local (~4 km) for
    # Kitas/cafés; wide (~7 km, matching nature-reach radius) for nature;
    # leisure matches the entertainment kernel EXACTLY (k=6, 1.5 km) so that
    # per-capita supply r = ent_density / catchment_leisure is apples-to-apples
    # (same disk, same decay) — the supply-relative crowding term in 20 needs it.
    pop_s = pd.Series(out["population"].values, index=out["h3"])
    cells = out["h3"].tolist()
    out["catchment_pop"] = disk_weighted_sum(pop_s, cells, k=4, scale_km=3.0).values
    out["catchment_wide"] = disk_weighted_sum(pop_s, cells, k=7, scale_km=4.0).values
    out["catchment_leisure"] = disk_weighted_sum(pop_s, cells, k=6, scale_km=1.5).values

    LAYERS.mkdir(parents=True, exist_ok=True)
    written = write_parquet_if_changed(out, LAYERS / "population.parquet", sort_cols=["h3"])
    print(f"population.parquet: {len(out)} cells ({'written' if written else 'unchanged'})")


if __name__ == "__main__":
    main()
