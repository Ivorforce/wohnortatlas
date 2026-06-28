"""Demographics layer: Zensus avg age grid + under-18 / over-65 share per cell."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from wohnen.config import LAYERS, RAW
from wohnen.h3util import disk_median
from wohnen.zensus import add_h3, read_grid_csv


def main():
    age = read_grid_csv(RAW / "zensus_age.zip",
                        "Zensus2022_Durchschnittsalter_1km-Gitter.csv",
                        "Durchschnittsalter")
    age = add_h3(age)
    avg_age = age.groupby("h3")["value"].mean().rename("avg_age")
    # start from the population layer (07a_population.py) so demographics.parquet stays a
    # SUPERSET (population + catchments + the demographic-character fields below) and every
    # existing consumer keeps reading one file; only the routing path (04b) reads
    # population.parquet directly. Editing the character logic here never touches that file.
    out = pd.read_parquet(LAYERS / "population.parquet")
    out["avg_age"] = out["h3"].map(avg_age)  # smoothed below (neighbourhood mix)

    # Age-band shares (% of residents) per cell — the life-stage signal the mean
    # can't carry. avg_age conflates a family Neubaugebiet (parents + kids) with a
    # student/WG area; both read ~39. share_u18 is immune (a WG has ~0 kids, a
    # student household of 4 still counts as 0 under-18s), share_65plus pins
    # senior enclaves. Stored as a fraction [0,1]. `–` (suppressed, <few residents)
    # → NaN, dropped by read_grid_csv. All three are smoothed below into a
    # neighbourhood mix.
    for col, zname, csv_name, vcol in [
        ("share_u18", "zensus_u18.zip",
         "Zensus2022_Anteil_unter_18_1km-Gitter.csv", "AnteilUnter18"),
        ("share_65plus", "zensus_65plus.zip",
         "Zensus2022_Anteil_ueber_65_1km-Gitter.csv", "AnteilUeber65"),
    ]:
        band = read_grid_csv(RAW / zname, csv_name, vcol)
        band = add_h3(band)
        frac = (band.groupby("h3")["value"].mean() / 100.0).rename(col)
        out[col] = out["h3"].map(frac)

    # population / inhabited-density / catchment sums (population, pop_inhabited_dens,
    # catchment_pop/wide/leisure) come in via the population.parquet read above — they're
    # produced by 07a_population.py so the routing path doesn't depend on this script.

    # Light k=1 median despeckle on the demographic inputs: 1 km-grid cells at village
    # edges / non-residential hexes (a Klinikum) are noisy. NOT a vibe-spread — a mean
    # would dilute small sharp enclaves (a student dorm averaged into surrounding
    # families loses its low-u18 signal). The "happy living NEXT TO a student/family
    # cluster" smoothing is done on the life-stage MEMBERSHIP score in the web (a
    # proximity max-with-decay that lifts neighbours without lowering the source — see
    # recomputeAge), where student-ness is a HIGH value that spreads correctly.
    for col in ["avg_age", "share_u18", "share_65plus"]:
        out[col] = disk_median(out.set_index("h3")[col], k=1).values

    LAYERS.mkdir(parents=True, exist_ok=True)
    out.to_parquet(LAYERS / "demographics.parquet", index=False)
    print(out[["avg_age", "share_u18", "share_65plus"]]
          .describe().round(3).to_string())


if __name__ == "__main__":
    main()
