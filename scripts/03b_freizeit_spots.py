"""Extract the s_freizeit point SOURCES from the full POI table into a minimal
freizeit_spots.parquet — the ONLY POI input to the expensive 04d_swim reverse-routing.

Isolating these rows means editing unrelated POI extraction (03_pois: doctors, shops,
schools, …) no longer invalidates the swim/kino/klettern/golf routing. Written
content-aware (write_parquet_if_changed): reach_spots.npz only re-routes when these
specific spot rows actually change, not whenever pois.parquet is regenerated."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from wohnen.config import INTERIM
from wohnen.freizeit import SOURCE_PAIRS
from wohnen.io import write_parquet_if_changed

COLS = ["category", "subcategory", "lat", "lon"]  # all 04d_swim.load_spots needs


def main():
    pois = pd.read_parquet(INTERIM / "pois.parquet")
    pairs = pd.MultiIndex.from_frame(pois[["category", "subcategory"]])
    keep = pois[pairs.isin(SOURCE_PAIRS)][COLS].reset_index(drop=True)
    written = write_parquet_if_changed(keep, INTERIM / "freizeit_spots.parquet", sort_cols=COLS)
    by_cat = keep.groupby(["category", "subcategory"]).size().to_dict()
    print(f"freizeit_spots.parquet: {len(keep)} spots {by_cat} "
          f"({'written' if written else 'unchanged'})")


if __name__ == "__main__":
    main()
