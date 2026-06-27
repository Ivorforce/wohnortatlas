"""Vacancy layer: Wohnungsleerstand per hex (Zensus 2022 grid) — "Leerstand & Verfall".

The closest open, national, sub-Kreis proxy for "is this place stable or hollowing
out" — the decline signal rent can't carry (rent says cheap-or-dear; among cheap
hexes, vacancy splits affordable-rural from emptying-out). Measures the housing
STOCK decaying, not the people, so it sits on the anti-stigma side of the line
(crime / Ausländeranteil are deliberately omitted).

Resolution is the whole game: the 100m Leerstandsquote is small-denominator noise
(clean cells read median ~23 %, P95 100 % — a 2-dwelling cell with 1 empty = 50 %).
At 1km the clean (non-KLAMMERN) cells match Germany's real vacancy (median ~5 %,
mean ~6 %), so we read the 1km grid and assign each res-8 hex its enclosing cell's
rate — still ~1000× finer than the Kreis-flat crime data we rejected. KLAMMERN
(low-reliability, small case count) cells are dropped, not trusted.

Coverage is urban-skewed: vacancy is a multi-dwelling concept, so rural
single-family areas carry no value (~74 % of population covered, ~21 % of hexes).
That is why this is a low-weight, opt-in layer with no-data = neutral (1.0) and
NO veto in the web — silence over a quiet village can't be read as decline.

  vacancy_score = 1 − clip((vacancy% − VAC_OK) / (VAC_BAD − VAC_OK), 0, 1)

VAC_OK = healthy frictional vacancy (no penalty); VAC_BAD = clear structural
decline (full penalty). dl-de/by-2-0; attribute Statistische Ämter / Zensus 2022.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h3
import numpy as np
import pandas as pd
from pyproj import Transformer

from wohnen.config import INTERIM, LAYERS, RAW
from wohnen.zensus import read_grid_csv

ZIP = RAW / "zensus_vacancy.zip"
CSV_1KM = "Zensus2022_Leerstandsquote_1km-Gitter.csv"

VAC_OK = 5.0    # vacancy % up to which there's no penalty (≈ national median / frictional)
VAC_BAD = 18.0  # vacancy % at which the score floors at 0 (≈ P97 — clear structural decline)

_T = Transformer.from_crs(4326, 3035, always_xy=True)


def main():
    # 1km grid, clean cells only — KLAMMERN (small case count) values are unreliable.
    df = read_grid_csv(ZIP, CSV_1KM, "Leerstandsquote",
                       usecols=["werterlaeuternde_Zeichen"])
    zeichen = df["werterlaeuternde_Zeichen"].fillna("").str.strip()
    clean = df[zeichen == ""].copy()
    print(f"1km cells in bbox: numeric {len(df)}, clean(non-KLAMMERN) {len(clean)} "
          f"(median {clean['value'].median():.1f} %, mean {clean['value'].mean():.1f} %)")

    # key each 1km cell by its SW-origin (floor of the EPSG:3035 center to 1 km)
    clean["cx"] = np.floor(clean["x"] / 1000).astype(np.int64)
    clean["cy"] = np.floor(clean["y"] / 1000).astype(np.int64)
    vac = clean.groupby(["cx", "cy"])["value"].mean()

    # each res-8 hex inherits its enclosing 1km cell's rate (a 1km cell spans ~1.4
    # hexes, so tagging only the cell center would miss most hexes — bin the hex).
    grid = pd.read_parquet(INTERIM / "grid.parquet")
    lat = np.array([h3.cell_to_latlng(c)[0] for c in grid["h3"]])
    lon = np.array([h3.cell_to_latlng(c)[1] for c in grid["h3"]])
    hx, hy = _T.transform(lon, lat)
    key = pd.MultiIndex.from_arrays(
        [np.floor(hx / 1000).astype(np.int64), np.floor(hy / 1000).astype(np.int64)])
    vac_pct = vac.reindex(key).to_numpy()  # NaN where the cell has no clean value

    score = 1.0 - np.clip((vac_pct - VAC_OK) / (VAC_BAD - VAC_OK), 0.0, 1.0)

    out = grid[["h3"]].copy()
    out["vacancy_pct"] = np.round(vac_pct, 1)       # tooltip; NaN = no data
    out["vacancy_score"] = score                    # NaN where no data (assemble fills 1.0)
    LAYERS.mkdir(parents=True, exist_ok=True)
    out.to_parquet(LAYERS / "vacancy.parquet", index=False)

    cov = np.isfinite(vac_pct)
    print(f"vacancy: {cov.mean():.1%} of hexes covered; "
          f"score<0.5: {(score < 0.5).sum()} hexes, score<0.2: {(score < 0.2).sum()}")
    print(out.loc[cov, ["vacancy_pct", "vacancy_score"]].describe().round(2).to_string())


if __name__ == "__main__":
    main()
