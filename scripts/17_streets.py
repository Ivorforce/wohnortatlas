"""Street-network 'grown-ness' for Ortsbild: orientation entropy per H3 cell.

Each street segment's compass bearing (length-weighted, two-way) goes into a
36-bin histogram per cell; aggregated over the cell + 1-ring, its normalized
Shannon entropy Hn ∈ [0,1] measures how ORDERED the street grid is:
  LOW  Hn = gridded / planned (Manhattan, postwar Rasterstadt) — bearings pile
            up on a few axes;
  HIGH Hn = organic / grown (medieval Altstadt) — bearings point every way.
A grown layout reads as "nicer/more character" than a grid (validated against
labelled places — Altstädte land p67-99, planned/grid p11-24; design.md).

`street_grain` = Hn stretched between empirical anchors (a real grid ≈ HN_LO → 0,
a medieval core ≈ HN_HI → 1) and GATED to street-rich cells (entropy is noise
where there's no fabric, and Ortsbild is moot in open country) → 0 elsewhere.
Dead-end share is not used: German Trabantenstädte are connected apartment
estates (not cul-de-sac sprawl) and Altstädte have Sackgassen, so it ranks
backwards. Entropy alone already ranks curvy-modern below the Altstädte.

Reads region-filtered.osm.pbf (its tags-filter keeps nwr/highway = ALL highway
classes, so the small 2 GB extract suffices); no routing.
Output: data/layers/streets.parquet (h3, street_grain, street_entropy, street_len_m).
"""
import sys
from collections import defaultdict
from math import atan2, cos, degrees, radians, sin, sqrt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h3
import numpy as np
import osmium
import pandas as pd

from wohnen.config import INTERIM, LAYERS
from wohnen.norm import clip01

SRC = INTERIM / "region-filtered.osm.pbf"
# the street fabric you live among (not motorways/ramps/service/tracks/paths)
STREET = {"primary", "secondary", "tertiary", "unclassified", "residential",
          "living_street", "pedestrian", "road"}
NBINS = 36                    # 10° bearing bins
DISK_K = 1                    # aggregate each cell over its 1-ring for a stable sample
HN_LO, HN_HI = 0.88, 0.98     # stretch anchors: planned grid → 0, medieval core → 1
MIN_LEN_M = 4000.0            # min street length in the disk to trust the entropy
BIN = 360 / NBINS


def _bearing(lon1, lat1, lon2, lat2):
    p1, p2, dl = radians(lat1), radians(lat2), radians(lon2 - lon1)
    x = sin(dl) * cos(p2)
    y = cos(p1) * sin(p2) - sin(p1) * cos(p2) * cos(dl)
    return (degrees(atan2(x, y)) + 360) % 360


def _metres(lon1, lat1, lon2, lat2):
    p1, p2, dl = radians(lat1), radians(lat2), radians(lon2 - lon1)
    a = sin((p2 - p1) / 2) ** 2 + cos(p1) * cos(p2) * sin(dl / 2) ** 2
    return 6371000 * 2 * atan2(sqrt(a), sqrt(1 - a))


class Streets(osmium.SimpleHandler):
    def __init__(self):
        super().__init__()
        self.hist = defaultdict(lambda: np.zeros(NBINS))  # cell -> length per bin

    def way(self, w):
        if w.tags.get("highway") not in STREET:
            return
        pts = [(nd.location.lon, nd.location.lat) for nd in w.nodes
               if nd.location.valid()]
        for (lo1, la1), (lo2, la2) in zip(pts, pts[1:]):
            L = _metres(lo1, la1, lo2, la2)
            if L <= 0:
                continue
            b = _bearing(lo1, la1, lo2, la2)
            hb = self.hist[h3.latlng_to_cell((la1 + la2) / 2, (lo1 + lo2) / 2, 8)]
            hb[int(b // BIN) % NBINS] += L
            hb[int((b + 180) // BIN) % NBINS] += L   # undirected


def _entropy(hist):
    s = hist.sum()
    if s <= 0:
        return np.nan
    p = hist[hist > 0] / s
    return float(-(p * np.log(p)).sum() / np.log(NBINS))   # normalized [0,1]


def main():
    h = Streets()
    print(f"reading {SRC} ...")
    h.apply_file(str(SRC), locations=True, idx="flex_mem")
    print(f"  {len(h.hist)} cells carry street segments")

    grid = pd.read_parquet(INTERIM / "grid.parquet")
    ent = np.full(len(grid), np.nan)
    length = np.zeros(len(grid))
    for i, c in enumerate(grid["h3"]):
        hist = np.zeros(NBINS)
        for nb in h3.grid_disk(c, DISK_K):
            hist += h.hist.get(nb, 0.0)
        length[i] = hist.sum()
        ent[i] = _entropy(hist)

    grain = clip01((ent - HN_LO) / (HN_HI - HN_LO))
    grain[length < MIN_LEN_M] = 0.0          # no fabric -> no street-grain character
    grain = np.nan_to_num(grain, nan=0.0)

    out = pd.DataFrame({
        "h3": grid["h3"],
        "street_grain": grain,
        "street_entropy": ent,
        "street_len_m": length,
    })
    LAYERS.mkdir(parents=True, exist_ok=True)
    out.to_parquet(LAYERS / "streets.parquet", index=False)

    gated = grain[length >= MIN_LEN_M]
    print(f"street_grain: {len(gated)} fabric cells; "
          f"p10/50/90 = {np.percentile(gated, [10, 50, 90]).round(2).tolist()}")
    for n, la, lo in [("Eichstätt Altstadt", 48.8916, 11.1844),
                      ("Wasserburg a.Inn", 48.0590, 12.2290),
                      ("München Maxvorstadt", 48.1510, 11.5660),
                      ("München Bogenhausen", 48.1530, 11.6160)]:
        idx = grid.index[grid["h3"] == h3.latlng_to_cell(la, lo, 8)]
        if len(idx):
            r = out.loc[idx[0]]
            print(f"  {n:22s} grain={r.street_grain:.2f} (Hn={r.street_entropy:.3f})")


if __name__ == "__main__":
    main()
