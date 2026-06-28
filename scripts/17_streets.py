"""Street-network 'grown-ness' for Ortsbild: orientation entropy × step-sharpness.

Each street segment's compass bearing (length-weighted, two-way) goes into a
36-bin histogram per cell; aggregated over the cell + 1-ring, its normalized
Shannon entropy Hn ∈ [0,1] measures how ORDERED the street grid is:
  LOW  Hn = gridded / planned (Manhattan, postwar Rasterstadt) — bearings pile
            up on a few axes;
  HIGH Hn = organic / grown (medieval Altstadt) — bearings point every way.

Entropy ALONE cannot tell a grown Altstadt from a planned layout whose streets
merely point many ways: a curvilinear slab estate on sweeping arcs (Gropiusstadt)
or a multi-axis Reißbrett (Tübingen Französisches Viertel) both fill the bearing
histogram, so both read high. The geometric tell is the STEP at each vertex:
organic streets bend sharply and irregularly (mean |turn| ≈ 8-12°), while a
designed arc or grid bends gently/straight (≈ 5° or less). So Hn is discounted by
a step-sharpness factor — a cell needs varied bearings AND sharp irregular
stepping to read as grown; smooth regular layouts keep only TURN_FLOOR of it.

`street_grain` = Hn stretched between empirical anchors (a real grid ≈ HN_LO → 0,
a medieval core ≈ HN_HI → 1), discounted by step-sharpness, and GATED to
street-rich cells (entropy is noise where there's no fabric, and Ortsbild is moot
in open country) → 0 elsewhere. Dead-end share is not used: German Trabantenstädte
are connected apartment estates (not cul-de-sac sprawl) and Altstädte have
Sackgassen, so it ranks backwards.

Reads region-filtered.osm.pbf (its tags-filter keeps nwr/highway = ALL highway
classes, so the small 2 GB extract suffices); no routing.
Output: data/layers/streets.parquet (h3, street_grain, street_entropy,
street_turn, street_len_m).
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
TURN_LO, TURN_HI = 5.0, 8.0   # mean |turn| per vertex (deg): a smooth arc/grid ≤5 → the
#                               floor, a sharply-stepping organic core ≥8 → full grain
TURN_FLOOR = 0.25             # grain a maximally-smooth (regular-arc / grid) cell keeps
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


def _wrap180(d):
    return (d + 180) % 360 - 180


class Streets(osmium.SimpleHandler):
    def __init__(self):
        super().__init__()
        self.hist = defaultdict(lambda: np.zeros(NBINS))  # cell -> length per bin
        self.turn_abs = defaultdict(float)   # cell -> Σ|turn| at its street vertices
        self.turn_n = defaultdict(float)     # cell -> vertex count

    def way(self, w):
        if w.tags.get("highway") not in STREET:
            return
        pts = [(nd.location.lon, nd.location.lat) for nd in w.nodes
               if nd.location.valid()]
        prev_b = None
        for (lo1, la1), (lo2, la2) in zip(pts, pts[1:]):
            L = _metres(lo1, la1, lo2, la2)
            if L <= 0:
                prev_b = None
                continue
            b = _bearing(lo1, la1, lo2, la2)
            hb = self.hist[h3.latlng_to_cell((la1 + la2) / 2, (lo1 + lo2) / 2, 8)]
            hb[int(b // BIN) % NBINS] += L
            hb[int((b + 180) // BIN) % NBINS] += L   # undirected
            if prev_b is not None:                   # turn at the shared vertex (lo1,la1)
                vc = h3.latlng_to_cell(la1, lo1, 8)
                self.turn_abs[vc] += abs(_wrap180(b - prev_b))
                self.turn_n[vc] += 1.0
            prev_b = b


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
    ta = np.zeros(len(grid))
    tn = np.zeros(len(grid))
    for i, c in enumerate(grid["h3"]):
        hist = np.zeros(NBINS)
        sturn = nturn = 0.0
        for nb in h3.grid_disk(c, DISK_K):
            hist += h.hist.get(nb, 0.0)
            sturn += h.turn_abs.get(nb, 0.0)
            nturn += h.turn_n.get(nb, 0.0)
        length[i] = hist.sum()
        ent[i] = _entropy(hist)
        ta[i] = sturn
        tn[i] = nturn

    mean_turn = np.divide(ta, tn, out=np.zeros_like(ta), where=tn > 0)
    # entropy grain, discounted by step-sharpness so smooth regular layouts
    # (arcs, multi-axis grids) keep only the floor of their entropy grain
    g_ent = clip01((ent - HN_LO) / (HN_HI - HN_LO))
    sharp = np.clip((mean_turn - TURN_LO) / (TURN_HI - TURN_LO), 0.0, 1.0)
    grain = g_ent * (TURN_FLOOR + (1.0 - TURN_FLOOR) * sharp)
    grain[length < MIN_LEN_M] = 0.0          # no fabric -> no street-grain character
    grain = np.nan_to_num(grain, nan=0.0)

    out = pd.DataFrame({
        "h3": grid["h3"],
        "street_grain": grain,
        "street_entropy": ent,
        "street_turn": mean_turn.round(2),
        "street_len_m": length,
    })
    LAYERS.mkdir(parents=True, exist_ok=True)
    out.to_parquet(LAYERS / "streets.parquet", index=False)

    gated = grain[length >= MIN_LEN_M]
    print(f"street_grain: {len(gated)} fabric cells; "
          f"p10/50/90 = {np.percentile(gated, [10, 50, 90]).round(2).tolist()}")
    for n, la, lo in [("Wasserburg organic", 48.0590, 12.2290),
                      ("Eichstätt Altstadt", 48.8916, 11.1844),
                      ("Rothenburg Altstadt", 49.3779, 10.1787),
                      ("Gropiusstadt (arc)", 52.4256, 13.4620),
                      ("Tübingen Franz.V.", 48.5120, 9.0630),
                      ("München Maxvorstadt", 48.1510, 11.5660)]:
        idx = grid.index[grid["h3"] == h3.latlng_to_cell(la, lo, 8)]
        if len(idx):
            r = out.loc[idx[0]]
            print(f"  {n:22s} grain={r.street_grain:.2f} "
                  f"(Hn={r.street_entropy:.3f} turn={r.street_turn:.1f}°)")


if __name__ == "__main__":
    main()
