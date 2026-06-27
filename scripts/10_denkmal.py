"""Character layer raw inputs: heritage density from two OSM sources —
  - historic=*: OSM historic objects, SUBTYPE-WEIGHTED (town/ensemble fabric
    valued high; high-count low-salience markers — rural wayside crosses,
    urban Stolpersteine/plaques — heavily discounted)
  - tourism=artwork: public art (sculptures, murals, installations)
Both are scored & combined (noisy-OR) in 20_assemble (needs population).
OSM objects are read from region-filtered.osm.pbf — its tags-filter keeps
`historic` (and `tourism`) — so this parses the small 2 GB extract, not the full
PBF. node()/way() early-exit on untagged elements (most of the file is geometry
nodes), so we never pay a per-element dict build for nothing.

(The Bavaria-only Wikidata P4244 Denkmalliste leg was dropped for the national
build: P4244 covers only Bavaria, so blending it would bias the map toward
Bavaria. OSM `historic` is the consistent nationwide signal — validated to
separate historic Altstädte from bland suburbs uniformly across Länder.)"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import osmium
import pandas as pd

from wohnen.config import BBOX, INTERIM, LAYERS
from wohnen.h3util import disk_weighted_sum, points_to_cells

# Heritage objects split into THREE streams by what they DO to a townscape, not
# by scale (validated 2026-06: historic=memorial is ~0% structural, so the cut
# lies cleanly along the OSM subtype):
#   STRUCTURAL  space-DEFINING built fabric (walls/buildings/towers that form the
#               enclosure you move through) -> the PRIMARY Ortsbild signal.
#   ART         space-OCCUPYING aesthetic/landmark objects (public artwork,
#               monuments, sculptural memorials) -> secondary art channel in 20.
#   MARKER      commemorative annotations (Stolpersteine — 45% of all memorials —
#               plaques, war-memorial plates, wayside crosses, milestones,
#               boundary stones, archaeological sites) -> DROPPED: they record
#               history without showing it, and by sheer count otherwise
#               saturated inner Berlin/Hamburg and bulk-mapped barrow fields.
# Anything not matched below (unlisted/ambiguous subtypes) is treated as MARKER.
STRUCT_W = {
    "castle": 1.0, "fort": 1.0, "fortress": 1.0, "city_gate": 1.0, "gate": 1.0,
    "citywalls": 1.0, "city_walls": 1.0, "tower": 0.9,
    "ruins": 0.9, "palace": 1.0, "manor": 0.9, "monastery": 1.0, "church": 0.9,
    "chapel": 0.8, "wayside_chapel": 0.7, "aqueduct": 0.9, "lighthouse": 0.8,
    "windmill": 0.7, "watermill": 0.7, "mill": 0.6, "building": 0.7,
}

# ART = tourism=artwork (1.0) + historic=monument (a standalone landmark, 1.0) +
# the genuinely sculptural memorials (statue/bust/obelisk, 0.5). Plain memorial
# stones/plaques/Stolpersteine are NOT art — they fall through to MARKER.
ART_W = 1.0                          # artwork + monument
ART_MEMORIAL_W = 0.5                 # statue / sculpture / bust / obelisk / ...
ART_MEMORIAL = {"statue", "sculpture", "bust", "obelisk", "land_art"}


def in_bbox(lon, lat):
    return BBOX[0] <= lon <= BBOX[2] and BBOX[1] <= lat <= BBOX[3]


class OsmHeritage(osmium.SimpleHandler):
    """Collect (signal, weight, lat, lon) for historic=* and tourism=artwork,
    from nodes and way centroids."""

    def __init__(self):
        super().__init__()
        self.rows = []

    def _kind(self, tags):
        if tags.get("tourism") == "artwork":
            return "art", ART_W
        h = tags.get("historic")
        if h is None:
            return None
        if h == "monument":
            return "art", ART_W
        if h == "memorial":
            if tags.get("memorial", "") in ART_MEMORIAL:
                return "art", ART_MEMORIAL_W
            return None                      # plaques / Stolpersteine / … = MARKER
        w = STRUCT_W.get(h)
        return ("structural", w) if w is not None else None  # unlisted = MARKER

    def node(self, n):
        if not n.tags:  # skip untagged geometry nodes before any per-tag work
            return
        k = self._kind(n.tags)  # osmium TagList supports .get() — no dict build
        if k and in_bbox(n.location.lon, n.location.lat):
            self.rows.append((k[0], k[1], n.location.lat, n.location.lon))

    def way(self, w):
        if not w.tags:
            return
        k = self._kind(w.tags)
        if not k:
            return
        xs, ys = [], []
        for nd in w.nodes:
            try:
                xs.append(nd.location.lon); ys.append(nd.location.lat)
            except Exception:  # node location missing (way clipped at bbox edge)
                pass
        if xs and in_bbox(np.mean(xs), np.mean(ys)):
            self.rows.append((k[0], k[1], float(np.mean(ys)), float(np.mean(xs))))


def weighted_density(rows, signal, cells):
    """1.5 km-kernel density of `signal` objects, each weighted by rows[:,1]."""
    sub = [(w, la, lo) for s, w, la, lo in rows if s == signal]
    if not sub:
        return np.zeros(len(cells))
    w, la, lo = zip(*sub)
    cl = points_to_cells(np.array(la), np.array(lo))
    counts = pd.Series(w, index=cl).groupby(level=0).sum().astype(float)
    return disk_weighted_sum(counts, cells, k=2, scale_km=1.5).values


def main():
    grid = pd.read_parquet(INTERIM / "grid.parquet")
    cells = grid["h3"].tolist()

    h = OsmHeritage()
    h.apply_file(str(INTERIM / "region-filtered.osm.pbf"), locations=True,
                 idx="flex_mem")
    n_struct = sum(1 for s, *_ in h.rows if s == "structural")
    n_art = sum(1 for s, *_ in h.rows if s == "art")

    # Scores are computed in 20_assemble (the density × per-capita blend + noisy-OR
    # needs catchment_leisure). This ships only the raw kernel densities.
    out = pd.DataFrame({
        "h3": cells,
        "hist_density": weighted_density(h.rows, "structural", cells),
        "art_density": weighted_density(h.rows, "art", cells),
    })
    LAYERS.mkdir(parents=True, exist_ok=True)
    out.to_parquet(LAYERS / "character.parquet", index=False)
    print(f"structural: {n_struct} (subtype-weighted)  |  art: {n_art}")
    print(out[["hist_density", "art_density"]].describe().round(2).to_string())


if __name__ == "__main__":
    main()
