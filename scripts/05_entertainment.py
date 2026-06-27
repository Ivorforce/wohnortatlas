"""Entertainment layer: distance-discounted density of nightlife/culture POIs."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from wohnen.config import INTERIM, LAYERS
from wohnen.h3util import disk_weighted_sum, points_to_cells

# cafés/restaurants are ambience, not nightlife — weigh culture/nightlife
# venues higher. Restaurants added 2026-06 (café-weight): village
# Wirtshäuser are tagged restaurant, not pub, and were invisible before.
# 2026-06: everyday gastro (fast_food/ice_cream) at low weight; culture
# enriched with museum/gallery (high) + library/community_centre (the latter
# the main culture signal small towns actually have, so low but nonzero).
WEIGHTS = {
    "bar": 1.0, "pub": 1.0, "biergarten": 1.0, "nightclub": 1.5, "cafe": 0.5,
    "restaurant": 0.5, "fast_food": 0.3, "ice_cream": 0.3,
    "cinema": 2.0, "theatre": 2.0, "arts_centre": 2.0, "music_venue": 2.0,
    "museum": 2.0, "gallery": 1.5, "library": 1.0, "community_centre": 0.5,
}
# breakdown groups (partition all of WEIGHTS) — emitted as separate weighted
# densities for the web detail view. Because disk_weighted_sum is linear, they
# sum EXACTLY to ent_density; the total score is unchanged.
GROUPS = {
    "ent_nightlife": ["bar", "pub", "biergarten", "nightclub"],
    "ent_gastro": ["cafe", "restaurant", "fast_food", "ice_cream"],
    "ent_culture": ["cinema", "theatre", "arts_centre", "music_venue",
                    "museum", "gallery", "library", "community_centre"],
}


def main():
    grid = pd.read_parquet(INTERIM / "grid.parquet")
    pois = pd.read_parquet(INTERIM / "pois.parquet")
    ent = pois[pois["category"] == "entertainment"].copy()
    ent["w"] = ent["subcategory"].map(WEIGHTS).fillna(1.0)
    ent["h3"] = points_to_cells(ent["lat"], ent["lon"])

    counts = ent.groupby("h3")["w"].sum()
    cells = grid["h3"].tolist()
    density = disk_weighted_sum(counts, cells, k=6, scale_km=1.5)

    out = pd.DataFrame({
        "h3": cells,
        "ent_local": grid["h3"].map(counts).fillna(0).values,
        "ent_density": density.values,
    })
    # per-group weighted densities (sum to ent_density). 04e_freizeit reads these
    # (going-out mass for the reach_activity / reach_kultur gravity surfaces).
    for col, subs in GROUPS.items():
        gcounts = ent[ent["subcategory"].isin(subs)].groupby("h3")["w"].sum()
        out[col] = disk_weighted_sum(gcounts, cells, k=6, scale_km=1.5).values
    LAYERS.mkdir(parents=True, exist_ok=True)
    out.to_parquet(LAYERS / "entertainment.parquet", index=False)
    print(f"entertainment: {len(ent)} POIs")
    print(out[["ent_density", *GROUPS]].describe().round(2).to_string())


if __name__ == "__main__":
    main()
