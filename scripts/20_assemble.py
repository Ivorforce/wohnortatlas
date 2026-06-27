"""Assemble all layer parquets into scores.parquet (normalized 0-1, higher=better)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h3
import numpy as np
import pandas as pd

from wohnen.config import INTERIM, LAYERS
from wohnen.norm import clip01

# Kita crowding is supply-relative: Kita+Grundschule per 1000 locals
# (3 km kernel, matching catchment_pop). Facilities scale ~1:1 with population
# (corr 0.93; per-1000 flat ≈0.75 rural→core), so the discount only bites where
# a place is genuinely under-served for its size, not for being dense.
KITA_SUPPLY_REF = 0.75
KITA_CROWD_MAX = 0.20

# s_green = "Grünes Viertel" — two SEPARATE quantities, noisy-OR'd, so water never
# masquerades as canopy (a treeless riverbank used to read as green as a forest):
#  1. green_land = vegetation as a fraction of the NON-WATER surface (like Wohnform
#     counting only built land). veg / (land · LAND_GREEN_FULL): a fully-forested
#     hex → 1 with NO water needed, and built land dilutes green by sitting in the
#     denominator but not the numerator. LAND_GREEN_FULL<1 = you needn't be literally
#     100 % forest to max. This carries most of the old built-up discrimination, but
#     NOT all: the OSM park_share + canopy overlays are additive on TOP of the
#     WorldCover partition (not in the land denominator), so a sealed Altstadt with a
#     dense street-tree cadastre (Marienplatz canopy→cap) still shows "green land". A
#     MILD superlinear built² penalty (BUILT_BITE, half the pre-split 0.8) plugs that:
#     light on a leafy half-built street, firm on a paved centre.
#  2. water_amb = the hex's own waterfront feel, a saturating term over open water +
#     the OSM narrow-river proxy (waterway_share — moved here from veg; it's water,
#     not land green). WATER_FULL is reached at ~30 % water, so a riverbank counts
#     without being half-submerged.
# Combined s_green = 1 − (1−green_land)·(1−WATER_STANDALONE·water_amb): water spends a
# fixed slice of the REMAINING headroom, so it makes a half-green hex nicer but can't
# over-max a leafy one, and pure water with zero trees floors at WATER_STANDALONE (a
# mid-Rhein hex reads ~0.2, badly-but-not-0). One knob sets both the floor and the
# max bonus — water can never dominate.
# CANOPY_* caps the OSM street-tree supplement (Munich's tree cadastre saturates
# tree_canopy on COUNT in the dense Altstadt): a bounded top-up where WorldCover
# misses canopy between houses, not a term that can dominate on a few street trees.
LAND_GREEN_FULL = 0.85   # green-fraction-of-land that reads as fully green
WATER_FULL = 0.30        # open-water + waterway level that saturates the water term
WATERWAY_W = 0.5         # OSM narrow-river (line proxy) weight inside the water term
WATER_STANDALONE = 0.20  # pure-water floor == max water bonus (the noisy-OR slice)
BUILT_BITE = 0.4         # mild superlinear sealing penalty for overlay leak (see above)
CANOPY_CAP, CANOPY_W = 0.20, 0.5

# Ortsbild = THREE signals — structural FABRIC (hist_density: space-defining built
# heritage — walls/towers/churches/buildings; markers like Stolpersteine/crosses
# are dropped in 10, not just down-weighted) + ART (art_density: public artwork,
# monuments, sculptural memorials) + street GRAIN (organic-vs-grid orientation
# entropy from 17) — each a [0,1] intensity combined by NOISY-OR (1 − Π(1 − wₖ·sₖ)).
# Union, not mean: each gives character on its OWN, extras only ADD, a missing one
# never drags down (an organic Altstadt with no monuments still reads high; a
# Denkmäler-rich town with no art stays high — a weighted mean wrongly tanked them).
# fabric uses a density × per-capita blend (√(d_score · pc_score)):
# density alone lets a big rich Innenstadt dominate and saturates early; per-capita
# (objects/1000 residents in the same 1.5 km kernel = catchment_leisure) lets an
# intact small town rival a city core. Art is density-only — a lively art
# scene is inherently urban, so we WANT it to credit cities, capped lowest
# (CHAR_ART_W). Street grain is gated by built-up (a grown street pattern is
# townscape only where there's a town: 58 % of high-grain cells are curvy RURAL
# roads, gated out via GRAIN_BUILT_*) and is orthogonal to the amenity layers
# (unlike heritage), which is why Ortsbild's overall weight rose off 0.4.
# K = density half-saturation; PC = objects/1000 residents reading as fully
# historic; POP_FLOOR mutes the tiny-population artifact.
CHAR_HIST_K, CHAR_HIST_PC = 22.0, 4.0
CHAR_ART_K = 28.0
CHAR_ART_W, CHAR_GRAIN_W = 0.35, 0.6  # noisy-OR caps: art / grain (hist is primary)
CHAR_POP_FLOOR = 1000.0
GRAIN_BUILT_LO, GRAIN_BUILT_HI = 0.04, 0.20   # built-up gate: rural→0, town core→1


def load(name: str) -> pd.DataFrame | None:
    f = LAYERS / f"{name}.parquet"
    if not f.exists():
        print(f"NOTE: layer missing, skipped: {name}")
        return None
    return pd.read_parquet(f)


def main():
    grid = pd.read_parquet(INTERIM / "grid.parquet")
    out = grid.copy()

    def merge(df):
        nonlocal out
        out = out.merge(df, on="h3", how="left")

    # (Anbindung is not a scores column: the cityness "Alle Branchen" + job-sector field
    # targets ship as M/B/O lazy chunks — reach_cityness.npz / reach_branche.npz, 04f/04h —
    # straight into 22, scored client-side by recomputeAnbindungMBO.)

    # ALL s_freizeit reachability surfaces, derived by 04e_freizeit from the routing
    # caches (reach_centers.npz / reach_spots.npz): reach_activity_* + reach_kultur_*
    # (going-out) and reach_{swim,kino,klettern,golf}_* (point sources).
    # Weight-only client layer; passed through verbatim (web computes s_freizeit).
    freizeit = load("freizeit")
    if freizeit is not None:
        merge(freizeit)

    oepnv = load("oepnv")
    if oepnv is not None:
        merge(oepnv)
        out["s_oepnv"] = out["oepnv_score"].fillna(0)

    ent = load("entertainment")
    if ent is not None:
        # ent_density only — for the 90_validate per-capita supply check; the
        # going-out signal itself ships via reach_* gravity (04e), not from here.
        merge(ent[["h3", "ent_density"]])

    rent = load("rent")
    if rent is not None:
        cols = ["h3", "rent_cal"]
        if "kreis_pay" in rent.columns:  # frontend "% vom Lohn" budget mode
            cols.append("kreis_pay")
        merge(rent[cols])

    green = load("greenness")
    noise = load("noise")
    if green is not None:
        merge(green)
        # district green at the ~500m daily scale: mature trees & parks vs paving;
        # trees dominant, no noise term (own layer). crop gets 0.25 (half of grass):
        # "grünes Viertel" is about what your daily walk LOOKS like, and a field is
        # verdant most of the growing season — clearly below meadow/forest (bare in
        # winter, not walkable) but well above pavement (this is NOT s_nature, where
        # crop barely counts as an outing target). WorldCover trees/grass carry it
        # (it sees real canopy, even overhanging a built street); park_share + a
        # CAPPED tree_canopy recover urban green WorldCover lumps into built-up
        # (small parks, Alleen). All veg terms are hex-fractions, summed then divided
        # by the NON-WATER surface → green-ness OF the land (the built fraction sits
        # in the denominator, not the numerator, so it dilutes green by itself).
        canopy = np.minimum(CANOPY_CAP, CANOPY_W * out["tree_canopy"].fillna(0))
        veg = (1.0 * out["tree_share"].fillna(0) + 0.5 * out["grass_share"].fillna(0)
               + 0.25 * out["crop_share"].fillna(0) + 1.0 * out["park_share"].fillna(0)
               + canopy)
        water = out["water_share"].fillna(0)
        land = (1.0 - water).clip(lower=0.0)
        # guard the divide on (near-)all-water hexes: no land → no land-green
        green_land = clip01(np.where(land > 0.02,
                                     veg / (land * LAND_GREEN_FULL), 0.0))
        # waterfront feel: open water + the OSM narrow-river proxy (a river/Bach you
        # pass — WorldCover misses narrow ones), saturating at WATER_FULL.
        water_amb = clip01((water + WATERWAY_W * out["waterway_share"].fillna(0))
                           / WATER_FULL)
        # noisy-OR: water spends a fixed slice of the remaining headroom, so a leafy
        # hex can't be over-maxed and a treeless one floors at WATER_STANDALONE.
        s = 1.0 - (1.0 - green_land) * (1.0 - WATER_STANDALONE * water_amb)
        # mild built² penalty for the park/canopy overlay leak the land denominator
        # can't see (a sealed centre with a rich street-tree cadastre).
        built = out["builtup_share"].fillna(0)
        out["s_green"] = s * (1 - BUILT_BITE * built ** 2)
    if noise is not None:
        merge(noise)
        out["s_quiet"] = 1 - out["noise_penalty"].fillna(0)

    nature = load("nature")
    if nature is not None:
        merge(nature)
        out["s_nature"] = out["nature_score"].fillna(0)

    char = load("character")
    if char is not None:
        merge(char)  # hist_density/art_density; the score is the density×per-capita
        # blend computed after demographics (needs catchment_leisure)

    streets = load("streets")
    if streets is not None:
        merge(streets)  # street_grain (organic-vs-grid orientation entropy, 17)

    schools = load("schools")
    if schools is not None:
        merge(schools)

    family = load("family")
    if family is not None:
        merge(family)

    demo = load("demographics")
    if demo is not None:
        merge(demo)
        # no s_alive anymore: age is a client-side preference pseudo-layer
        # (s_age, Gaussian around a selectable target on avg_age — like
        # s_density); its nightlife component was redundant with the going-out signal

        # Kita crowding is supply-RELATIVE, not raw headcount (2026-06): Kita/
        # Grundschule supply scales ~1:1 with population (corr 0.93, per-1000
        # flat rural→core, München even above norm), so the old raw-pop discount
        # just penalized dense areas for being dense. r = facilities per 1000
        # locals in the same 3 km kernel as catchment_pop; below KITA_SUPPLY_REF
        # a place is genuinely under-served (building lagged growth) → discount.
        # Resident pop is a clean demand proxy here: Grundschule is catchment-
        # bound and Kita near-home, so no commuters/tourists to miss (unlike
        # leisure). Limitation: demand is children, not all residents.
        if {"kita_supply", "catchment_pop"} <= set(out.columns):
            rf = 1000 * out["kita_supply"] / out["catchment_pop"].replace(0, np.nan)
            crowd = (1 - rf / KITA_SUPPLY_REF).clip(lower=0, upper=1).fillna(0)
            # shipped to the web: the client multiplies the KiTa/Grundschule access
            # score by this Kita-spot-availability factor (the signal the user flagged).
            out["kita_crowd"] = 1 - KITA_CROWD_MAX * crowd

    # Ortsbild score: per-signal intensities -> noisy-OR (see CHAR_*/GRAIN_*).
    # Here so catchment_leisure (demographics) is merged.
    if "hist_density" in out.columns:
        cl = (out["catchment_leisure"].clip(lower=CHAR_POP_FLOOR)
              if "catchment_leisure" in out.columns else None)

        def _intensity(col, K, anchor):
            d = out[col].fillna(0) if col in out.columns else 0.0
            dens = d**2 / (d**2 + K**2)              # saturating density
            if cl is None:
                return dens                          # pop-less fallback: density only
            return np.sqrt(dens * clip01(1000 * d / cl / anchor))  # × per-capita

        # hist (OSM historic) is the primary signal — the Bavaria-only Wikidata
        # Denkmal leg was dropped for the national build (see 10_denkmal).
        s_hist = _intensity("hist_density", CHAR_HIST_K, CHAR_HIST_PC)
        d_art = out["art_density"].fillna(0) if "art_density" in out.columns else 0.0
        s_art = d_art**2 / (d_art**2 + CHAR_ART_K**2)   # density-only (urban OK)
        # street grain (17), gated by built-up: townscape only where there's a town
        grain = out["street_grain"].fillna(0) if "street_grain" in out.columns else 0.0
        built = out["builtup_share"].fillna(0) if "builtup_share" in out.columns else 0.0
        gate = clip01((built - GRAIN_BUILT_LO) / (GRAIN_BUILT_HI - GRAIN_BUILT_LO))
        s_grain = grain * gate
        if "street_grain" in out.columns:
            out["street_grain"] = s_grain   # ship the GATED value so the tooltip
            #   word matches the actual contribution (a curvy rural road is 0, not
            #   "gewachsen" — its grain was gated out of the score)
        out["character_score"] = (1 - (1 - s_hist)
                                  * (1 - CHAR_ART_W * s_art)
                                  * (1 - CHAR_GRAIN_W * s_grain))
        out["s_character"] = out["character_score"].fillna(0)

    bb = load("broadband")
    if bb is not None:
        merge(bb)
        out["s_broadband"] = out["bb_score"].fillna(0)

    clim = load("climate")
    if clim is not None:
        merge(clim)
        out["s_climate"] = out["climate_score"].fillna(0.5)

    flood = load("flood")
    if flood is not None:
        merge(flood)
        out["s_flood"] = out["flood_score"].fillna(1.0)

    # "Leerstand & Verfall": Wohnungsleerstand (Zensus 2022, 1km). No data (rural
    # single-family areas, no multi-dwelling vacancy concept) = neutral 1.0 — silence
    # is not decline. Low-weight, no veto in the web (see 18_vacancy.py).
    vacancy = load("vacancy")
    if vacancy is not None:
        merge(vacancy)
        out["s_vacancy"] = out["vacancy_score"].fillna(1.0)

    # inhabited = enough housing here to plausibly find an offer (Zensus 2022
    # population grid); proxy fallback if demographics layer is missing. Single
    # source for the web's "Unbewohnte Waben" filter and the percentile
    # weighting. 15 inhabitants over a 0.7 km² hex ≈ 8 dwellings — a liquidity
    # floor, not just "non-empty": a 5-dwelling hamlet (pop 10) clears presence
    # but realistically never has an open-market unit. Per-hex pop is the right
    # unit (the map scores where you'd live cell-by-cell, so an empty hex next
    # to a town is still un-rentable — don't pool neighbors). Threshold is also
    # quoted in the web checkbox tooltip copy (index.html).
    if "population" in out.columns:
        inhabited = out["population"].fillna(0) >= 15
    else:
        inhabited = pd.Series(False, index=out.index)
        if "ftth_share" in out.columns:
            inhabited |= out["ftth_share"].notna()
        if "builtup_share" in out.columns:
            inhabited |= out["builtup_share"].fillna(0) >= 0.015
    out["inhabited"] = inhabited
    print(f"inhabited: {inhabited.mean():.1%} of cells")

    score_cols = [c for c in out.columns if c.startswith("s_")]
    out.to_parquet(LAYERS / "scores.parquet", index=False)
    print(f"scores.parquet: {len(out)} cells, layers: {score_cols}")
    print(out[score_cols].describe().round(3).to_string())


if __name__ == "__main__":
    main()
