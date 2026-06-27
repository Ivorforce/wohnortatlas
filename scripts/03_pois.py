"""Extract POIs, place nodes, and major-road/rail sample points from the clipped PBF."""

import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import osmium
from osmium.filter import EmptyTagFilter
import pandas as pd

from wohnen.config import BBOX, INTERIM

SRC = INTERIM / "region-filtered.osm.pbf"

# PROFILE_03=1 prints a per-phase timing breakdown (handler pass / road+stream
# sampling / tree-row sampling / green WKT / water area pass) then RETURNS without
# writing — it's diagnostic-only, so it never touches the outputs. Zero overhead
# when unset (the hot-path timers are gated on this flag). For the "03 is slow"
# question: the cost is the geometry sampling, not the POI classification.
PROFILE = bool(os.environ.get("PROFILE_03"))

ENTERTAINMENT = {
    "bar", "pub", "biergarten", "cafe", "nightclub", "cinema", "theatre",
    "arts_centre", "music_venue",
    # village Wirtshäuser are usually tagged restaurant, not pub — without
    # them small-town Gasthaus culture is invisible to the Freizeit gravity
    "restaurant",
    # 2026-06: everyday gastro (döner/imbiss, Eisdielen) at low weight, and
    # culture venues (library/community_centre via amenity; museum/gallery via
    # tourism, handled below) — enriches the thin culture mode + gastro depth
    "fast_food", "ice_cream", "library", "community_centre",
}
# culture venues tagged under tourism= rather than amenity=
ENTERTAINMENT_TOURISM = {"museum", "gallery"}
# sport venues (leisure=*). NO LONGER CONSUMED (2026-06): the sport access layer
# was dropped — generic sport proximity is a population-density echo (it saturates
# wherever people live) with no independent recommendation power; swimming covers
# the one place-specific active-leisure factor. Extraction kept (cheap, harmless)
# so removing it doesn't invalidate the other pois.parquet consumers (05/noise/
# 13/labels). Drop SPORT_LEISURE entirely if you ever rebuild those anyway.
SPORT_LEISURE = {"sports_centre", "sports_hall", "pitch", "fitness_centre",
                 "track", "stadium"}
FAMILY_AMENITY = {"pharmacy", "doctors"}
PLACES = {"city", "town", "village", "suburb", "neighbourhood", "hamlet"}
ROAD_CLASSES = {"motorway", "trunk", "primary", "secondary", "tertiary"}
SAMPLE_M = 100.0
# designated district green WorldCover misses at 10 m (small parks, gardens,
# cemeteries read as built-up). Forest/meadow landuse is left to WorldCover &
# s_nature — only access-style urban green ambience here.
GREEN_LEISURE = {"park", "garden", "village_green"}
GREEN_LANDUSE = {"recreation_ground", "allotments", "cemetery"}
TREE_SAMPLE_M = 12.0  # densify natural=tree_row lines into point trees
# inaccessible green: you can see it but can't enter (military training grounds,
# fenced/private woods) — masked out of s_nature's reachable area in 13.
NOACCESS_NATURAL = {"wood", "scrub", "heath", "grassland", "wetland"}
NOACCESS_LANDUSE = {"forest", "meadow"}


def in_bbox(lon, lat):
    return BBOX[0] <= lon <= BBOX[2] and BBOX[1] <= lat <= BBOX[3]


def green_kind(tags) -> str | None:
    """Return the green-area kind for a way's tags, else None."""
    l = tags.get("leisure")
    if l in GREEN_LEISURE:
        return l
    lu = tags.get("landuse")
    if lu in GREEN_LANDUSE:
        return lu
    return None


def no_access_kind(tags) -> str | None:
    """Green you can't enter: military land, or fenced/private woods & reserves."""
    if tags.get("landuse") == "military" or "military" in tags:
        return "military"
    if tags.get("access") in ("no", "private") and (
            tags.get("landuse") in NOACCESS_LANDUSE
            or tags.get("natural") in NOACCESS_NATURAL
            or tags.get("leisure") == "nature_reserve"):
        return "restricted"
    return None


# s_nature's lake source reads WorldCover satellite water_share, which can't tell a
# tranquil lake from an engineered basin. OSM water types can: grade the lake-source
# quality down for engineered water (13 multiplies satellite water_share by it).
# A reservoir (the Ismaninger Speichersee — real open water, birds, often an NSG, but
# you can't swim/picnic there) keeps SOME nature signal; a stormwater/fish basin less;
# a sewage/wastewater basin almost none. A swim signal (leisure=swimming_area /
# sport=swimming) means it IS a usable outing → rescued to full natural quality (the
# swim itself is scored in s_freizeit; here it's only read as accessibility evidence).
WATER_Q = {"reservoir": 0.45, "basin": 0.25, "wastewater": 0.05}
BASIN_KINDS = {"retention", "infiltration", "detention"}


def water_quality(tags) -> float | None:
    """Reduced lake-source quality for engineered water; None = natural (full 1.0)."""
    w = tags.get("water")
    mm = tags.get("man_made", "")
    # sewage/industrial water: clarifiers, aeration ditches, treatment basins — never
    # an outing regardless of any stray swim tag, so not rescued.
    if (w == "wastewater" or tags.get("reservoir_type") == "sewage"
            or mm == "wastewater_plant" or "clarifier" in mm or "oxidation" in mm):
        return WATER_Q["wastewater"]
    swim = (tags.get("leisure") == "swimming_area"
            or tags.get("sport") in ("swimming", "scuba_diving")
            or tags.get("swimming") == "yes")
    if w == "reservoir":
        return None if swim else WATER_Q["reservoir"]
    if (w == "basin" or tags.get("landuse") == "basin" or mm == "basin"
            or tags.get("basin") in BASIN_KINDS):
        return None if swim else WATER_Q["basin"]
    return None  # natural water / lake / pond / untyped → full quality (default)


def extract_water_quality(src) -> pd.DataFrame:
    """Assemble water AREAS (incl. multipolygon relations — the big reservoirs like
    the Speichersee are relations the way-pass can't reconstruct) and keep only the
    DEMOTED ones as (quality, wkt). Absent = full natural quality.

    `.with_areas()` on the full PBF assembles EVERY multipolygon in the region (all
    forests/buildings/landuse) — ~20 min just to discard the non-water 99%. So first
    shrink the assembler's input with `osmium tags-filter` to the ~0.5 M water areas
    (tags-filter auto-includes referenced nodes/members, so geometry stays complete):
    the whole second pass then runs in ~30 s instead."""
    wktfab = osmium.geom.WKTFactory()
    rows = []
    with tempfile.NamedTemporaryFile(suffix=".osm.pbf", delete=False) as tf:
        water_pbf = tf.name
    try:
        subprocess.run(
            ["osmium", "tags-filter", "--overwrite", "-o", water_pbf, str(src),
             "nwr/water", "nwr/natural=water",
             "nwr/landuse=basin,reservoir", "nwr/man_made=wastewater_plant"],
            check=True, capture_output=True)
        for o in osmium.FileProcessor(water_pbf).with_areas():
            if not isinstance(o, osmium.osm.Area):
                continue
            tags = dict((t.k, t.v) for t in o.tags)
            q = water_quality(tags)
            if q is None:
                continue  # natural/rescued water → no row → defaults to 1.0 in 09
            try:
                wkt = wktfab.create_multipolygon(o)
            except (RuntimeError, ValueError):
                continue
            rows.append((q, wkt))
    finally:
        Path(water_pbf).unlink(missing_ok=True)
    return pd.DataFrame(rows, columns=["quality", "wkt"])


def extract_heath(src) -> pd.DataFrame:
    """Assemble heath AREAS (`natural=heath` / `landuse=heath`) as (multi)polygon WKT.

    WorldCover can't see heath — it labels Calluna/dwarf-shrub heath as grassland, so
    09 burns these OSM polygons to recover `heath_share`. Uses the area assembler (same
    tags-filter→`with_areas()` pattern as the water pass) because the premier heaths —
    the Lüneburger, Lübtheener, Kyritz-Ruppiner Heide — are multipolygon RELATIONS the
    closed-way pass can't reconstruct, and with no WorldCover backstop, missing them
    would lose exactly the bodies that matter. Access (active ranges like Bergen-Hohne)
    is handled downstream by the no_access military mask, so keep every heath polygon."""
    wktfab = osmium.geom.WKTFactory()
    rows = []
    with tempfile.NamedTemporaryFile(suffix=".osm.pbf", delete=False) as tf:
        heath_pbf = tf.name
    try:
        subprocess.run(
            ["osmium", "tags-filter", "--overwrite", "-o", heath_pbf, str(src),
             "nwr/natural=heath", "nwr/landuse=heath"],
            check=True, capture_output=True)
        for o in osmium.FileProcessor(heath_pbf).with_areas():
            if not isinstance(o, osmium.osm.Area):
                continue
            try:
                rows.append((wktfab.create_multipolygon(o),))
            except (RuntimeError, ValueError):
                continue
    finally:
        Path(heath_pbf).unlink(missing_ok=True)
    return pd.DataFrame(rows, columns=["wkt"])


def seg_len_m(lat1, lon1, lat2, lon2):
    kx = 111_320 * math.cos(math.radians((lat1 + lat2) / 2))
    return math.hypot((lon2 - lon1) * kx, (lat2 - lat1) * 111_320)


def classify(tags) -> tuple[str, str] | None:
    """Return (category, subcategory) for POI-like tags, else None."""
    a = tags.get("amenity")
    if a in ENTERTAINMENT:
        return "entertainment", a
    if tags.get("tourism") in ENTERTAINMENT_TOURISM:
        return "entertainment", tags["tourism"]
    if a == "school":
        name = (tags.get("name") or "").lower()
        isced = tags.get("isced:level") or ""
        if "gymnasium" in name or re.search(r"\bgym\b", name):
            return "school", "gymnasium"
        # Standard German school abbreviations (esp. Niedersachsen, where JedeSchule
        # ships no coords so OSM names are the ONLY type signal): KGS/IGS = kooperative/
        # integrierte Gesamtschule, GS = Grundschule. Word-boundary so they don't fire
        # inside other words. (OBS is skipped — it's also Dutch 'openbare basisschool'
        # = a PRIMARY school, ambiguous in border towns.)
        if re.search(r"\b(kgs|igs)\b", name):
            return "school", "gesamtschule"
        # "Grund- und X" combos count as Grundschule (the walk-critical primary
        # need wins over the attached secondary track).
        if "grund- und" in name:
            return "school", "grundschule"
        # Secondary schools are bucketed by the Abschlüsse they deliver, NOT by
        # brand name: most Länder merged the standalone Real-/Hauptschule into a
        # combined secondary track under a regional name (Ober-/Sekundar-/Regel-/
        # Regionale Schule = Haupt+Real; Gemeinschafts-/Stadtteilschule add
        # Abitur). 12 unions each form into gym/real/mittel by its certificates,
        # so e.g. Jena (no "Realschule") still gets Realschul-coverage from its
        # Regelschulen + Gemeinschaftsschulen. (Berufs-/Förderschule stay "other".)
        if "gesamtschule" in name:
            return "school", "gesamtschule"
        if "gemeinschaftsschule" in name:
            return "school", "gemeinschaftsschule"
        if "stadtteilschule" in name:  # Hamburg
            return "school", "stadtteilschule"
        if "werkrealschule" in name:  # BW — before the 'realschule' substring
            return "school", "werkrealschule"
        if "realschule plus" in name:  # RLP — grants Berufsreife + mittlerer → real+mittel
            return "school", "realschule_plus"
        if "realschule" in name:
            return "school", "realschule"
        if "oberschule" in name:  # Sachsen/Niedersachsen/Berlin-BB/Bremen
            return "school", "oberschule"
        if "sekundarschule" in name:  # NRW/Sachsen-Anhalt
            return "school", "sekundarschule"
        if "regelschule" in name:  # Thüringen
            return "school", "regelschule"
        if "regionale schule" in name or "regionalschule" in name:  # MV/RLP
            return "school", "regionale_schule"
        if "wirtschaftsschule" in name:  # Bavaria — mittlerer Abschluss only
            return "school", "wirtschaftsschule"
        # Freie Waldorfschulen run grades 1–13 and grant every Abschluss incl.
        # Abitur → full track. (Montessori is NOT included: that set is a mix of
        # primary-only schools, all-grade schools and even some Kitas, so a
        # blanket all-track credit would over-state secondary coverage.)
        if "waldorf" in name:
            return "school", "waldorf"
        if ("grundschule" in name or "volksschule" in name
                or re.search(r"\bgs\b", name) or "1" in isced.split(";")):
            return "school", "grundschule"
        if "mittelschule" in name or "hauptschule" in name:
            return "school", "mittelschule"
        return "school", "other"
    if a == "pharmacy":
        return "family", "pharmacy"
    if a == "doctors" or tags.get("healthcare") == "doctor":
        spec = tags.get("healthcare:speciality") or ""
        if "paediatric" in spec:
            return "family", "pediatrician"
        return "family", "doctor"
    # dentist: a distinct everyday-care need a GP can't cover (Nahversorgung
    # AND-component, 2026-06). amenity=dentist or healthcare=dentist.
    if a == "dentist" or tags.get("healthcare") == "dentist":
        return "family", "dentist"
    # Two food tiers the web "In der Nähe" scores on (12 ships t_vollsort_min +
    # t_frische_min). VOLLSORTIMENT (the full weekly shop incl. the non-food long
    # tail — shampoo, toilet paper): supermarket (incl. discounters, which OSM tags
    # as supermarket), the small grocery (convenience), the rural general store.
    # FRISCHE (daily fresh-food gap-fillers — capped below a real grocery in the web
    # food score): Bäckerei, Metzgerei, Obst/Gemüse, Hofladen (shop=farm), Bioladen.
    # KIOSK is split out on purpose — a newspaper/cigarette booth is NOT daily food
    # and must not fake food access (merged into convenience, it inflated the old col).
    if tags.get("shop") == "supermarket":
        return "family", "supermarket"
    if tags.get("shop") == "convenience":
        return "family", "convenience"
    if tags.get("shop") == "general":
        return "family", "general"
    if tags.get("shop") == "bakery":
        return "family", "bakery"
    if tags.get("shop") == "butcher":
        return "family", "butcher"
    if tags.get("shop") == "greengrocer":
        return "family", "greengrocer"
    if tags.get("shop") == "farm":
        return "family", "farm"
    if tags.get("shop") == "health_food":
        return "family", "health_food"
    if tags.get("shop") == "kiosk":
        return "family", "kiosk"
    if tags.get("shop") == "chemist":
        return "family", "drugstore"
    # amenity=childcare holds Krippen/Kindertagespflege that mappers split out of
    # kindergarten (+~3.5k nationally). NRW ground-truth: these two tags cover 90%
    # of all Kita-named objects; the rest is noise (bus stops, playgrounds, parking).
    if a in ("kindergarten", "childcare"):
        return "family", "kita"
    if a == "hospital":
        return "family", "hospital"
    if tags.get("shop") == "bicycle":
        return "family", "bikeshop"
    if (a == "public_bath" or tags.get("leisure") == "water_park"
            or (tags.get("leisure") == "swimming_pool"
                and ("name" in tags or tags.get("access") in ("yes", "public")))):
        return "family", "pool"
    # open-water bathing (Badesee): a seasonal substitute for a pool in the
    # Freizeit "schwimmen" source (03b -> 04d reach_swim). water_park above is a
    # built facility -> full "pool" credit; beach/swimming_area is open water.
    if tags.get("leisure") == "swimming_area" or tags.get("natural") == "beach":
        return "family", "badesee"
    if tags.get("leisure") == "playground":
        return "family", "playground"
    # niche s_freizeit leisure sources (2026-06): lumpy, place-specific activities
    # 04d reverse-routes into reach_* gravity surfaces. Distinct from the unused
    # generic SPORT_LEISURE echo. Klettern = indoor Kletterhallen (sport=climbing
    # AT a built facility, NOT natural crags); Golf = courses. Checked BEFORE
    # SPORT_LEISURE so a climbing sports_centre is "klettern", not generic "sport".
    if (tags.get("sport") == "climbing"
            and tags.get("leisure") in ("sports_centre", "sports_hall",
                                        "fitness_centre", "climbing")):
        return "leisure", "klettern"
    if tags.get("leisure") == "golf_course":
        return "leisure", "golf"
    if tags.get("leisure") in SPORT_LEISURE:
        return "sport", tags["leisure"]
    if tags.get("tourism") == "viewpoint":
        return "sight", "viewpoint"
    if tags.get("natural") in ("peak", "waterfall"):
        return "sight", tags["natural"]
    return None


class Handler(osmium.SimpleHandler):
    def __init__(self):
        super().__init__()
        self.pois = []
        self.places = []
        self.roads = []
        self.green = []  # (kind, polygon-WKT) for district green
        self.noaccess = []  # (kind, polygon-WKT) for inaccessible green
        self.trees = []  # (lat, lon) street/park trees
        # PROFILE accumulators (only written when PROFILE) — see the report in main()
        self.t_sample = self.t_trees = self.t_poly = 0.0
        self.n_node = self.n_way = 0

    def _add_poi(self, cat, sub, name, lon, lat):
        if in_bbox(lon, lat):
            self.pois.append((cat, sub, name, lat, lon))

    def node(self, n):
        if PROFILE:
            self.n_node += 1
        tags = dict((t.k, t.v) for t in n.tags)
        if not tags:
            return
        lon, lat = n.location.lon, n.location.lat
        place = tags.get("place")
        if place in PLACES and "name" in tags and in_bbox(lon, lat):
            self.places.append((place, tags["name"], lat, lon))
            return
        if tags.get("natural") == "tree":
            if in_bbox(lon, lat):
                self.trees.append((lat, lon))
            return
        c = classify(tags)
        if c:
            self._add_poi(c[0], c[1], tags.get("name", ""), lon, lat)

    def way(self, w):
        if PROFILE:
            self.n_way += 1
        tags = dict((t.k, t.v) for t in w.tags)
        if not tags:
            return
        cls = None
        hw = tags.get("highway")
        if hw in ROAD_CLASSES:
            cls = hw
        elif tags.get("railway") == "rail" and tags.get("usage") in ("main", "branch"):
            cls = "rail"
        elif tags.get("waterway") == "river":
            cls = "river"
        elif tags.get("waterway") == "stream":
            cls = "stream"  # Bäche: small water — modest s_nature + green credit
        if cls:
            if PROFILE:
                t0 = perf_counter(); self._sample_way(w, cls); self.t_sample += perf_counter() - t0
            else:
                self._sample_way(w, cls)
            return
        if tags.get("natural") == "tree_row":
            if PROFILE:
                t0 = perf_counter(); self._sample_trees(w); self.t_trees += perf_counter() - t0
            else:
                self._sample_trees(w)
            return
        gk = green_kind(tags)
        if gk:
            t0 = perf_counter() if PROFILE else 0
            wkt = self._poly_wkt(w)
            if PROFILE:
                self.t_poly += perf_counter() - t0
            if wkt:
                self.green.append((gk, wkt))
            return
        nak = no_access_kind(tags)
        if nak:
            t0 = perf_counter() if PROFILE else 0
            wkt = self._poly_wkt(w)
            if PROFILE:
                self.t_poly += perf_counter() - t0
            if wkt:
                self.noaccess.append((nak, wkt))
            return
        c = classify(tags)
        if c:
            pts = [(nd.location.lon, nd.location.lat) for nd in w.nodes
                   if nd.location.valid()]
            if pts:
                lon = sum(p[0] for p in pts) / len(pts)
                lat = sum(p[1] for p in pts) / len(pts)
                self._add_poi(c[0], c[1], tags.get("name", ""), lon, lat)

    def _sample_way(self, w, cls):
        locs = [(nd.location.lat, nd.location.lon) for nd in w.nodes
                if nd.location.valid()]
        carry = 0.0
        for (la1, lo1), (la2, lo2) in zip(locs, locs[1:]):
            d = seg_len_m(la1, lo1, la2, lo2)
            if d <= 0:
                continue
            pos = carry
            while pos < d:
                f = pos / d
                la, lo = la1 + (la2 - la1) * f, lo1 + (lo2 - lo1) * f
                if in_bbox(lo, la):
                    self.roads.append((cls, la, lo))
                pos += SAMPLE_M
            carry = pos - d

    def _poly_wkt(self, w):
        """Reconstruct a closed-way polygon as WKT (lon lat), or None."""
        pts = [(nd.location.lon, nd.location.lat) for nd in w.nodes
               if nd.location.valid()]
        if len(pts) < 3 or not any(in_bbox(lo, la) for lo, la in pts):
            return None
        if pts[0] != pts[-1]:
            pts.append(pts[0])  # close the ring for a valid POLYGON
        ring = ", ".join(f"{lo:.7f} {la:.7f}" for lo, la in pts)
        return f"POLYGON (({ring}))"

    def _sample_trees(self, w):
        locs = [(nd.location.lat, nd.location.lon) for nd in w.nodes
                if nd.location.valid()]
        carry = 0.0
        for (la1, lo1), (la2, lo2) in zip(locs, locs[1:]):
            d = seg_len_m(la1, lo1, la2, lo2)
            if d <= 0:
                continue
            pos = carry
            while pos < d:
                f = pos / d
                la, lo = la1 + (la2 - la1) * f, lo1 + (lo2 - lo1) * f
                if in_bbox(lo, la):
                    self.trees.append((la, lo))
                pos += TREE_SAMPLE_M
            carry = pos - d


def main():
    h = Handler()
    t0 = perf_counter()
    # Skip the ~150 M untagged geometry-only nodes (road/river vertices that tags-filter
    # keeps for way geometry) at the C level: each one otherwise pays a Python callback
    # only to hit `if not tags: return` — that was 98 % of the runtime (PROFILE_03).
    # EmptyTagFilter discards exactly those zero-tag objects (== the existing early-return,
    # moved to C); with_locations() caches ALL node locations BEFORE the filter, so way
    # centroids / road sampling / green polygons still resolve.
    fp = (osmium.FileProcessor(str(SRC))
          .with_locations("flex_mem")
          .with_filter(EmptyTagFilter()))
    for o in fp:
        if isinstance(o, osmium.osm.Node):
            h.node(o)
        elif isinstance(o, osmium.osm.Way):
            h.way(o)
    t_handler = perf_counter() - t0

    pois = pd.DataFrame(h.pois, columns=["category", "subcategory", "name", "lat", "lon"])
    places = pd.DataFrame(h.places, columns=["place", "name", "lat", "lon"])
    roads = pd.DataFrame(h.roads, columns=["cls", "lat", "lon"])

    green = pd.DataFrame(h.green, columns=["kind", "wkt"])
    noaccess = pd.DataFrame(h.noaccess, columns=["kind", "wkt"])
    trees = pd.DataFrame(h.trees, columns=["lat", "lon"])

    # engineered-water polygons need assembled AREAS (relations), so a second pass
    t0 = perf_counter()
    water_q = extract_water_quality(SRC)
    t_water = perf_counter() - t0

    # heath areas (incl. relations) — same area-assembler pass; WorldCover misses heath
    heath = extract_heath(SRC)

    if PROFILE:
        rem = t_handler - h.t_sample - h.t_trees - h.t_poly
        print("=== 03_pois timing (PROFILE_03) ===")
        print(f"  handler pass (1 PBF scan):  {t_handler:7.1f}s  "
              f"(nodes {h.n_node:,}, ways {h.n_way:,})")
        print(f"    road/stream sampling:     {h.t_sample:7.1f}s  ({len(roads):,} pts)")
        print(f"    tree-row sampling:        {h.t_trees:7.1f}s")
        print(f"    green/no-access WKT:      {h.t_poly:7.1f}s")
        print(f"    rest (tag-dict/classify/tree-nodes/iter): {rem:7.1f}s "
              f"({len(trees):,} tree nodes)")
        print(f"  water area pass (2nd scan): {t_water:7.1f}s")
        return  # PROFILE is diagnostic-only: skip the writes (don't touch the outputs)

    pois.to_parquet(INTERIM / "pois.parquet", index=False)
    places.to_parquet(INTERIM / "places.parquet", index=False)
    roads.to_parquet(INTERIM / "roads.parquet", index=False)
    green.to_parquet(INTERIM / "green_areas.parquet", index=False)
    noaccess.to_parquet(INTERIM / "no_access.parquet", index=False)
    water_q.to_parquet(INTERIM / "water_quality.parquet", index=False)
    heath.to_parquet(INTERIM / "heath_areas.parquet", index=False)
    trees.to_parquet(INTERIM / "trees.parquet", index=False)

    print(f"pois: {len(pois)}")
    print(pois.groupby(["category", "subcategory"]).size().to_string())
    print(f"places: {len(places)} ({places['place'].value_counts().to_dict()})")
    print(f"road/rail samples: {len(roads)} ({roads['cls'].value_counts().to_dict()})")
    print(f"green areas: {len(green)} ({green['kind'].value_counts().to_dict() if len(green) else {}})")
    print(f"no-access areas: {len(noaccess)} ({noaccess['kind'].value_counts().to_dict() if len(noaccess) else {}})")
    print(f"engineered water: {len(water_q)} ({water_q['quality'].value_counts().to_dict() if len(water_q) else {}})")
    print(f"heath areas: {len(heath)}")
    print(f"trees: {len(trees)}")


if __name__ == "__main__":
    main()
