"""Shared school-type bucketing: OSM subcategory + JedeSchule school_type → the four
reachability tracks (grund/gym/real/mittel). ONE Abschluss model, two sources.

Tracks are NOT substitutes (Schülerbeförderung funds transport only to the nearest
school OF THE CHOSEN FORM), so a school is a member of every track whose Abschluss it
delivers — combined forms join several. 03c_schools.py takes EVERY JedeSchule school per
track (authoritative, located) and adds an OSM school of that track only where JedeSchule
has none nearby — so OSM fills the no-coord states and the schools JedeSchule misses.
"""

TRACKS = ("grund", "gym", "real", "mittel")

# OSM subcategory (from 03_pois.classify) → tracks. FULL = combined forms that also
# reach the Abitur (like Gesamtschule); HAUPT_REAL = the merged Haupt+Real secondary
# many Länder use in place of a standalone Realschule (e.g. Thüringen's Regelschule).
_OSM_FULL = {"gesamtschule", "gemeinschaftsschule", "stadtteilschule", "waldorf"}
_OSM_HAUPT_REAL = {"oberschule", "sekundarschule", "regelschule",
                   "regionale_schule", "werkrealschule", "realschule_plus"}


def osm_buckets(subcat) -> frozenset:
    """Tracks an OSM school (by its 03_pois subcategory) counts toward."""
    s = subcat if isinstance(subcat, str) else ""
    out = set()
    if s == "grundschule":
        out.add("grund")
    if s == "gymnasium" or s in _OSM_FULL:
        out.add("gym")
    if (s in {"realschule", "wirtschaftsschule"}
            or s in _OSM_FULL or s in _OSM_HAUPT_REAL):
        out.add("real")
    if s == "mittelschule" or s in _OSM_FULL or s in _OSM_HAUPT_REAL:
        out.add("mittel")
    return frozenset(out)


# JedeSchule school_type (277 per-state values incl. INSPIRE English codes) → tracks.
# Non-allgemeinbildend (Berufs/Förder/admin) AND ambiguous codes (bare "education",
# "lowerSecondary…" with no track) return EMPTY: those rows neither enrich an OSM
# school nor get added — OSM's own name-based type stands instead (safe default).
# NB: skip on "förderschule"/"förderzentr"/"schwerpunkt", NOT bare "förder" — the
# latter also matches "Sprachförderung" on otherwise-normal Grundschulen.
_JS_SKIP = ("beruf", "fachschule", "fachober", "fachakad", "kolleg", "studiensem",
            "schulaufsicht", "administ", "verwaltung", "landwirt", "weiterbildung",
            "förderschule", "foerderschule", "förderzentr", "foerderzentr",
            "sonderp", "special", "schwerpunkt", "klinik",
            "abendgymn", "orientierungsstufe", "seminar")


def js_buckets(school_type) -> frozenset:
    """Tracks a JedeSchule record delivers; empty = skip (non-school / ambiguous)."""
    s = school_type.lower() if isinstance(school_type, str) else ""
    if not s.strip() or any(k in s for k in _JS_SKIP):
        return frozenset()
    out = set()
    if ("grundschule" in s or "grund- und" in s or "primaryeducation" in s
            or "volksschule" in s):
        out.add("grund")
    if ("gesamtschule" in s or "gemeinschaftsschule" in s
            or "stadtteilschule" in s or "waldorf" in s):
        out |= {"gym", "real", "mittel"}            # full track (→ Abitur)
    if "gymnasium" in s or "gymnasien" in s or "uppersecondary" in s:
        out.add("gym")
    if ("oberschule" in s or "sekundarschule" in s or "regelschule" in s
            or "regionale schule" in s or "regionalschule" in s
            or "werkrealschule" in s or "realschule plus" in s
            or "haupt- und realschule" in s
            or "lowersecondaryeducation" in s or "lowersecondaryeduction" in s):
        out |= {"real", "mittel"}                   # merged Haupt+Real secondary
    if "realschule" in s or "wirtschaftsschule" in s:
        out.add("real")
    if "hauptschule" in s or "mittelschule" in s:
        out.add("mittel")
    return frozenset(out)
