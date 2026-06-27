"""Offline address matching against official Land cadastre points (OpenAddresses).

JedeSchule ships a Schulart + street address but no coords for ~3.7k schools in the
no-coord Länder (NI 100 %, SH 100 %, SL 49 %, HE/TH ~9 %). These helpers place them by
matching (street, house number) to the official cadastre — authoritative, ~31 m median
accuracy, vs ~150 m–1 km and 43 % coverage for an OSM-address geocode (measured). Used by
03d_geocode. The index is built FILTERED to the target schools' (street, hnr) keys, so RAM
stays bounded by the few thousand schools regardless of how many million addresses we scan.

Disambiguation: a (street, hnr) pair can repeat across towns, so a hit is accepted only when
PLZ matches, else city (exact, then token-overlap for Gemeinde/Ortsteil naming), else when the
pair is unique, else when all candidates cluster in one place. OpenAddresses omits PLZ for NI
and HE, which is why the city fallbacks carry those two states (lifts NI 56 %→96 %).
"""

import re

import numpy as np

_UML = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss",
                      "Ä": "ae", "Ö": "oe", "Ü": "ue"})


def norm_street(s) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower().translate(_UML)
    s = s.replace("strasse", "str").replace("straße", "str").replace("str.", "str")
    s = re.sub(r"\bst\b", "str", s)
    return re.sub(r"[^a-z0-9]", "", s)


def norm_city(s) -> str:
    if not isinstance(s, str):
        return ""
    return re.sub(r"[^a-z0-9]", "", s.lower().translate(_UML))


def house_num(h) -> int | None:
    """Leading integer of a house number ('91-93', '12a' -> 91, 12)."""
    if not isinstance(h, str):
        return None
    m = re.match(r"\s*(\d+)", h)
    return int(m.group(1)) if m else None


def split_address(a) -> tuple[str | None, str | None]:
    """Split a German 'Straße 12a' into (street, housenumber)."""
    if not isinstance(a, str):
        return None, None
    m = re.match(r"^(.*?)[\s,]+(\d+\s*[a-zA-Z]?(?:\s*[-/]\s*\d+\s*[a-zA-Z]?)?)\s*$", a.strip())
    return (m.group(1), m.group(2)) if m else (None, None)


def geocode(idx, plz, city, street, hnr):
    """Resolve one address to (lat, lon, match_type) against a filtered index.

    `idx` maps (norm_street, house_num) -> list of (lat, lon, plz5, norm_city).
    Returns (None, reason) when unplaced. GOOD holds the trusted match types.
    """
    ns, hn = norm_street(street), house_num(hnr)
    if not ns or hn is None:
        return None, None, "no_addr"
    cands = idx.get((ns, hn))
    if not cands:
        return None, None, "miss"
    if plz:
        m = [c for c in cands if c[2] == plz]
        if m:
            return np.mean([c[0] for c in m]), np.mean([c[1] for c in m]), "plz_match"
    if city:
        nc = norm_city(city)
        m = [c for c in cands if c[3] == nc]
        if m:
            return np.mean([c[0] for c in m]), np.mean([c[1] for c in m]), "city_match"
        # Gemeinde vs Ortsteil / name suffixes: accept a containment either way
        m = [c for c in cands if c[3] and (c[3] in nc or nc in c[3])]
        if m:
            return np.mean([c[0] for c in m]), np.mean([c[1] for c in m]), "city_fuzzy"
    if len(cands) == 1:
        return cands[0][0], cands[0][1], "unique"
    las, los = [c[0] for c in cands], [c[1] for c in cands]
    if (max(las) - min(las)) < 0.01 and (max(los) - min(los)) < 0.015:  # ~1 km span
        return float(np.mean(las)), float(np.mean(los)), "clustered"
    return None, None, "ambig"


GOOD = frozenset({"plz_match", "city_match", "city_fuzzy", "unique", "clustered"})
