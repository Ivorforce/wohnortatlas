"""Shared configuration: study area, grid resolution, destinations, score anchors."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
LAYERS = DATA / "layers"
WEB = ROOT / "web"
CACHE = RAW / "cache"

# lon_min, lat_min, lon_max, lat_max — Germany (mainland incl. North/Baltic islands)
BBOX = (5.8, 47.2, 15.1, 55.1)
H3_RES = 8


def in_bbox(lon, lat):
    return BBOX[0] <= lon <= BBOX[2] and BBOX[1] <= lat <= BBOX[3]

BIKE_KMH = 15.0  # school runs (kids)
WALK_KMH = 4.5
DETOUR_FACTOR = 1.3
# R5 car routing is free-flow (no congestion model), so the routed car minutes
# (04c/04d) bake in a realistic door-to-door adjustment: AM-peak congestion, plus a
# fixed parking / last-leg overhead at the destination (mirrors the web's
# MODE_OVH.car = 6). Keeps the car honest against transit times that already
# include waiting. The web adds only the per-mode preference penalty on top.
CAR_CONGESTION = 1.3
CAR_PARK_MIN = 6.0
# RMS distance from a uniform in-cell point to the res-8 cell center
# (exact polygon moments; 337-340 m across the bbox). Two uses:
#  - 13_nature.py / h3util: single-point de-bias E[dist to POI] ~= sqrt(d^2 + RMS^2)
#    (cell-to-cell doubles it, both endpoints spread).
#  - 12_schools_family.py: the target RMS for the 6 in-cell sample points
#    (cell_samples) — multi-point nearest captures that dense supply puts a
#    resident's nearest option closer than the single nearest-to-centre.
CELL_RMS_M = 338.0
TRAVEL_SENTINEL_MIN = 120

URLS = {
    "osm_germany": "https://download.geofabrik.de/europe/germany-latest.osm.pbf",
    "gtfs_germany": "https://download.gtfs.de/germany/free/latest.zip",
    "zensus_rent_size": "https://www.destatis.de/static/DE/zensus/gitterdaten/Durchschnittliche_Nettokaltmiete_nach_Gebaeudealter_und_Wohnungsgroe%C3%9Fe.zip",
    "zensus_rent": "https://www.destatis.de/static/DE/zensus/gitterdaten/Zensus2022_Durchschn_Nettokaltmiete.zip",
    "zensus_age": "https://www.destatis.de/static/DE/zensus/gitterdaten/Durchschnittsalter_in_Gitterzellen.zip",
    # age-band shares per cell — the life-stage signal the mean can't carry (kids vs
    # students, both pull avg_age down; under-18 share is immune to the WG household count).
    "zensus_u18": "https://www.destatis.de/static/DE/zensus/gitterdaten/Anteil_unter_18-jaehrige_in_Gitterzellen.zip",
    "zensus_65plus": "https://www.destatis.de/static/DE/zensus/gitterdaten/Anteil_ab_65-jaehrige_in_Gitterzellen.zip",
    "zensus_pop": "https://www.destatis.de/static/DE/zensus/gitterdaten/Zensus2022_Bevoelkerungszahl.zip",
    # Wohnungsleerstand (Leerstandsquote, alle Wohnungen) per grid cell — the
    # "Leerstand & Verfall" signal. 100m values are small-denominator noise
    # (median 23 %, P95 100 %); 18_vacancy reads the 1km grid, where clean cells
    # match Germany's real ~5 % vacancy. dl-de/by-2-0 (same as the other Zensus grids).
    "zensus_vacancy": "https://www.destatis.de/static/DE/zensus/gitterdaten/Leerstandsquote_in_Gitterzellen.zip",
    "breitband": "https://data.bundesnetzagentur.de/Bundesnetzagentur/GIGA/DE/Breitbandatlas/Downloads/Versorgungsdaten_Gitterzellen_Stand_20251231_gpkg.zip",
    "vg250": "https://daten.gdz.bkg.bund.de/produkte/vg/vg250_ebenen_0101/aktuell/vg250_01-01.utm32s.shape.ebenen.zip",
    # JedeSchule (Code for Germany / Datenschule) — all 16 Länder-Schulverzeichnisse
    # merged, CC0. Authoritative Schulart + coords; enriches the OSM school typing
    # (03c_schools.py). The jedeschule.de static page is stale 2017 — use the live
    # codefor.de export. Snapshot like GTFS: delete the file + re-download to refresh.
    "jedeschule": "https://jedeschule.codefor.de/csv-data/latest.csv",
}

# OpenAddresses — official Land cadastre address points (Hauskoordinaten),
# aggregated into one schema (line-delimited GeoJSON, WGS84, house-number level).
# Used by 03d_geocode to place the JedeSchule schools that ship no coords (the
# no-coord Länder). Only the states with a current coord gap are fetched — the
# rest ship full JedeSchule coords, so their ~25 M addresses would be dead weight
# (and the index would balloon RAM). Extend this list if another state regresses.
# Job ids rotate as OA reprocesses; 00_download resolves the latest via the batch
# API. Licences differ per state (CC-BY-4.0 NI/SH, DL-DE/BY-2.0 SL/TH, DL-DE/Zero
# HE) — attributed in web/method.html. Snapshot like GTFS: delete to refresh.
OA_STATES = ["ni", "sh", "sl", "he", "th"]
OA_BATCH_API = "https://batch.openaddresses.io/api/data?source=de%2F{st}%2Fstatewide"
OA_JOB_URL = "https://v2.openaddresses.io/batch-prod/job/{job}/source.geojson.gz"

# BBSR INKAR open data (dl-de/by-2-0). Gruppe 58 = Wiedervermietungsmieten
# inserierter Wohnungen (Angebotsmieten), €/m² net cold, Kreis level,
# rounded to whole euros in the open release.
INKAR_API = "https://www.inkar.de"
INKAR_RENT_GRUPPE = "58"

# GENESIS / Regionaldatenbank (regionalstatistik.de) webservice — open data,
# redistribution permitted with source citation (≈ dl-de/by-2-0). POST-only REST;
# credentials go in the HTTP HEADER (username+password of a free account, or a
# personal token in the username field). No batch needed for this table (~70 KB,
# returned synchronously). Table 52111-07-01-4 = Unternehmensregister "Abhängig
# Beschäftigte der Niederlassungen (B-N, P-S) nach ausgewählten Wirtschaftsabschnitten,
# Kreise" — workplace employment by WZ-Abschnitt; the Branche (sector) opportunity mass.
GENESIS_API = "https://www.regionalstatistik.de/genesisws/rest/2020"
JOBS_TABLE = "52111-07-01-4"
# Branche buckets: client key -> WZ-2008 Abschnitt letters summed into it. Public
# administration (O) + agriculture (A) are outside the Unternehmensregister (B-N, P-S),
# so there is no such bucket; "Industrie" is residual-filled where C is suppressed.
JOBS_BUCKETS = {
    "industrie": ["C", "B", "D", "E"],
    "it": ["J"],
    "gesundheit": ["Q"],
    "bildung": ["P"],
    "handel": ["G", "H"],
    "gastgewerbe": ["I"],
    "finanz_dienste": ["K", "L", "M", "N"],
    "bau": ["F"],
    "kunst_medien": ["R", "S"],
}
JOBS_BUCKET_LABELS = {
    "industrie": "Industrie & Produktion",
    "it": "IT & Kommunikation",
    "gesundheit": "Gesundheit & Soziales",
    "bildung": "Bildung & Erziehung",
    "handel": "Handel & Logistik",
    "gastgewerbe": "Gastgewerbe & Tourismus",
    "finanz_dienste": "Finanz- & Unternehmensdienste",
    "bau": "Bau & Handwerk",
    "kunst_medien": "Kunst, Medien & Kultur",
}
EBA_WFS = "https://geoinformation.eisenbahn-bundesamt.de/wfs/eba/services/wfs"
# UBA federal Umgebungslärm (END/Lärmkartierung 2022) WMS — Germany-wide, one
# ArcGIS MapServer with numeric layer IDs (road Lden 30+27, aircraft 16+13,
# rail 23+20). Same 5-band classified Lden palette as the per-Land services
# (verified byte-for-byte), so the band decoder is unchanged. dl-de/by-2-0.
UBA_NOISE_WMS = "https://datahub.uba.de/server/services/VeLa/LK/MapServer/WMSServer"
WORLDCOVER_S3 = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map"
# JRC/Copernicus river-flood hazard maps (CEMS-EFAS; Dottori et al. 2022, ESSD).
# EU-wide inundation-depth GeoTIFFs (metres) per return period, 100 m, CC-BY-4.0
# — the national replacement for the Bavaria-only LfU flood WMS, giving an
# interpretable frequency x severity risk (Σ P(rp)·depth). Filenames:
# Europe_RP{rp}_filled_depth.tif + Europe_permanent_water_bodies.tif.
JRC_FLOOD_BASE = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/CEMS-EFAS/flood_hazard"
JRC_FLOOD_RPS = [10, 20, 30, 40, 50, 75, 100, 200, 500]  # return periods (years)

USER_AGENT = "wohnen-research/0.1 (personal housing analysis; lukas.tenbrink@gmail.com)"
