"""Download all raw datasets into data/raw (idempotent)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests

from wohnen.config import (
    JRC_FLOOD_BASE, JRC_FLOOD_RPS, OA_BATCH_API, OA_JOB_URL, OA_STATES, RAW, URLS,
)
from wohnen.dl import cached_download

FILES = {
    "osm_germany": "germany-latest.osm.pbf",
    "gtfs_germany": "gtfs-germany.zip",
    "zensus_rent_size": "zensus_rent_by_size.zip",
    "zensus_rent": "zensus_rent.zip",
    "zensus_age": "zensus_age.zip",
    "zensus_u18": "zensus_u18.zip",
    "zensus_65plus": "zensus_65plus.zip",
    "zensus_pop": "zensus_pop.zip",
    "zensus_vacancy": "zensus_vacancy.zip",
    "breitband": "breitband_gitterzellen.gpkg.zip",
    "vg250": "vg250.zip",
    "jedeschule": "jedeschule.csv",
}


def download_openaddresses():
    """Fetch the per-state official cadastre address points (03d_geocode input).

    OpenAddresses rotates a numeric job id per source as it reprocesses, so there is
    no static URL: resolve the latest job via the batch API, then pull its
    source.geojson.gz. Snapshot-cached by filename (skip the API call when the file
    is already present) — delete data/raw/openaddresses/ to refresh, like the feeds."""
    oa = RAW / "openaddresses"
    oa.mkdir(parents=True, exist_ok=True)
    for st in OA_STATES:
        dest = oa / f"de_{st}.geojson.gz"
        if dest.exists() and dest.stat().st_size > 0:
            continue
        meta = requests.get(OA_BATCH_API.format(st=st), timeout=60).json()
        if not meta:
            print(f"  openaddresses: no statewide source for {st} — skipped")
            continue
        cached_download(OA_JOB_URL.format(job=meta[0]["job"]), dest, desc=f"oa_{st}")


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    for key, fname in FILES.items():
        cached_download(URLS[key], RAW / fname, desc=fname)
    download_openaddresses()

    # JRC/Copernicus flood-depth rasters (~2.8 GB; resumable, into jrc_flood/)
    jrc = RAW / "jrc_flood"
    jrc.mkdir(parents=True, exist_ok=True)
    jrc_files = [f"Europe_RP{rp}_filled_depth.tif" for rp in JRC_FLOOD_RPS]
    jrc_files.append("Europe_permanent_water_bodies.tif")
    for fname in jrc_files:
        cached_download(f"{JRC_FLOOD_BASE}/{fname}", jrc / fname, desc=fname)

    print("all downloads present:")
    for fname in list(FILES.values()) + [f"jrc_flood/{n}" for n in jrc_files]:
        f = RAW / fname
        print(f"  {fname:42s} {f.stat().st_size / 1e6:10.1f} MB")


if __name__ == "__main__":
    main()
