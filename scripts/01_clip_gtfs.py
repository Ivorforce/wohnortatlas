"""Clip the Germany GTFS feed to the study bbox; pick & persist a departure datetime."""

import datetime as dt
import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from wohnen.config import BBOX, INTERIM, RAW

SRC = RAW / "gtfs-germany.zip"
DST = INTERIM / "gtfs_region.zip"


def clip():
    if DST.exists() and DST.stat().st_size > 0:
        print(f"using existing {DST}")
        return
    # despite the CLI docs claiming lat-first, the implementation does
    # shapely.box(*bounds) against x=lon geometry — so pass lon-first
    bounds = f"[{BBOX[0]},{BBOX[1]},{BBOX[2]},{BBOX[3]}]"
    subprocess.run(
        ["poetry", "run", "gtfs-utils", "filter", str(SRC),
         "-o", str(DST), "-b", bounds, "--overwrite"],
        check=True,
    )


def read_csv_from_zip(zf: zipfile.ZipFile, name: str) -> pd.DataFrame | None:
    if name not in zf.namelist():
        return None
    return pd.read_csv(io.BytesIO(zf.read(name)), dtype=str)


def pick_departure() -> dt.datetime:
    """Next Tue/Wed/Thu 08:00 that lies within the feed's calendar validity."""
    def gtfs_dates(s: pd.Series) -> pd.Series:
        return pd.to_datetime(s, format="%Y%m%d", errors="coerce").dropna()

    with zipfile.ZipFile(DST) as zf:
        cal = read_csv_from_zip(zf, "calendar.txt")
        starts = gtfs_dates(cal["start_date"]) if cal is not None else pd.Series(dtype="datetime64[ns]")
        ends = gtfs_dates(cal["end_date"]) if cal is not None else pd.Series(dtype="datetime64[ns]")
        if not len(starts):
            cd = read_csv_from_zip(zf, "calendar_dates.txt")
            starts = ends = gtfs_dates(cd["date"])
        start, end = starts.min(), ends.max()

    day = max(dt.date.today() + dt.timedelta(days=1), start.date())
    while day.weekday() not in (1, 2, 3):
        day += dt.timedelta(days=1)
    if day > end.date():
        raise SystemExit(
            f"feed validity {start.date()}..{end.date()} has no usable weekday "
            "left — re-download GTFS (rm data/raw/gtfs-germany.zip; make download)"
        )
    departure = dt.datetime.combine(day, dt.time(8, 0))
    print(f"feed valid {start.date()}..{end.date()}, departure = {departure}")
    return departure


def main():
    INTERIM.mkdir(parents=True, exist_ok=True)
    clip()
    departure = pick_departure()
    (INTERIM / "departure.json").write_text(
        json.dumps({"departure": departure.isoformat()})
    )
    with zipfile.ZipFile(DST) as zf:
        n_trips = len(read_csv_from_zip(zf, "trips.txt"))
        n_stops = len(read_csv_from_zip(zf, "stops.txt"))
    size_mb = DST.stat().st_size / 1e6
    print(f"clipped GTFS: {n_trips} trips, {n_stops} stops, {size_mb:.0f} MB -> {DST}")
    assert size_mb < 500, "R5 caps GTFS at 500 MB"


if __name__ == "__main__":
    main()
