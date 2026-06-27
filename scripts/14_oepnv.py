"""Local ÖPNV quality: weekday departures × line diversity near each hex.

Pure timetable aggregation from the clipped GTFS (no routing): for each stop,
count trips active on the reference weekday and the distinct routes (rail/
subway/tram weighted 2×, bus 1×); aggregate per hex over grid_disk(1) ≈ 900 m.
"""

import datetime as dt
import io
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h3
import numpy as np
import pandas as pd

from wohnen.config import INTERIM, LAYERS

GTFS = INTERIM / "gtfs_region.zip"
RAIL_TYPES = {0, 1, 2}  # tram, subway, rail get double weight


def read(zf, name, usecols=None):
    if name not in zf.namelist():
        return None
    return pd.read_csv(io.BytesIO(zf.read(name)), usecols=usecols, dtype=str)


def active_services(zf, day: dt.date) -> set:
    weekday_cols = ["monday", "tuesday", "wednesday", "thursday", "friday",
                    "saturday", "sunday"]
    active = set()
    cal = read(zf, "calendar.txt")
    if cal is not None:
        col = weekday_cols[day.weekday()]
        d = int(day.strftime("%Y%m%d"))
        m = ((cal[col] == "1")
             & (pd.to_numeric(cal["start_date"]) <= d)
             & (pd.to_numeric(cal["end_date"]) >= d))
        active = set(cal.loc[m, "service_id"])
    cd = read(zf, "calendar_dates.txt")
    if cd is not None:
        cd = cd[cd["date"] == day.strftime("%Y%m%d")]
        active |= set(cd.loc[cd["exception_type"] == "1", "service_id"])
        active -= set(cd.loc[cd["exception_type"] == "2", "service_id"])
    return active


def main():
    day = dt.datetime.fromisoformat(
        json.loads((INTERIM / "departure.json").read_text())["departure"]).date()
    zf = zipfile.ZipFile(GTFS)

    services = active_services(zf, day)
    trips = read(zf, "trips.txt", usecols=["trip_id", "route_id", "service_id"])
    trips = trips[trips["service_id"].isin(services)]
    routes = read(zf, "routes.txt", usecols=["route_id", "route_type"])
    routes["w"] = np.where(pd.to_numeric(routes["route_type"], errors="coerce")
                           .isin(RAIL_TYPES), 2.0, 1.0)
    trips = trips.merge(routes[["route_id", "w"]], on="route_id", how="left")
    print(f"{day}: {len(services)} active services, {len(trips)} trips")

    st = read(zf, "stop_times.txt", usecols=["trip_id", "stop_id"])
    st = st.merge(trips[["trip_id", "route_id", "w"]], on="trip_id", how="inner")

    # per stop: weekday departures + weighted distinct routes
    deps = st.groupby("stop_id").size().rename("deps")
    stop_routes = st.drop_duplicates(["stop_id", "route_id"])

    stops = read(zf, "stops.txt", usecols=["stop_id", "stop_lat", "stop_lon"])
    stops["h3"] = [h3.latlng_to_cell(float(la), float(lo), 8)
                   for la, lo in zip(stops["stop_lat"], stops["stop_lon"])]
    stop_cell = stops.set_index("stop_id")["h3"]

    cell_deps = deps.groupby(stop_cell).sum()
    sr = stop_routes.assign(cell=stop_routes["stop_id"].map(stop_cell))
    cell_route_sets = sr.drop_duplicates(["cell", "route_id"]).groupby("cell")[
        "route_id"].agg(set)

    grid = pd.read_parquet(INTERIM / "grid.parquet")
    deps_n = np.zeros(len(grid))
    routes_n = np.zeros(len(grid))
    rw_lookup = dict(zip(sr.drop_duplicates("route_id")["route_id"],
                         sr.drop_duplicates("route_id")["w"]))
    for i, c in enumerate(grid["h3"]):
        disk = h3.grid_disk(c, 1)
        d = sum(cell_deps.get(x, 0) for x in disk)
        rs = set()
        for x in disk:
            rs |= cell_route_sets.get(x, set())
        deps_n[i] = d
        routes_n[i] = sum(rw_lookup.get(r, 1.0) for r in rs)

    # half-saturation calibrated to observed range: Hbf ~25k deps/263 lines,
    # Kreisstadt ~5-9k/30-60, rural village ~30/4
    freq_score = deps_n / (deps_n + 3000.0)
    div_score = routes_n / (routes_n + 15.0)
    out = pd.DataFrame({
        "h3": grid["h3"],
        "deps_per_day": deps_n,
        "n_lines_w": routes_n,
        "oepnv_score": np.sqrt(freq_score * div_score),
    })
    LAYERS.mkdir(parents=True, exist_ok=True)
    out.to_parquet(LAYERS / "oepnv.parquet", index=False)
    print(out[["deps_per_day", "n_lines_w", "oepnv_score"]]
          .describe().round(2).to_string())


if __name__ == "__main__":
    main()
