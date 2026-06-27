"""Shared r5py routing scaffold: the JDK-21 + max-memory setup, the parallel
TravelTimeMatrix monkeypatch, and the `matrix()` helper. Imported lazily (via
wohnen.reach.load_r5) by the reverse-routing scripts — 04c (centers), 04d (POI
spots), 04g (egress) — so the r5py / JDK-21 dependency is pulled in exactly once
and only when something actually routes.

Importing this module starts the JVM-bound r5py with a 22 GB heap and asserts
JAVA_HOME points at JDK 21, so keep it out of the pure-derivation steps (04e/04f).
"""

import datetime as dt
import os
import sys

# r5py reads --max-memory from sys.argv at import; set it before importing r5py.
# This replaces the caller's argv, so the routing scripts parse their own args
# before calling load_r5 (they do — arg parsing happens up front in each main()).
sys.argv = [sys.argv[0], "--max-memory", "22G"]

assert "21" in os.popen(f"{os.environ.get('JAVA_HOME','')}/bin/java -version 2>&1").read(), \
    "JAVA_HOME must point to JDK 21 (run via make, or export JAVA_HOME)"

import concurrent.futures

import pandas as pd

import r5py


# r5py 1.1.6 computes origins sequentially (plain list comprehension in
# TravelTimeMatrix._compute). Per-origin work is independent and jpype releases
# the GIL during the Java call, so a thread pool over origins parallelizes it
# the same way Conveyal's own analysis backend does (shared read-only network,
# request copied per origin).
def _compute_parallel(self):
    self._prepare_origins_destinations()
    self.request.destinations = self.destinations
    ids = list(self.origins.id)
    parts = []
    workers = max(4, (os.cpu_count() or 8) - 2)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        for i, part in enumerate(ex.map(self._travel_times_per_origin, ids), 1):
            parts.append(part)
            if i % 1000 == 0:
                print(f"  {i}/{len(ids)} origins", flush=True)
    od_matrix = pd.concat(parts, ignore_index=True)
    try:
        od_matrix = od_matrix.to_crs(self._origins_crs)
    except AttributeError:
        pass
    return od_matrix


r5py.TravelTimeMatrix._compute = _compute_parallel


def matrix(network, origins, dests, departure, modes, **kw) -> pd.DataFrame:
    ttm = r5py.TravelTimeMatrix(
        network,
        origins=origins,
        destinations=dests,
        departure=departure,
        departure_time_window=dt.timedelta(hours=1),
        percentiles=[50],
        max_time=dt.timedelta(minutes=120),
        transport_modes=modes,
        **kw,
    )
    return ttm.pivot(index="from_id", columns="to_id", values="travel_time")
