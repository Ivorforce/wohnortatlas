"""Derive ALL s_freizeit reachability surfaces from the persisted routing caches —
NO routing, NO r5py (the JDK-free tuning step, and the single home of the Freizeit
surface math). Re-run after changing any knob below; it takes seconds.

Reads:
  reach_centers.npz (04c) : per-center×cell MINUTES → going-out GRAVITY surfaces
    reach_activity_* (broad mass, local τ) + reach_kultur_* (urban mass, gentle τ),
    each P99.9-anchored. mass = each center's going-out density (entertainment.parquet),
    so τ/mass/norm are all tunable HERE.
  reach_spots.npz   (04d) : per-source gravity SUM + per-cell NEAREST-spot time, per mode
    → reach_{swim,kino,klettern,golf}_* via the "need just one" model (wohnen.freizeit:
    nearest sets a distance ceiling, variety fills it). POINT_B/POINT_K re-tune here;
    the decay SHAPE (POINT_T0/SIG/P) is baked into the sum at route time → re-route 04d.

Writes layers/freizeit.parquet (h3 + all 21 surfaces); 20_assemble folds it into
scores. `--sweep` prints a reference-town τ table for the going-out surfaces.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from wohnen.config import LAYERS
from wohnen.freizeit import (
    POINT_B, POINT_K, POINT_P, POINT_SIG, POINT_T0, SOURCES, point_reach)
from wohnen.reach import MODE_COLS, gravity_sum

# --- going-out surfaces (from center MINUTES) -------------------------------
# reach_activity (baseline): BROAD mass (culture+nightlife+gastro), LOCAL τ (15 min —
#   a 15-min trip counts ~37 %, a 40-min restaurant drive doesn't), so a cell reads by
#   its OWN going-out density. (client applies a GENTLE γ=0.6.)
# reach_kultur ("Ausgehen"): URBAN mass (culture+nightlife, gastro DROPPED), GENTLE τ
#   (20 min) so reaching München's scene is a penalized bonus — the city selector.
#   (client applies a MILD γ=1.3.)
# Each cell's OWN going-out density floors the gravity (np.maximum, see native_for): centers
#   are sparse pop-peaks, so a city cell between them otherwise loses the few-minute trip to
#   the nearest scene though its own is right here.
# P99.9 anchor (not P98): at small τ the surface is mostly empty, so a P98 anchor
#   saturates every town to 1.0 and kills the local gradient.
ACTIVITY_TAU = 15.0
KULTUR_TAU = 20.0
GOINGOUT_HORIZON = 60.0
FREIZEIT_PCT = 99.9
GOINGOUT = [  # (output prefix, mass columns, τ)
    ("reach_activity", ["ent_culture", "ent_nightlife", "ent_gastro"], ACTIVITY_TAU),
    ("reach_kultur", ["ent_culture", "ent_nightlife"], KULTUR_TAU),
]


def _pctnorm(v):
    p = v[v > 0]
    return np.clip(v / max(np.percentile(p, FREIZEIT_PCT) if p.size else 1.0, 1e-9), 0.0, 1.0)


def _load_centers():
    """reach_centers.npz → (modes_t dict {transit,bike,car}: (centers,N) minutes,
    mass_for(cols) → per-center pull, native_for(cols) → per-CELL own density,
    is_center mask, cell_ids)."""
    d = np.load(LAYERS / "reach_centers.npz", allow_pickle=False)
    cids, cells = d["center_ids"], list(d["cell_ids"])
    # keep the raw uint8 minutes (≥255 sentinel); gravity_sum masks ≥horizon per chunk,
    # so we never materialize 5 float64 (centers × cells) copies (~11 GB each → OOM as the
    # center count grows). transit = min over its two access legs, in uint8.
    reach = {m: d[m] for m in MODE_COLS}
    modes_t = {"transit": np.minimum(reach["transit_hbf_min"], reach["transit_bike_min"]),
               "bike": reach["bike_hbf_min"], "foot": reach["walk_min"], "car": reach["car_hbf_min"]}
    centers = pd.read_parquet(LAYERS / "centers.parquet")
    h3_of = centers.assign(id=centers["id"].astype(str)).set_index("id")["h3"]
    ent = pd.read_parquet(LAYERS / "entertainment.parquet").set_index("h3")
    # ent_* are pre-blurred densities (nonzero almost everywhere), so a cell's OWN going-out
    # density is a faithful "you're already here" anchor — see native_for below.
    is_center = pd.Series(cells).isin(set(h3_of.to_numpy())).to_numpy()

    def _cell_mass(mass_cols):
        cols = [c for c in mass_cols if c in ent.columns]
        return ent[cols].sum(axis=1) if cols else pd.Series(0.0, index=ent.index)

    def mass_for(mass_cols):
        return h3_of.reindex(cids).map(_cell_mass(mass_cols)).fillna(0.0).to_numpy()

    def native_for(mass_cols):
        # each cell's own going-out density as a t=0 self-source (decay 1) — the analogue of
        # the jobs native: between sparse centers a city cell is otherwise penalized for the
        # few minutes to the nearest one, though its OWN scene is right here. Combined by MAX
        # (a floor, "need just one"): a cell scores at least as if it were a lone center with
        # its own density, never less. Centers already carry this via the routing diagonal, so
        # mask them out (max is idempotent for them, but keep the intent explicit).
        nat = _cell_mass(mass_cols).reindex(cells).fillna(0.0).to_numpy()
        return np.where(is_center, 0.0, nat)

    return modes_t, mass_for, native_for, cells


def going_out_surfaces() -> pd.DataFrame:
    modes_t, mass_for, native_for, cells = _load_centers()
    out = pd.DataFrame({"h3": cells})
    for prefix, mass_cols, tau in GOINGOUT:
        mass = mass_for(mass_cols)
        native = native_for(mass_cols)  # own density at t=0 (mode-independent floor)
        for k, t in modes_t.items():
            raw = np.maximum(gravity_sum(t, mass, tau, GOINGOUT_HORIZON), native)
            out[f"{prefix}_{k}"] = _pctnorm(raw)
    return out


def point_surfaces() -> pd.DataFrame:
    """04d's per-source (gravity sum + nearest-spot time) → [0,1] via the "need just one"
    model (wohnen.freizeit.point_reach: nearest sets a distance ceiling, variety fills it).
    Only sources still in SOURCES are emitted — a dropped source's stale npz keys are
    ignored, so no re-route is needed."""
    d = np.load(LAYERS / "reach_spots.npz", allow_pickle=False)
    out = pd.DataFrame({"h3": list(d["cell_ids"])})
    for key in d.files:
        if key == "cell_ids" or key.endswith("_tmin") or key.split("_")[0] not in SOURCES:
            continue
        if f"{key}_tmin" not in d.files:
            raise KeyError(f"reach_spots.npz lacks '{key}_tmin' — re-run 04d_swim "
                           "(the nearest-spot times the point model needs are missing).")
        out[f"reach_{key}"] = point_reach(d[f"{key}_tmin"].astype(float), d[key].astype(float))
    return out


def sweep():
    """--sweep: going-out reach (max over modes, P99.9) at candidate τ for reference
    towns — pick ACTIVITY_TAU / KULTUR_TAU from data, route-free."""
    import h3
    modes_t, mass_for, native_for, cells = _load_centers()
    idx = {c: i for i, c in enumerate(cells)}
    refs = {"München": (48.137, 11.575), "Freising": (48.402, 11.741),
            "Rosenheim": (47.856, 12.123), "Dorfen": (48.265, 12.158),
            "Garching(S)": (48.249, 11.651), "Starnberg": (47.997, 11.343),
            "Land(BWald)": (48.75, 12.4)}
    refs = {n: idx[h3.latlng_to_cell(la, lo, 8)] for n, (la, lo) in refs.items()
            if h3.latlng_to_cell(la, lo, 8) in idx}
    for label, mass_cols, _ in GOINGOUT:
        mass = mass_for(mass_cols)
        native = native_for(mass_cols)
        print(f"\n  {label} — max-over-modes reach by τ (norm P{FREIZEIT_PCT}):")
        print("    τ   " + "".join(f"{n:>12}" for n in refs))
        for tau in (6, 8, 10, 12, 15, 20, 25):
            s = np.maximum.reduce([_pctnorm(np.maximum(gravity_sum(t, mass, tau, GOINGOUT_HORIZON), native))
                                   for t in modes_t.values()])
            print(f"   {tau:>3}   " + "".join(f"{s[i]:>12.2f}" for i in refs.values()))


def main():
    if "--sweep" in sys.argv:
        sweep()
        return
    out = going_out_surfaces().merge(point_surfaces(), on="h3", how="outer")
    LAYERS.mkdir(parents=True, exist_ok=True)
    out.to_parquet(LAYERS / "freizeit.parquet", index=False)
    print(f"wrote freizeit.parquet ({len(out)} cells, {len(out.columns) - 1} surfaces; "
          f"ACTIVITY_TAU={ACTIVITY_TAU} KULTUR_TAU={KULTUR_TAU} | point model "
          f"T0={POINT_T0} SIG={POINT_SIG} P={POINT_P} B={POINT_B} K={POINT_K})")


if __name__ == "__main__":
    main()
