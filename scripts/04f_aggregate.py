"""Derive ALL route-free commute target maps from the persisted per-center reach
(reach_centers.npz, written by 04c) — NO routing, NO r5py (route-free, like 04e).
Two jobs, both keyed off the SAME per-center×cell minutes, so they re-derive in
seconds when a knob changes (the routing in 04c is the only expensive step):

1. Per-CITY / per-PART reach (reach_cities.npz + cities / reach_parts.npz + parts.parquet) —
   the Lebensmittelpunkt picker's named targets, shipped as uint8 (targets × cells) matrices
   (255 = unreachable), cell_ids stored once — a long parquet was ~1.16e9 rows nationally.
   A whole CITY is a catchment-weighted
   PERCENTILE (PCT_CITY) over its centers: "reach MOST of the city's people decently",
   the right reading of "I have ties all across it". The old nanmin shipped the NEAREST
   corner, so a home hugging one edge looked great while the far side was an hour away —
   and that was ~redundant with "Irgendeine Großstadt" anyway. A PART is a single center
   (a job in one specific corner scores honestly, no min). 22_build_web ships these as
   lazy per-target chunks; the manifests are cities.parquet / parts.parquet.

2. The "Alle Branchen (Stadt/Großstadt)" cityness field targets (reach_cityness.npz). Per
   tier (o_any / o_gross) × mode, the M/B/O triple (wohnen/mbo) over the cityness opportunity
   + the cell's own native cityness — the SAME shape as a job sector (04h), so the web scores
   them with one path (recomputeAnbindungMBO's budget interp). The tier is just the O: o_any
   includes towns (c/(c+C_HALF)); o_gross (a catchment smoothstep, wohnen/cityness.py) gates to
   metropolises. 22_build_web emits these as the branche-{any,gross} lazy chunks.

Tuning PCT_CITY / the cityness smoothstep = re-run this (seconds), no re-routing.
Reads reach_centers.npz (04c) + centers.parquet + population.parquet. Writes reach_cityness.npz
(→ 22's branche chunks) + reach_cities.npz/cities.parquet + reach_parts.npz/parts.parquet
(22_build_web's picker).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from wohnen.cityness import cityness_o
from wohnen.config import LAYERS, TRAVEL_SENTINEL_MIN
from wohnen.mbo import mbo_surfaces
from wohnen.reach import MODE_COLS, MODE_DECAY

SENTINEL = 255        # reach_centers.npz uint8 no-reach marker
PCT_CITY = 0.50       # catchment-weighted percentile for whole-city reach: the shipped
                      # minutes are the time to the MEDIAN (typical) part of the city's
                      # people. p80 was too harsh + dominated by the slow far-edge tail
                      # (the Hbf itself read ~40 min ÖPNV); the median keeps the
                      # center-vs-suburb gradient but reads as "the typical Münchner is
                      # T min away". Lower (p30) drifts back toward rewarding edge-hugging.

TIERS = ("any", "gross")


def slugify(name):
    return "".join(c.lower() if c.isalnum() else "-" for c in str(name)).strip("-") or "ort"


def weighted_percentile(vals, weights, q):
    """Weighted q-quantile of `vals` over axis 0 (the city's centers), 'lower' method
    (no interpolation — minutes are coarse enough). vals (C, N) finite, weights (C,).
    A single center → that center's row. Unreachable centers must already be filled
    with a large value (so they push the high percentiles up, not get skipped)."""
    C = vals.shape[0]
    if C == 1:
        return vals[0].copy()
    w = np.asarray(weights, dtype=float)
    if not np.isfinite(w).all() or w.sum() <= 0:
        w = np.ones(C)
    order = np.argsort(vals, axis=0, kind="stable")          # (C, N) per-cell order
    sv = np.take_along_axis(vals, order, axis=0)
    cw = np.cumsum(w[order], axis=0)                         # weights aligned to sorted vals
    target = q * cw[-1]                                       # (N,) — cw[-1] = Σw, constant
    idx = np.clip((cw < target[None, :]).sum(axis=0), 0, C - 1)
    return np.take_along_axis(sv, idx[None, :], axis=0)[0]


LINK_KM = 50.0   # within-name spatial-cluster link distance. Centers of one real city sit a
                 # few km apart (Fürth's 3 ~5 km, Erfurt's 3 ~7 km); distinct municipalities
                 # that merely share a Gemeinde name sit >150 km apart (Senden/Bayern vs
                 # Senden/NRW = 433 km). 50 km cleanly separates the two cases.


def _clusters(lat, lon, link_km):
    """Single-linkage spatial clusters of points within link_km (equirectangular km
    approx). Returns a list of index-arrays into the input. Groups are tiny (≤ a
    handful of centers) so the O(k²) union-find is trivial."""
    k = len(lat)
    parent = list(range(k))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    lim = link_km * link_km
    for i in range(k):
        for j in range(i + 1, k):
            dy = (lat[i] - lat[j]) * 111.0
            dx = (lon[i] - lon[j]) * 111.0 * np.cos(np.radians((lat[i] + lat[j]) / 2))
            if dx * dx + dy * dy <= lim:
                parent[find(i)] = find(j)
    groups = {}
    for i in range(k):
        groups.setdefault(find(i), []).append(i)
    return [np.asarray(g) for g in groups.values()]


def city_groups(name, district, lat, lon, catch):
    """Group centers into named cities, SPLITTING same-name centers that are not
    spatially connected (single-linkage, LINK_KM) into separate targets — grouping by
    name string alone merged distinct municipalities (corrupting the whole-city
    percentile and drawing a hull across Germany). Within a name the largest-catchment
    cluster keeps the bare slug; the rest get a -2/-3/… suffix and, where the
    representative neighbourhood label differs, a disambiguated display name. Yields
    (slug, display_name, pos-array into the center rows), stable order."""
    for nm in pd.unique(name):
        allpos = np.flatnonzero(name == nm)
        base = slugify(nm)
        clusters = _clusters(lat[allpos], lon[allpos], LINK_KM)
        clusters.sort(key=lambda c: (-catch[allpos[c]].sum(), int(allpos[c].min())))
        for n, c in enumerate(clusters):
            pos = allpos[c]
            slug = base if n == 0 else f"{base}-{n + 1}"
            disp = nm
            if n > 0:                                  # disambiguate the namesake in the picker
                drep = district[pos[int(np.argmax(catch[pos]))]]
                disp = f"{nm} ({drep})" if drep and drep != nm else f"{nm} ({n + 1})"
            yield slug, disp, pos


def main():
    d = np.load(LAYERS / "reach_centers.npz", allow_pickle=False)
    cids, cells = d["center_ids"].astype(str), list(d["cell_ids"])
    N = len(cells)

    # Reach stays uint8 (255 = unreachable). Converting all five modes to float64 at once was
    # ~57 GB and swap-thrashed at national scale; minimum.reduce works directly on uint8 (255
    # is the max, so the min picks the reachable value), and float is taken only transiently
    # per mode-group below.
    U = {c: d[c] for c in MODE_COLS}                 # 5 × (C, N) uint8 ≈ 7 GB
    centers = pd.read_parquet(LAYERS / "centers.parquet")
    centers = centers.assign(id=centers["id"].astype(str)).set_index("id")

    # per-cell "native value": a cell's OWN cityness — the same O a center sitting on it
    # would carry — folded into the FOOT mode at zero distance, so cells BETWEEN a city's
    # routed centers aren't punished for not coinciding with one. catchment_pop is the
    # smoothed 3-km mass surface (07a), so cityness_o is just the per-center weight evaluated
    # everywhere (wohnen/cityness.py — same definition 04b samples at the peaks).
    cat = (pd.DataFrame({"h3": cells})
           .merge(pd.read_parquet(LAYERS / "population.parquet")[["h3", "catchment_pop"]],
                  on="h3", how="left")["catchment_pop"].fillna(0.0).to_numpy(float))

    arange = np.arange(N)

    # --- M/B/O cityness for the web "Alle Branchen" field targets (same shape as the job
    # sectors, 04h, via wohnen/mbo): per tier (any/gross) × mode, the M/B/O triple over the
    # cityness opportunity (o_any/o_gross) + the cell's own native cityness (cityness_o). ------
    # per-centre cityness O from catchment via cityness_o (the SAME definition as the native +
    # 22's outline) — NOT centers["o_*"] (04b's, which uses whatever GROSS_LO/HI shipped then),
    # so retuning the smoothstep needs only a 04f rerun, no 04b/re-route. native = the cell's own
    # cityness (a self centre at decay 1). Both depend only on the tier, not the mode.
    center_cat = centers["catchment_pop"].reindex(cids).fillna(0.0).to_numpy(float)
    Otier = {tier: cityness_o(center_cat, tier).astype(np.float32) for tier in TIERS}
    natier = {tier: cityness_o(cat, tier).astype(np.float32) for tier in TIERS}
    cy = {}                                                  # f"{tier}_{m|b|o}_{mode}" -> (N,) u8
    for mode, per_tier in mbo_surfaces(U, MODE_DECAY, Otier, natier, arange).items():
        for tier, (M_, B_, O_) in per_tier.items():
            cy[f"{tier}_m_{mode}"], cy[f"{tier}_b_{mode}"], cy[f"{tier}_o_{mode}"] = M_, B_, O_
    np.savez_compressed(LAYERS / "reach_cityness.npz",
                        tiers=np.asarray(list(TIERS), dtype=str),
                        cell_ids=np.asarray(cells, dtype=str), **cy)

    LAYERS.mkdir(parents=True, exist_ok=True)
    cov = {t: f"{(cy[f'{t}_m_car'] < 255).mean():.0%}" for t in TIERS}

    # --- per-CITY / per-PART reach as uint8 MATRICES (the picker's named targets) --------
    # Ship reach_cities/reach_parts as npz matrices (targets × cells, uint8 minutes, 255 =
    # unreachable) with cell_ids stored ONCE — the reach_centers.npz model. The old long
    # parquet repeated the h3 string + float64 minutes per (target, cell) row: 2160 cities ×
    # 536 553 cells ≈ 1.16e9 rows (~90 GB) → OOM. uint8 matrix is ~6 GB in RAM, <1 GB on
    # disk; full-minute resolution is lossless (routing caps at max_time, 255 = beyond reach).
    cm = centers.reindex(cids)                       # centers aligned to the npz row order
    name = cm["name"].to_numpy()
    catch = cm["catchment_pop"].to_numpy()
    district = cm["district"].astype(str).to_numpy()
    lat, lon = cm["lat"].to_numpy(), cm["lon"].to_numpy()
    cell_arr = np.asarray(cells, dtype=str)

    # whole city = catchment-weighted PCT_CITY percentile over its centers (reach most of
    # its people), NOT the nearest corner. Unreachable centers count as the sentinel so a
    # far-side gap pushes the percentile up instead of being silently dropped.
    groups = list(city_groups(name, district, lat, lon, catch))
    city_slugs = [g[0] for g in groups]
    city_mat = {m: np.full((len(groups), N), 255, np.uint8) for m in MODE_COLS}
    city_meta = []
    for i, (slug, disp, pos) in enumerate(groups):
        for m in MODE_COLS:
            v = U[m][pos].astype(np.float32)                             # (k, N)
            allun = (v >= SENTINEL).all(axis=0)                          # unreachable from every center
            vf = np.where(v >= SENTINEL, float(TRAVEL_SENTINEL_MIN), v)
            pct = weighted_percentile(vf, catch[pos], PCT_CITY)
            city_mat[m][i] = np.where(allun, 255, np.clip(np.round(pct), 0, 254)).astype(np.uint8)
        b = pos[int(np.argmax(catch[pos]))]
        city_meta.append((slug, disp, "city", float(lat[b]), float(lon[b]), float(catch[b])))
    cities = pd.DataFrame(city_meta, columns=["id", "name", "kind", "lat", "lon", "catchment_pop"])
    np.savez_compressed(LAYERS / "reach_cities.npz",
                        target_ids=np.asarray(city_slugs, dtype=str), cell_ids=cell_arr, **city_mat)
    del city_mat

    # per-PART reach: in a MULTI-center city each center is individually selectable (a
    # specific corner where e.g. a job sits) — vs the whole-city percentile above. A single
    # center carries no percentile, so it scores a far-side home honestly. Single-center towns
    # have no parts (the part would equal the city). Search surfaces these; chips don't.
    part_ids, part_meta, seen = [], [], set()
    part_cols = {m: [] for m in MODE_COLS}
    for cityslug, disp, pos in groups:
        if len(pos) <= 1:                   # single-center cluster: the part would equal the city
            continue
        for p in pos:
            base = f"{cityslug}-{slugify(district[p])}"
            pslug, k = base, 2
            while pslug in seen:            # rare: two centers share a district slug
                pslug, k = f"{base}-{k}", k + 1
            seen.add(pslug)
            label = f"{disp} (Zentrum)" if district[p] == disp else f"{disp} – {district[p]}"
            part_ids.append(pslug)
            for m in MODE_COLS:
                part_cols[m].append(U[m][p])                             # the center's own row (uint8)
            part_meta.append((pslug, label, cityslug, float(lat[p]), float(lon[p]), cids[p]))

    parts = pd.DataFrame(part_meta, columns=["id", "name", "city", "lat", "lon", "center_h3"])
    part_mat = {m: (np.stack(part_cols[m]) if part_ids else np.empty((0, N), np.uint8))
                for m in MODE_COLS}
    np.savez_compressed(LAYERS / "reach_parts.npz",
                        target_ids=np.asarray(part_ids, dtype=str), cell_ids=cell_arr, **part_mat)
    cities.to_parquet(LAYERS / "cities.parquet", index=False)
    parts.to_parquet(LAYERS / "parts.parquet", index=False)

    print(f"wrote reach_cityness.npz ({N} cells, M/B/O × {len(MODE_DECAY)} modes); "
          f"car-present any/gross = {cov['any']}/{cov['gross']}")
    print(f"wrote reach_cities.npz ({cities.shape[0]} cities, p{int(PCT_CITY * 100)} "
          f"catchment-weighted, uint8 matrix), cities.parquet, reach_parts.npz/parts.parquet "
          f"({parts.shape[0]} parts in {parts['city'].nunique() if len(parts) else 0} cities)")


if __name__ == "__main__":
    main()
