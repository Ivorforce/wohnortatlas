"""Derive the Branche (occupational sector) Anbindung targets — route-free, like 04f.

A Branche target ships, per cell + mode, an M/B/O triple that the client turns into a
budget-dependent score (a dedicated field-target path in recomputeAnbindung):
  M = minutes to the BEST job center reachable within B_WINDOW (the app's min budget) — not
      the nearest (latches onto a far one) nor the global best (latches onto a distant city);
      the commute you'd make + the reachEff / over-budget filter (drop a mode where T < M).
  B = that center's opportunity, time-penalized — the FLOOR (your score at T = M).
  O = 1 − Π_centers(1 − O·decay) — a NOISY-OR over everything reachable within HORIZON (2 h),
      the CEILING (your score at T = HORIZON). Saturating + bounded [0,1] + always ≥ B, so a
      dense city reads thick by bike as well as car (mode-robust) and reaching MORE hubs lifts
      O toward 1 with diminishing returns.
Client score for a mode: filter if T < M·pen, else B + (T·…−M)/(HORIZON−M)·(O − B) — it scales
with BOTH the time budget and the reachable opportunity, B ≤ score ≤ O. Modes combine
best-wins. The cell's own field mass enters as a self center at decay 1 ("you're already here").

Sector jobs (GENESIS 52111-07-01-4, per Kreis × WZ-Abschnitt → JOBS_BUCKETS, wohnen/
genesis.py) are placed on each Kreis's population centers by catchment share, then turned
into a per-center weight O = combine_o(m/(m+HALF), lq) with a PER-SECTOR half-saturation so
sectors are comparable — picking "Gastgewerbe" isn't penalised for having fewer total jobs
than "Industrie".

Reads reach_centers.npz (04c) + centers.parquet (04b) + GENESIS (cached). Writes
jobs_kreis.parquet (the Kreis table, change-gated) + reach_branche.npz (per-sector, per-mode
m_/b_/o_ matrices). 22_build_web emits the lazy chunks web/reach/branche-<key>.bin + the
BRANCHES manifest. Tuning HORIZON / JOBS_O_HALF_PCT / JOBS_SPEC_W = rerun this (no routing).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

from wohnen.config import BBOX, INTERIM, JOBS_BUCKETS, LAYERS, RAW
from wohnen.genesis import fetch_jobs_kreis
from wohnen.io import write_parquet_if_changed
from wohnen.mbo import decay as mbo_decay, mbo_triple   # shared M/B/O derivation (also 04f)
from wohnen.reach import MODE_COLS, MODE_DECAY

SENTINEL = 255
# Field-strength O = blend of ABSOLUTE opportunity and SPECIALISATION (see combine_o). The
# client uses it as a LINEAR gradient multiplier (NOT the cityness anbindungGate), so a
# sector pick gives a real hub→rural gradient AND different sectors give DIFFERENT maps
# (a metro tops only the fields it's actually over-weighted in — Berlin IT≫industry). Both
# knobs are route-free (rerun 04h):
JOBS_O_HALF_PCT = 70.0   # absolute-opportunity half-saturation = this percentile of the
                         # positive per-center sector masses (lower → more places count).
JOBS_SPEC_W = 0.5        # specialisation weight in the blend: 0 = pure absolute (every metro
                         # tops every sector → maps look alike), 1 = pure location-quotient
                         # (tiny monoculture towns). 0.5 = balanced; raise to differentiate
                         # harder. (More jobs = more competitors too, so absolute shouldn't
                         # fully win — hence a mix.)
NATIVE_FEATHER_KM = 2.5  # Gaussian feather of the native sector surfaces across Kreis borders:
                         # jobs/lq are Kreis-flat (hard step at every boundary, cf. kreis_pay).
                         # Only the Kreis-flat density + lq are feathered, so the step becomes a
                         # gradient while the sharp within-Kreis texture (cell_catch, applied
                         # after) is preserved. Tighter than pay's 6 km — jobs concentrate more
                         # sharply than wage levels.

O_MODES = ["transit", "bike", "walk", "car"]  # the 4 gate columns shipped per sector


def load_gemeinden() -> gpd.GeoDataFrame:
    """VG250 Gemeinden (land only) with AGS + geometry, clipped to the bbox — the same
    source 06_rent.py uses; AGS[:5] is the Kreis key."""
    import zipfile
    z = RAW / "vg250.zip"
    shp = next(n for n in zipfile.ZipFile(z).namelist() if n.endswith("VG250_GEM.shp"))
    vg = gpd.read_file(f"zip://{z}!{shp}").to_crs(4326)
    vg = vg[vg["GF"] == 4]                      # GF==4 = land (drop water bodies)
    return vg[["AGS", "geometry"]].cx[BBOX[0]:BBOX[2], BBOX[1]:BBOX[3]]


def points_kreis(lon, lat, gem=None) -> pd.Series:
    """5-digit Kreis AGS per point via point-in-Gemeinde join (06_rent.py:221 pattern).
    Used for the centers AND for every cell (the native-value Kreis lookup)."""
    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(np.asarray(lon), np.asarray(lat)), crs=4326)
    j = gpd.sjoin(pts, gem if gem is not None else load_gemeinden(), how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")]   # a point on a border can match two Gemeinden
    return j["AGS"].str[:5].reindex(range(len(pts))).reset_index(drop=True)


def center_mass(centers: pd.DataFrame, kreis: pd.Series, jobs: pd.DataFrame) -> pd.DataFrame:
    """Per-center sector mass = Kreis sector jobs × (center catchment / Σ catchment of that
    Kreis's centers). Distributes each Kreis's jobs onto its population peaks."""
    df = centers[["catchment_pop"]].copy()
    df["kreis"] = kreis.values
    tot = df.groupby("kreis")["catchment_pop"].transform("sum")
    share = (df["catchment_pop"] / tot).fillna(0.0)
    mass = pd.DataFrame(index=centers.index)
    for key in JOBS_BUCKETS:
        kreis_jobs = df["kreis"].map(jobs[key]).fillna(0.0)   # NaN bucket (suppressed) → 0
        mass[key] = (kreis_jobs * share).to_numpy()
    return mass


def cell_native_inputs(cells, gem):
    """Per-cell (Kreis AGS, catchment_pop, xy-km) for the native-value weight: grid coords +
    the smoothed 3-km mass (07a), Kreis via the same point-in-Gemeinde join as the centers.
    Aligned to the reach_centers.npz cell order. xy = equirectangular km (the feather metric)."""
    df = pd.DataFrame({"h3": np.asarray(cells, dtype=str)})
    df = df.merge(pd.read_parquet(INTERIM / "grid.parquet")[["h3", "lat", "lon"]], on="h3", how="left")
    df = df.merge(pd.read_parquet(LAYERS / "population.parquet")[["h3", "catchment_pop"]],
                  on="h3", how="left")
    kreis = points_kreis(df["lon"].to_numpy(), df["lat"].to_numpy(), gem)
    lat, lon = df["lat"].to_numpy(), df["lon"].to_numpy()
    lat0 = np.deg2rad(np.nanmean(lat))
    xy = np.column_stack([np.deg2rad(lon) * np.cos(lat0) * 6371.0, np.deg2rad(lat) * 6371.0])
    return kreis, df["catchment_pop"].fillna(0.0).to_numpy(float), xy


def feather_surfaces(xy, cols, sigma_km, radius_km):
    """Row-normalized Gaussian smooth of every column of `cols` (N, k) over cells within
    radius_km, via ONE shared sparse operator (a cKDTree distance matrix → Gaussian weights).
    Feathers the Kreis-flat native surfaces (job density, lq) across Kreis borders into a
    gradient — like kreis_pay's disk_gaussian_mean but batched over the 9 sectors at once.
    Uniform-neighbourhood interiors are unchanged; only border zones blend."""
    tree = cKDTree(xy)
    D = tree.sparse_distance_matrix(tree, radius_km, output_type="coo_matrix")
    w = np.exp(-(D.data ** 2) / (2.0 * sigma_km ** 2))
    W = csr_matrix((w, (D.row, D.col)), shape=D.shape)
    W.setdiag(1.0)                                   # self (Gaussian at d=0); guards empty rows
    W = W.multiply(1.0 / np.asarray(W.sum(1)).ravel()[:, None]).tocsr()
    return W @ cols


def location_quotient(jobs: pd.DataFrame) -> pd.DataFrame:
    """Per-Kreis sector over-representation: (sector's share of the Kreis's jobs) / (national
    sector share). >1 = the field is concentrated here beyond what the Kreis's size implies.
    The center catchment share cancels (it's identical across sectors at a center), so this is
    a clean Kreis-level quantity."""
    tot = jobs.sum(axis=1)
    natl = jobs.sum(axis=0)
    return jobs.div(tot, axis=0).div(natl / natl.sum(), axis=1)


def sector_half(m: np.ndarray) -> float:
    """Absolute-opportunity half-saturation = JOBS_O_HALF_PCT percentile of the POSITIVE
    per-CENTER masses. Computed once from the centers and reused for the per-cell native
    weights, so center O and native O sit on one scale."""
    pos = m[m > 0]
    return float(np.percentile(pos, JOBS_O_HALF_PCT)) if pos.size else 1.0


def combine_o(m: np.ndarray, lq: np.ndarray, half: float) -> np.ndarray:
    """Field-strength weight ∈ [0,1]: a geometric blend of ABSOLUTE opportunity (more jobs =
    more openings/variety, saturating at `half`) and SPECIALISATION (location quotient).
    Absolute alone makes every metro top every sector (maps look identical); LQ alone surfaces
    tiny monoculture towns (rural Vorpommern reads health-rich only because nothing else is
    there). The product keeps genuine hubs, differentiates the metros (Berlin IT≫industry), and
    the absolute factor kills the LQ artefacts. JOBS_SPEC_W sets the mix."""
    abs_o = m / (m + max(half, 1e-9))
    lq_o = lq / (lq + 1.0)                      # saturating: lq=1 (proportional) → 0.5
    return abs_o ** (1.0 - JOBS_SPEC_W) * lq_o ** JOBS_SPEC_W


def main():
    jobs = fetch_jobs_kreis()
    if jobs.empty:
        print("NOTE: no GENESIS jobs — Branche targets skipped"); return
    write_parquet_if_changed(jobs.reset_index(), LAYERS / "jobs_kreis.parquet",
                             sort_cols=["kreis_ags"])

    d = np.load(LAYERS / "reach_centers.npz", allow_pickle=False)
    cids, cells = d["center_ids"].astype(str), d["cell_ids"]
    N = len(cells)
    U = {c: d[c] for c in MODE_COLS}                          # 5 × (C, N) uint8
    centers = pd.read_parquet(LAYERS / "centers.parquet")
    centers = centers.assign(id=centers["id"].astype(str)).set_index("id").reindex(cids)

    gem = load_gemeinden()
    kreis = points_kreis(centers["lon"].to_numpy(), centers["lat"].to_numpy(), gem)
    kreis.index = centers.index
    mass = center_mass(centers, kreis, jobs)                  # (C, buckets)

    keys = list(JOBS_BUCKETS)
    B, C = len(keys), len(cids)
    arange = np.arange(N)
    lq = location_quotient(jobs)
    half = {key: sector_half(mass[key].to_numpy(np.float32)) for key in keys}
    Ogate = {key: combine_o(mass[key].to_numpy(np.float32),
                            kreis.map(lq[key]).fillna(0.0).to_numpy(np.float32), half[key])
             for key in keys}   # (C,) field-strength (absolute × specialisation) per sector

    # per-cell native sector value (same scale as the per-center mass): the cell's Kreis jobs
    # spread onto it by its OWN catchment share, then the SAME combine_o (shared `half`). It
    # enters M/B/O below as a "self center" at decay 1 (you're already here, no commute). The
    # Kreis-flat pieces — job density (jobs / Σ Kreis-centers' catchment) and lq — are FEATHERED
    # across Kreis borders (else a kreisfreie Stadt vs its Landkreis shows an administrative
    # step); the sharp within-Kreis texture is cell_catch, multiplied in after the feather.
    cell_kreis, cell_catch, cell_xy = cell_native_inputs(cells, gem)
    kreis_catch_sum = pd.Series(centers["catchment_pop"].to_numpy(),
                                index=kreis.values).groupby(level=0).sum()
    denom = cell_kreis.map(kreis_catch_sum).to_numpy(float)
    inv_denom = np.where(denom > 0, 1.0 / np.where(denom > 0, denom, 1.0), 0.0)
    dens = np.column_stack([cell_kreis.map(jobs[k]).fillna(0.0).to_numpy(float) * inv_denom
                            for k in keys])                                  # (N, B) Kreis-flat
    lqc = np.column_stack([cell_kreis.map(lq[k]).fillna(0.0).to_numpy(float) for k in keys])
    sm = feather_surfaces(cell_xy, np.hstack([dens, lqc]), NATIVE_FEATHER_KM, 2.0 * NATIVE_FEATHER_KM)
    dens_f, lq_f = sm[:, :len(keys)], sm[:, len(keys):]
    native_O = {key: combine_o((dens_f[:, i] * cell_catch).astype(np.float32),
                               lq_f[:, i].astype(np.float32), half[key])
                for i, key in enumerate(keys)}

    # Per (mode, sector): the M/B/O triple (wohnen/mbo.mbo_triple). decay depends only on the
    # mode → compute once; reuse the (C, N) score buffer (it holds term = O·decay).
    score = np.empty((C, N), np.float32)
    Mm = {m: np.full((B, N), 255, np.uint8) for m in O_MODES}
    Bm = {m: np.full((B, N), 255, np.uint8) for m in O_MODES}
    Om = {m: np.full((B, N), 255, np.uint8) for m in O_MODES}
    for mode, decay_cols in MODE_DECAY.items():
        t = np.minimum.reduce([U[c] for c in decay_cols])               # (C, N) uint8 door-to-door
        dk = mbo_decay(t)
        for bi, key in enumerate(keys):
            np.multiply(dk, Ogate[key][:, None], out=score)             # term = O·decay, reused
            Mm[mode][bi], Bm[mode][bi], Om[mode][bi] = mbo_triple(score, t, native_O[key], arange)
        del dk, t

    LAYERS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        LAYERS / "reach_branche.npz",
        bucket_ids=np.asarray(keys, dtype=str), cell_ids=np.asarray(cells, dtype=str),
        **{f"m_{m}": Mm[m] for m in O_MODES}, **{f"b_{m}": Bm[m] for m in O_MODES},
        **{f"o_{m}": Om[m] for m in O_MODES})
    cov = {k: f"{(Mm['car'][i] < 255).mean():.0%}" for i, k in enumerate(keys)}
    print(f"wrote reach_branche.npz ({B} sectors × {N} cells, M/B/O × {len(O_MODES)} modes); "
          "car-reachable: " + ", ".join(f"{k} {v}" for k, v in cov.items()))


if __name__ == "__main__":
    main()
