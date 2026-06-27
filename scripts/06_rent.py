"""Rent layer: Zensus 2022 (>65 m² class) base, INKAR asking-rent calibration.

rent_2022: stock cold rent €/m² per hex (Zensus 100m grid, flats >65 m²)
rent_cal:  calibrated to asking level via INKAR (Kreis, open data dl-de/by-2-0,
           lags ~1.5 y; BBSR Gruppe 58 Wiedervermietungsmieten)
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import geopandas as gpd
import numpy as np
import pandas as pd

from wohnen.config import (BBOX, INKAR_API, INKAR_RENT_GRUPPE,
                           INTERIM, LAYERS, RAW)
from wohnen.dl import cached_post_json
from wohnen.h3util import RES8_STEP_KM, disk_gaussian_mean
from wohnen.zensus import add_h3, read_grid_csv

UPLIFT_FALLBACK = 1.32  # stock 2022 -> asking 2026, used when INKAR unavailable
FACTOR_RANGE = (1.0, 2.2)

KERNEL_SIGMA_KM = 1.5   # spatial bandwidth of the rent surface
PRIOR_LAMBDA = 300.0    # prior strength: "pseudo-residents" at the Gemeinde mean

# INKAR pay is Kreis-resolution -> a hard step at every Kreis border. Feather
# it with a wide Gaussian so the city->suburb falloff reads as a gradient
# (Kreis means already trend outward from the centres) while interiors stay
# flat. Override via PAY_SIGMA_KM env var; 0 disables blending.
PAY_SIGMA_KM = 6.0


def kernel_shrink(values: pd.Series, pop: pd.Series, prior: pd.Series,
                  k_max: int = 4) -> pd.Series:
    """Population-weighted Gaussian-kernel average, shrunk to a local prior.

    result = (Σ popᵢ·K(dᵢ)·vᵢ + λ·prior) / (Σ popᵢ·K(dᵢ) + λ)

    One closed form replaces median/min-support/widening/dominance rules:
    thin data falls smoothly to the prior, a cheap hamlet contributes
    proportionally instead of pinning a median, uninhabited cells inherit.
    """
    import h3 as h3lib
    data = {c: (v, pop.get(c, 0.0)) for c, v in values.items()
            if pd.notna(v) and pop.get(c, 0.0) > 0}
    kern = [float(np.exp(-(r * RES8_STEP_KM) ** 2 / (2 * KERNEL_SIGMA_KM ** 2)))
            for r in range(k_max + 1)]
    result = {}
    for cell in values.index:
        num = den = 0.0
        for ring in range(k_max + 1):
            k = kern[ring]
            for nb in (h3lib.grid_ring(cell, ring) if ring else [cell]):
                d = data.get(nb)
                if d:
                    w = d[1] * k
                    num += w * d[0]
                    den += w
        p = prior.get(cell)
        if pd.isna(p):
            p = float(np.nanmedian(prior.values))
        result[cell] = (num + PRIOR_LAMBDA * p) / (den + PRIOR_LAMBDA)
    return pd.Series(result).reindex(values.index)


def zensus_base() -> pd.DataFrame:
    big = read_grid_csv(
        RAW / "zensus_rent_by_size.zip",
        "Zensus2022_Gebalter_Wohngr_Miete_100m-Gitterzellen.csv",
        "durchschnMieteQM",
        usecols=["WOHNUNGSGROESSE"],
    )
    big = big[big["WOHNUNGSGROESSE"] == "gross_ueber_65qm"]
    big = add_h3(big)
    base = big.groupby("h3")["value"].mean().rename("rent_2022")

    overall = read_grid_csv(
        RAW / "zensus_rent.zip",
        "Zensus2022_Durchschn_Nettokaltmiete_1km-Gitter.csv",
        "durchschnMieteQM",
    )
    overall = add_h3(overall)
    fill = overall.groupby("h3")["value"].mean()
    ratio = float(base.mean() / fill.mean())
    print(f"zensus: {len(base)} hexes (>65qm), {len(fill)} hexes (1km overall), "
          f"size ratio {ratio:.3f}")
    return base, fill * ratio


def load_gemeinden() -> gpd.GeoDataFrame:
    import zipfile
    zf = zipfile.ZipFile(RAW / "vg250.zip")
    shp = next(n for n in zf.namelist() if n.endswith("VG250_GEM.shp"))
    vg = gpd.read_file(f"zip://{RAW / 'vg250.zip'}!{shp}")
    vg = vg.to_crs(4326)
    return vg[["AGS", "GEN", "geometry"]].cx[BBOX[0]:BBOX[2], BBOX[1]:BBOX[3]]


def _inkar_ca_bundle() -> str:
    """certifi + GoDaddy G2 intermediate: inkar.de omits it from its chain."""
    import certifi
    from wohnen.config import CACHE
    from wohnen.dl import cached_download
    inter = cached_download("https://certs.godaddy.com/repository/gdig2.crt.pem",
                            CACHE / "inkar" / "gdig2.crt.pem")
    bundle = CACHE / "inkar" / "ca_bundle.pem"
    if not bundle.exists():
        bundle.write_bytes(Path(certifi.where()).read_bytes()
                           + b"\n" + inter.read_bytes())
    return str(bundle)


def _inkar_json(path: str, body: dict, cache_key: str):
    """INKAR API call; responses are double-encoded (JSON string of JSON)."""
    data = cached_post_json(f"{INKAR_API}/{path}", body, cache_key=cache_key,
                            rate_bucket="inkar", rate_limit_s=1.0,
                            verify=_inkar_ca_bundle())
    return json.loads(data) if isinstance(data, str) else data


def inkar_factor(vg: gpd.GeoDataFrame, hex_gem: pd.DataFrame,
                 rent: pd.Series, pop: pd.Series) -> pd.Series:
    """Per-Gemeinde asking/stock factor from BBSR INKAR Angebotsmieten.

    Open data (dl-de/by-2-0) — re-hostable, unlike the scraped sources.
    Kreis resolution only (one factor per 5-digit AGS prefix) and rounded to
    whole €/m²; both coarser than Homeday, but the error is below the ±15 %
    agreement bar of the scraped sources. Latest year discovered via the
    Wizard endpoint (currently lags ~1.5 years behind asking level).
    """
    avail = _inkar_json(
        "Wizard/GetM%C3%B6glich",
        {"IndicatorCollection": [{"Gruppe": INKAR_RENT_GRUPPE}],
         "TimeCollection": "", "SpaceCollection": [{"level": "KRE"}]},
        cache_key=f"avail_g{INKAR_RENT_GRUPPE}_kre")
    times = avail["Möglich"]
    latest = max(times, key=lambda t: t["Zeit"])
    rows = _inkar_json(
        "Table/GetDataTable",
        {"IndicatorCollection": [{"Gruppe": INKAR_RENT_GRUPPE}],
         "TimeCollection": [{"group": latest["Gruppe"],
                             "indicator": latest["IndID"],
                             "level": "KRE", "time": latest["ZeitID"]}],
         "SpaceCollection": [{"level": "KRE"}], "pageorder": "1"},
        cache_key=f"rent_g{INKAR_RENT_GRUPPE}_kre_{latest['ZeitID']}")["Daten"]
    asking = pd.Series({r["Schlüssel"]: float(r["Wert"]) for r in rows
                        if r.get("Wert") is not None})
    print(f"inkar: {len(asking)} Kreise, year {latest['Zeit']}, "
          f"asking {asking.min():.0f}..{asking.max():.0f} €/m²")

    # population-weighted Kreis stock rent: INKAR asking rents are listing-
    # weighted (≈ where people live), so a plain hex mean over a whole Kreis
    # would overweight cheap rural cells and inflate the factor
    df = hex_gem.join(rent, on="h3")
    df["pop"] = df["h3"].map(pop).fillna(0)
    df = df.dropna(subset=["ags", "rent_2022"])
    df = df[df["pop"] > 0]
    df["kreis"] = df["ags"].str[:5]
    stock = (df.groupby("kreis")
             .apply(lambda g: np.average(g["rent_2022"], weights=g["pop"]),
                    include_groups=False))
    f_kreis = (asking / stock).dropna().clip(*FACTOR_RANGE)
    print(f"factor: {len(f_kreis)} Kreise in bbox, "
          f"median {f_kreis.median():.2f}, "
          f"range {f_kreis.min():.2f}..{f_kreis.max():.2f}")

    all_ags = pd.Index(vg["AGS"].unique())
    f = pd.Series(all_ags.str[:5], index=all_ags).map(f_kreis)
    print(f"factor coverage: {f.notna().mean():.0%} of Gemeinden")
    return f


def inkar_kreis_pay() -> pd.Series:
    """Median gross pay (€/month, SV-Vollzeit) per Kreis — INKAR Gruppe 236.

    Feeds the frontend's "ich verdiene ortsüblich" budget mode: the rent
    ceiling becomes share × local pay instead of a fixed € amount.
    """
    grp = "236"
    avail = _inkar_json(
        "Wizard/GetM%C3%B6glich",
        {"IndicatorCollection": [{"Gruppe": grp}],
         "TimeCollection": "", "SpaceCollection": [{"level": "KRE"}]},
        cache_key=f"avail_g{grp}_kre")
    latest = max(avail["Möglich"], key=lambda t: t["Zeit"])
    rows = _inkar_json(
        "Table/GetDataTable",
        {"IndicatorCollection": [{"Gruppe": grp}],
         "TimeCollection": [{"group": latest["Gruppe"],
                             "indicator": latest["IndID"],
                             "level": "KRE", "time": latest["ZeitID"]}],
         "SpaceCollection": [{"level": "KRE"}], "pageorder": "1"},
        cache_key=f"pay_g{grp}_kre_{latest['ZeitID']}")["Daten"]
    pay = pd.Series({r["Schlüssel"]: float(r["Wert"]) for r in rows
                     if r.get("Wert") is not None})
    print(f"inkar pay: {len(pay)} Kreise, year {latest['Zeit']}, "
          f"median {pay.median():.0f} €/Monat")
    return pay


def main():
    grid = pd.read_parquet(INTERIM / "grid.parquet")
    base, fill = zensus_base()

    out = grid[["h3"]].copy()
    out["rent_2022"] = out["h3"].map(base)
    out["rent_2022"] = out["rent_2022"].fillna(out["h3"].map(fill))

    demo_f = LAYERS / "demographics.parquet"
    pop = out[["h3"]].merge(
        pd.read_parquet(demo_f)[["h3", "population"]], on="h3", how="left"
    )["population"].fillna(0) if demo_f.exists() else pd.Series(1.0, index=out.index)
    pop_s = pd.Series(pop.values, index=out["h3"])

    vg = load_gemeinden()
    pts = gpd.GeoDataFrame(grid[["h3"]],
                           geometry=gpd.points_from_xy(grid["lon"], grid["lat"]),
                           crs=4326)
    hex_gem = gpd.sjoin(pts, vg, how="left")[["h3", "AGS"]].rename(
        columns={"AGS": "ags"}).drop_duplicates("h3")

    rent_series = out.set_index("h3")["rent_2022"]
    source = "inkar"  # only open-data source in the public build
    try:
        f = inkar_factor(vg, hex_gem, rent_series, pop_s)
        if f.notna().sum() < 50:
            raise RuntimeError(f"only {f.notna().sum()} factors")
        hex_f = hex_gem["ags"].map(f)
        hex_f = pd.Series(hex_f.values, index=hex_gem["h3"]).reindex(out["h3"])
        med = float(np.nanmedian(hex_f))
        out["uplift"] = hex_f.fillna(med).values
        print(f"uplift: median {med:.2f}, range "
              f"{np.nanmin(hex_f):.2f}..{np.nanmax(hex_f):.2f}")
    except Exception as e:
        print(f"WARNING: {source} calibration failed ({e}); "
              f"constant uplift {UPLIFT_FALLBACK}")
        out["uplift"] = UPLIFT_FALLBACK

    # local median pay per hex (Kreis level) for the frontend's
    # "ortsüblich verdienen" budget mode; missing Kreise -> bbox median
    try:
        pay = inkar_kreis_pay()
        kp = pd.Series(hex_gem["ags"].str[:5].map(pay).values,
                       index=hex_gem["h3"]).reindex(out["h3"])
        sigma = float(os.environ.get("PAY_SIGMA_KM", PAY_SIGMA_KM))
        if sigma > 0:
            k = max(1, round(2.0 * sigma / RES8_STEP_KM))
            kp = disk_gaussian_mean(pd.Series(kp.values, index=out["h3"]),
                                    k=k, sigma_km=sigma).reset_index(drop=True)
            print(f"kreis_pay: blended sigma={sigma:.1f} km (disk k={k}), "
                  f"range {np.nanmin(kp):.0f}..{np.nanmax(kp):.0f} €/Monat")
        out["kreis_pay"] = kp.fillna(float(kp.median())).values
    except Exception as e:
        print(f"WARNING: kreis_pay failed ({e}); column omitted")

    # calibrated rent on data cells only; the kernel+prior produces the final
    # surface for every cell, so no separate fill chain is needed
    cal_raw = out["rent_2022"] * out["uplift"]
    n_nan = cal_raw.isna().sum()

    # Gemeinde prior: population-weighted mean of calibrated data cells,
    # Kreis mean where a Gemeinde has no data
    prior_df = pd.DataFrame({
        "h3": out["h3"], "cal": cal_raw.values, "pop": pop.values,
    }).merge(hex_gem, on="h3", how="left")
    pd_data = prior_df.dropna(subset=["cal"])
    pd_data = pd_data[pd_data["pop"] > 0]
    gem_prior = (pd_data.groupby("ags")
                 .apply(lambda g: np.average(g["cal"], weights=g["pop"]),
                        include_groups=False))
    kreis_prior = gem_prior.groupby(gem_prior.index.str[:5]).median()
    prior_per_hex = prior_df["ags"].map(gem_prior)
    prior_per_hex = prior_per_hex.fillna(prior_df["ags"].str[:5].map(kreis_prior))
    prior_s = pd.Series(prior_per_hex.values, index=out["h3"])

    out["rent_cal"] = kernel_shrink(
        pd.Series(cal_raw.values, index=out["h3"]), pop_s, prior_s).values
    # The web recomputes the affordability score client-side from rent_cal +
    # the user's budget; the pipeline ships only the raw €/m² (rent_cal) and the
    # Kreis pay level (kreis_pay) for the "% vom Lohn" budget mode.

    LAYERS.mkdir(parents=True, exist_ok=True)
    out.to_parquet(LAYERS / "rent.parquet", index=False)
    print(f"rent: {n_nan} hexes without own data (estimated via kernel+prior)")
    print(out[["rent_2022", "rent_cal"]].describe().round(2).to_string())


if __name__ == "__main__":
    main()
