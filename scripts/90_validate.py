"""Executable ground truth: spot checks accumulated from debugging sessions.

Every check here corresponds to a bug found by eye or a calibration anchored
to local knowledge. Run after scoring changes; WARNs are real regressions
until proven otherwise. Exit code 1 if any check fails.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h3
import numpy as np
import pandas as pd

from wohnen.config import LAYERS

FAILED = []


def cell(lat, lon):
    return h3.latlng_to_cell(lat, lon, 8)


def check(name, value, lo, hi, unit=""):
    ok = lo <= value <= hi
    print(f"{'PASS' if ok else 'WARN':4s} {name:55s} {value:8.2f}{unit}"
          f"  (expect {lo}..{hi})")
    if not ok:
        FAILED.append(name)


def school_completeness():
    """Are (almost) all schools present + typed? Run on the authoritative typed set
    (03c_schools: OSM locations typed from JedeSchule CC0, OSM name-typing fallback).

    Track counts are EFFECTIVE buckets (a combined form like Gesamt-/Gemeinschaftsschule
    joins several tracks), so they sit ABOVE the pure Destatis Schulart counts
    (Grundschule 15,566; Gymnasium ~3,100; Realschule 1,699; IGS ~2,210 — destatis.de
    table 21111). Bands here catch a track COLLAPSE, not precise calibration.
    PER-KREIS: no Kreis should be a school desert (a whole-Kreis hole = a typing/coverage
    gap, e.g. Jena showing no Realschulabschluss before the Regelschule fix).
    """
    import zipfile

    import geopandas as gpd

    from wohnen.config import INTERIM, RAW

    sp_path = INTERIM / "schools_points.parquet"
    if not sp_path.exists():
        print("WARN school completeness skipped (no schools_points.parquet)")
        return
    sp = pd.read_parquet(sp_path)

    print("== schools: typed coverage (effective track buckets) ==")
    check("Grundschule points (Destatis 15,566)", int(sp["grund"].sum()), 13000, 18500)
    check("Gymnasium-track points (Destatis Gym+IGS ~5,300)", int(sp["gym"].sum()), 3500, 7000)
    check("Realschul-track points (effective)", int(sp["real"].sum()), 3500, 7000)
    check("Mittelschul-track points (effective)", int(sp["mittel"].sum()), 2800, 7000)

    shp = next(n for n in zipfile.ZipFile(RAW / "vg250.zip").namelist()
               if n.endswith("VG250_KRS.shp"))
    krs = gpd.read_file(f"zip://{RAW / 'vg250.zip'}!{shp}").to_crs(4326)
    krs = krs[krs["GF"] == 4]  # land only; GF==2 are coastal-water dupes (same AGS)
    gdf = gpd.GeoDataFrame(sp, crs=4326,
                           geometry=gpd.points_from_xy(sp["lon"], sp["lat"]))
    gdf["sec"] = gdf[["gym", "real", "mittel"]].any(axis=1)
    j = gpd.sjoin(gdf, krs[["AGS", "geometry"]], how="inner", predicate="within")
    per = j.groupby("AGS")[["grund", "sec"]].any()
    n_krs = len(krs)  # Kreise with zero schools never appear in `per` -> count 0
    print("== schools: per-Kreis spatial coverage (no school desert) ==")
    check("Kreise with >=1 Grundschule", per["grund"].sum() / n_krs, 0.97, 1.0)
    check("Kreise with >=1 secondary school", per["sec"].sum() / n_krs, 0.97, 1.0)


def main():
    sc = pd.read_parquet(LAYERS / "scores.parquet").set_index("h3")

    def v(col, lat, lon):
        return float(sc.loc[cell(lat, lon), col])

    print("== rent (calibrated asking, €/m²) ==")
    # € bounds calibrated at the INKAR uplift level, ~10 % above the legacy
    # Homeday level (which level is right is unresolved;
    # rent is used as a comparison layer, so bounds follow the source level)
    check("Glockenbach expensive", v("rent_cal", 48.1294, 11.5696), 18, 27)
    check("Schwabing dense urban", v("rent_cal", 48.1664, 11.5879), 18, 27)
    check("Dachau mid", v("rent_cal", 48.2538, 11.4341), 12, 18)
    check("rural Bockhorn cheap", v("rent_cal", 48.3107, 11.9962), 5, 13)
    print("   -- historical bug sites (sinkholes/seams) --")
    for name, (lat, lon) in {"Geiselgasteig": (48.0465, 11.5550),
                             "Neuherberg": (48.2225, 11.5953)}.items():
        cells = list(h3.grid_disk(cell(lat, lon), 2))
        vals = [sc.loc[c, "rent_cal"] for c in cells if c in sc.index]
        check(f"{name} neighborhood spread (max-min)",
              max(vals) - min(vals), 0, 7, " €")

    print("== noise (0-1) ==")
    # airport_penalty is now official END aircraft Lden (LfU flughaefen WMS),
    # not the old perception-tuned corridor model. Lden is a 24 h energy
    # average, so intermittent flyovers that feel loud (high per-event Lmax)
    # average well below contour: Attaching is mapped at ~61 dB Lden (hex
    # energy-mean ~0.25 as the village straddles the 55 dB edge), and Pulling
    # is below 55 dB (NoData -> 0) despite audible single flyovers.
    check("Schwabing loud (urban)", v("noise_penalty", 48.1664, 11.5879), 0.7, 1.0)
    check("Attaching under approach", v("airport_penalty", 48.3661, 11.7670), 0.15, 0.6)
    check("Pulling low Lden (flyovers, not avg)", v("airport_penalty", 48.3793, 11.7307), 0, 0.15)
    check("Freising Altstadt no airport", v("airport_penalty", 48.4028, 11.7489), 0, 0.10)
    check("Schwaig S of runway affected", v("airport_penalty", 48.3475, 11.8000), 0.2, 0.9)
    check("rural quiet", v("noise_penalty", 48.3107, 11.9962), 0, 0.35)

    print("== nature (crowding-discounted) ==")
    # bands shifted down 2026: s_nature re-anchored above its universal floor (13_nature
    # NATURE_FLOOR) so the worst cells reach 0 and the layer bites as a geo-mean pref.
    # The RELATIVE order (high > mid, Geretsried >= Schwabing) is the real invariant.
    check("Herrsching (Ammersee) high", v("s_nature", 48.0036, 11.1782), 0.45, 0.85)
    check("Schwabing dense urban mid", v("s_nature", 48.1664, 11.5879), 0.08, 0.35)
    n_ger = v("s_nature", 47.8572, 11.4811)
    n_sch = v("s_nature", 48.1664, 11.5879)
    check("Geretsried >= Schwabing", n_ger - n_sch, -0.02, 1)
    # Ismaninger Speichersee: an engineered reservoir / bird sanctuary, not a swim/
    # picnic lake — demoted via water_recr_q (09/13) so it reads as PARTIAL nature
    # (open water + green, no usable outing), well below a real lake.
    n_speicher = v("s_nature", 48.2195, 11.7612)
    check("Speichersee partial (engineered water)", n_speicher, 0.25, 0.60)
    check("Herrsching (real lake) > Speichersee",
          v("s_nature", 48.0036, 11.1782) - n_speicher, 0.10, 1)
    # Lüneburger Heide (Undeloh): premier Calluna heath. WorldCover labels it grassland,
    # so heath is sourced from OSM natural=heath (03 → 09) and reweighted to forest tier
    # (W .80, 13). Guards the fix — without it the Heide core read as plain grass (~0.64).
    check("Lüneburger Heide (Undeloh) premier heath", v("s_nature", 53.18, 9.99), 0.75, 1.0)

    print("== green (land-green / non-water surface, noisy-OR water bonus, × built^2) ==")
    # green_land = veg as a fraction of the NON-WATER surface, noisy-OR'd with a capped
    # water bonus (a treeless riverbank no longer reads as green as a forest; pure water
    # floors ~0.2), then a mild built^2 sealing penalty for the park/canopy overlay leak.
    # cropland LOW (open != green); leafy-but-built reads green; paved centre low;
    # forest/leafy-residential high.
    check("Lechfeld cropland low (open != green)", v("s_green", 48.205, 10.86), 0.25, 0.55)
    check("Moosham pasture+crop below forest", v("s_green", 47.9967, 12.2123), 0.45, 0.80)
    check("Höhenkirchner Forst forest high", v("s_green", 48.008, 11.717), 0.85, 1.0)
    check("Grünwald leafy residential high", v("s_green", 48.043, 11.527), 0.70, 1.0)
    # leafy-but-built must read green: canopy carries a half-sealed tree-lined
    # Viertel (built^2 penalty bites lightly here), unlike a paved centre.
    check("Schwabing-Nord leafy-but-built green", v("s_green", 48.1714, 11.5879), 0.65, 0.95)
    check("Marienplatz dense center low (sealed)", v("s_green", 48.137, 11.575), 0.0, 0.45)
    # the layer must discriminate: not "almost all rural is perfect" (was 91 %). Floor
    # cut 0.40→0.30 with the land-green split: only forest (~35 %) now maxes, while
    # agriculture-heavy cropland correctly sits below 0.9 (open != green) and waterside
    # rural no longer maxes on water alone — ~60 % of rural still reads >0.7.
    rural = sc[sc["builtup_share"].fillna(0) < 0.05]
    check("rural cells maxed-out (built<5%, s_green>0.9)",
          (rural["s_green"] > 0.9).mean(), 0.30, 0.85)

    print("== ortsbild (heritage noisy-OR + street grain, gated by built-up) ==")
    # street grain: organic Altstadt high, planned grid ~0 (orientation entropy, 17)
    check("Wasserburg organic streets high grain", v("street_grain", 48.0590, 12.2290), 0.70, 1.0)
    check("Maxvorstadt 19c grid low grain", v("street_grain", 48.1510, 11.5660), 0.0, 0.25)
    # whole layer: heritage OR grain — Eichstätt has both; an organic town with
    # modest heritage (Pfaffenhofen) is lifted by grain; rural stays low.
    check("Eichstätt Altstadt character high", v("s_character", 48.8916, 11.1844), 0.80, 1.0)
    check("Pfaffenhofen organic town lifted", v("s_character", 48.5310, 11.5060), 0.55, 1.0)
    check("rural Bockhorn low character", v("s_character", 48.3100, 11.9600), 0.0, 0.30)

    print("== leisure (going-out venue density) ==")
    # s_leisure (the old bundled leisure score) + ent_score/ent_diversity were
    # dropped 2026-06; the web Freizeit signal now lives in the reach_* gravity
    # (04e), built from ent_density. Guard the raw going-out density that feeds
    # it: absolute supply still leads — München center is top-tier and clears a
    # town edge by a wide margin (the old "not pop-penalized" + ">Freising"
    # regressions, restated on the surviving signal).
    inh_ed = sc[sc["population"].fillna(0) > 0]["ent_density"]
    p95 = float(inh_ed.quantile(0.95))
    check("München center going-out density top-tier",
          v("ent_density", 48.137, 11.575), p95, 1e9)
    muc = v("ent_density", 48.1715, 11.5879)
    fre = v("ent_density", 48.3919, 11.7661)
    check("München going-out density > Freising edge", muc - fre, 0.0, 1e9)

    print("== familie & alltag (supply-relative crowding + provisioning) ==")
    # Kita crowding is supply-relative, not raw headcount: Kita/Grundschule
    # scale ~1:1 with population, so the DENSEST area must not be the most
    # penalized (the old raw-pop −20 % bug). München's per-1000 supply sits at
    # the regional norm or above → its crowding discount ≈ 0.
    rf_muc = 1000 * v("kita_supply", 48.137, 11.575) / v("catchment_pop", 48.137, 11.575)
    check("München Kita/school supply ≥ norm", rf_muc, 0.70, 1.5)
    # daily provisioning sees the Frischeversorgung (Bäcker/Metzger/Hofladen), not just
    # the Vollsortimenter: in a supermarket-less village the nearest fresh-food shop is
    # usually CLOSER than the (far) full grocery, so the web "In der Nähe" credits the
    # capped gap-fill food access it would otherwise miss — guards the shipped
    # t_frische_min row against being silently dropped (it would collapse this toward 0).
    inh = sc[sc["population"].fillna(0) > 0]
    novoll = inh[inh["t_vollsort_min"] > 15]
    closer = novoll["t_frische_min"] < novoll["t_vollsort_min"]
    check("Frische closer than Vollsortiment in villages",
          float(closer.mean()), 0.40, 0.90)
    # secondary schooling counts all Sek-I tracks, not just Gymnasium: in a big
    # share of cells a Real-/Mittelschule is the nearest secondary, so dropping
    # them back into "other" (the pre-2026-06 state) would collapse this.
    sek_closer = float(((np.minimum(sc["t_realschule_min"], sc["t_mittelschule_min"])
                         < sc["t_gymnasium_min"] - 2)).mean())
    check("Real-/Mittelschule broaden secondary coverage", sek_closer, 0.30, 0.80)
    school_completeness()

    print("== flood (JRC/Copernicus river-flood depth) ==")
    check("Wasserburg Altstadt (Inn) floods deep",
          v("flood_depth_hq100", 48.057, 12.220), 2.0, 12.0)
    check("Wasserburg flood penalty (frequent)",
          v("flood_score", 48.057, 12.220), 0.2, 0.75)
    check("Isar floodplain Freising near-veto",
          v("flood_score", 48.395, 11.740), 0.0, 0.25)
    check("Kolbermoor (Mangfall 2013) floods",
          v("flood_score", 47.849, 12.067), 0.30, 0.85)
    check("Schwabing dry", v("flood_score", 48.1664, 11.5879), 0.85, 1.0)
    # hex-share semantics: a hex flags when PART of it floods, so these run
    # above address-level exposure (GDV: ~10 % of addresses) — settlements
    # historically hug rivers. (JRC is defense-agnostic, so embanked corridors
    # show residual hazard — accepted continental-model limitation.)
    pen = 1 - sc[sc["inhabited"]]["flood_score"]
    check("inhabited cells w/ mild+ flood risk", float((pen > 0.1).mean()), 0.08, 0.25)
    check("inhabited cells w/ severe flood risk", float((pen > 0.5).mean()), 0.01, 0.09)

    print("== vacancy (Zensus 2022 Leerstand, 1km) ==")
    # Data sanity: covered cells must match Germany's real ~5 % vacancy, not the
    # 100m small-denominator noise (median ~23 %) we deliberately avoid by reading 1km.
    cov = sc[sc["vacancy_pct"].notna()]
    check("covered cells median vacancy %",
          float(cov["vacancy_pct"].median()), 3.0, 7.0, " %")
    # Decline tail scores low: East-German Plattenbau / shrinking cores (real 2022
    # vacancy), the independent signal this layer exists to carry.
    check("Halle-Neustadt Plattenbau high vacancy",
          v("vacancy_pct", 51.4790, 11.9210), 12.0, 25.0, " %")
    check("Halle-Neustadt decline penalty",
          v("s_vacancy", 51.4790, 11.9210), 0.0, 0.35)
    check("Dessau shrinking core floors",
          v("s_vacancy", 51.8330, 12.2450), 0.0, 0.25)
    check("Gera moderate decline (TH)",
          v("s_vacancy", 50.8800, 12.0800), 0.15, 0.55)
    # Documented limitations, pinned so a future "fix" can't silently break them:
    # (1) a TIGHT market reads healthy regardless of internal poverty — Hasenbergl
    # (München social-housing) has ~0 vacancy because every unit is occupied, so the
    # layer is blind to intra-metro stigma (by design, not a bug). (2) rural single-
    # family areas carry no 1km value -> neutral 1.0 (silence is not decline).
    check("München Hasenbergl: tight market reads healthy",
          v("s_vacancy", 48.2107, 11.5616), 0.85, 1.0)
    check("rural no-data neutral (Bockhorn)",
          v("s_vacancy", 48.3000, 11.9700), 0.99, 1.0)

    print("== climate (DWD 1991-2020) ==")
    check("Alpenrand wet (Miesbach)", v("rain_mm", 47.789, 11.834), 1200, 1800)
    check("Donau plain dry (Ingolstadt N)", v("rain_mm", 48.786, 11.42), 550, 800)
    check("Munich sunny", v("sun_h", 48.137, 11.575), 1750, 1900)
    check("Alpenrand snowy (Miesbach)", v("snow_days", 47.789, 11.834), 55, 120)
    # de Martonne bottoms out in the mitteldeutsches Trockengebiet (Harz lee,
    # Mansfelder Land ~51.4N,11.1E): ~21, the real German floor. The too-dry
    # gate is meant to fire there (mild penalty by design, see 15_climate
    # DRY_ANCHORS) — not be inert. Guard the floor instead: below ~18 would be
    # semi-arid (impossible in Germany -> corruption); above ~25 would mean the
    # drought grid isn't being read / over-smoothed away.
    check("dry-aridity floor in Trockengebiet", float(sc["martonne"].min()), 18, 25)

    print("== structure ==")
    pop = sc["population"].fillna(0)
    check("inhabited share (pop>=10)", float((pop >= 10).mean()), 0.40, 0.55)
    # bleeding metric: uninhabited cells vs populated neighbours
    rent = sc["rent_cal"]
    diffs = []
    for c in sc.index[::3]:
        if pop.get(c, 0) > 0:
            continue
        nb = [n for n in h3.grid_ring(c, 1)
              if n in sc.index and pop.get(n, 0) >= 50]
        if len(nb) >= 2:
            diffs.append(rent[c] - float(np.mean([rent[n] for n in nb])))
    check("uninhabited-cell rent bleeding (P95 |dev|)",
          float(np.percentile(np.abs(diffs), 95)), 0, 1.5, " €")

    score_cols = [c for c in sc.columns if c.startswith("s_")]
    # 9 = the shipped factor layers (incl. s_oepnv and s_vacancy). Miete/Nähe/Freizeit
    # are recomputed client-side from raw columns, not stored as s_* scores; s_density
    # and s_age are client-side pseudo-layers (not in scores.parquet).
    check("score layers present", len(score_cols), 9, 9)

    print(f"\n{len(FAILED)} failures" if FAILED else "\nall checks passed")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
