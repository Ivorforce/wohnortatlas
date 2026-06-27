"""Shared spec for the s_freizeit point SOURCES.

Single source of truth for both the routing (04d_swim reverse-routes each source) and the
extraction (03b_freizeit_spots pulls just these rows out of the full pois.parquet into the
minimal freizeit_spots.parquet that 04d depends on). Keeping the spec here is what lets the
two stay in sync without 03b importing 04d's routing module.

(category, subcategory, mass) per leisure source. mass scales the spot's pull; Badesee keeps
the 0.7 seasonal discount (cf. 12_schools_family swim mode). Each is a lumpy, place-specific
activity that separates a crowd (NOT a pop-density or rural echo): Kino (one-per-town),
Klettern-Hallen, Golf. The web adds each as a weight-only s_freizeit checkbox source.
(Reiten/Wintersport/Segeln were considered and rejected.)"""

import numpy as np

SOURCES = {
    "swim": [("family", "pool", 1.0), ("family", "badesee", 0.7)],
    "kino": [("entertainment", "cinema", 1.0)],
    "klettern": [("leisure", "klettern", 1.0)],
    "golf": [("leisure", "golf", 1.0)],
}

# the distinct (category, subcategory) pairs to extract — derived from SOURCES so it can
# never drift from what 04d actually routes.
SOURCE_PAIRS = {(cat, sub) for spec in SOURCES.values() for cat, sub, _ in spec}

# --- point-source reachability model (shared by 04d route + 04e derive) ------
# A "need just ONE" facility (a pool, a cinema): you only need to reach the NEAREST one,
# but variety adds happiness and distance must ALWAYS pay (a million cinemas an hour out
# is never a 10/10). A plain gravity SUM fails the last point (count saturates regardless
# of distance); nearest-only fails the variety point. So:
#
#   d(t)  = exp(−( max(0, t−T0) / SIG )^P )         per-spot distance decay
#   d_max = d(t_nearest)                            the CEILING — distance can't be bought off
#   raw   = Σ_spot mass·d(t)                        distance-weighted "how much choice"
#   reach = d_max · [ B + (1−B)·(1 − exp(−(raw − d_max)/K)) ]
#
# d(t) is a FLAT top to T0 (sub-cell distance pays nothing) then a generalized Gaussian
# (P between exp=1 and Gaussian=2: a gentle shoulder, a firm tail). The ceiling makes
# distance a hard wall variety can fill UP TO but never past — infinite count → d_max < 1
# for any t > T0. Single spot → B·d_max (one cinema next door ≈ B). Tuning, with examples:
#   1 next door 0.60 · 20@2min 1.00 · 10@10min 0.90 · 1@30min 0.23 · 100@1h 0.11 · 1e6@1h 0.11
# T0/SIG/P are baked into raw at route time (04d) → re-route to change; B/K are applied in
# 04e (route-free). The ceiling d_max is recomputed from the persisted nearest time, so it
# also re-tunes route-free.
POINT_T0 = 5.0     # free radius (min): ≤ T0 → full credit (grid cells are ~5 min wide)
POINT_SIG = 25.4   # decay width (min)
POINT_P = 1.385    # shape exponent (1 = exponential, 2 = Gaussian)
POINT_B = 0.6      # value of a single reachable spot, as a fraction of its ceiling
POINT_K = 1.5      # variety fill rate (distance-weighted extra spots to saturate)


def point_decay(t):
    """Per-spot distance decay d(t) ∈ (0,1]; flat to POINT_T0 then generalized Gaussian.
    Accepts a scalar or numpy array of minutes (inf/NaN → 0)."""
    t = np.asarray(t, dtype=float)
    x = np.maximum(0.0, t - POINT_T0) / POINT_SIG
    with np.errstate(over="ignore"):
        d = np.exp(-(x ** POINT_P))
    return np.where(np.isfinite(t), d, 0.0)


def point_reach(t_min, raw):
    """Final [0,1] point-source surface from the persisted nearest-spot time (t_min, min;
    inf = unreachable) and the gravity sum raw = Σ mass·point_decay(t). The ceiling d_max
    comes from t_min (so it re-tunes route-free); variety fills B·d_max → d_max."""
    d_max = point_decay(t_min)
    surplus = np.maximum(0.0, np.asarray(raw, dtype=float) - d_max)
    variety = 1.0 - np.exp(-surplus / POINT_K)
    return d_max * (POINT_B + (1.0 - POINT_B) * variety)
