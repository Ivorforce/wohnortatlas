"""Shared M/B/O field-target derivation — job sectors (04h) and cityness aggregates (04f).

A field Anbindung target ships, per cell + mode, three u8 surfaces that the client turns into
a budget-dependent score (recomputeAnbindungMBO):
  M = minutes to the BEST opportunity reachable within B_WINDOW (the app's min budget) — not
      the nearest (latches onto a far weak centre) nor the global best (latches onto a distant
      city). Also the reachEff / over-budget filter.
  B = that centre's opportunity, time-penalized — the FLOOR (score at T = M).
  O = 1 − Π_centres(1 − opportunity·decay) — a NOISY-OR over everything reachable within HORIZON
      (2 h): saturating, bounded [0,1], always ≥ B. Far reach counts; reaching MORE lifts O.
The client scores B + (T−M)/(HORIZON−M)·(O−B), best-wins over modes. The cell's own field mass
enters as a self centre at decay 1 ("you're already here"). Derived route-free from a per-centre
opportunity weight + per-cell native weight; the caller supplies the reach times.
"""

import numpy as np

HORIZON = 120.0    # 2 h ceiling (min) = the routed-time cap (TRAVEL_SENTINEL_MIN); O's horizon
B_WINDOW = 25.0    # M/B come from the best opportunity within this commute (the app's min budget)
NOR_CHUNK = 128    # centre-axis chunk for the noisy-OR product Π(1−term) (memory only)


def decay(t):
    """Per-centre distance weight (HORIZON−t)/HORIZON ∈ [0,1] (≥HORIZON / 255-sentinel → 0)."""
    return np.clip((HORIZON - np.asarray(t, np.float32)) / HORIZON, 0.0, 1.0)


def mbo_triple(term, t, native, arange):
    """(M, B, O) u8 per cell for ONE (mode, field) surface.

    term   = opportunity·decay over centres (C, N) — the caller's reused score buffer.
    t      = the mode's per-centre×cell minutes (C, N) uint8 (≥HORIZON / 255 = unreachable).
    native = the cell's own opportunity at decay 1 (N,) — a self centre.

    M, B come from the BEST opportunity within B_WINDOW; if nothing with REAL opportunity
    (term>0) is in the window, they fall back to the NEAREST centre that actually HAS the
    opportunity (not just any centre — else a 5k town with no Großstadt scores a faint 0 off a
    2 h-distant one and never filters). A cell with no opportunity centre reachable + no native
    is absent (255) → filtered. Returns M (minutes, 255 = none), B/O (×254, 255 = none)."""
    C, N = term.shape
    # ONE pass over centres: the noisy-OR ceiling Π(1−term), the BEST opportunity within the
    # window (max term, t ≤ B_WINDOW), AND the NEAREST opportunity overall (min t where term>0).
    prod = (1.0 - native).astype(np.float64)                    # noisy-OR self centre (t=0)
    bw_term = np.full(N, -1.0, np.float32)                      # best within-window term (-1 = none)
    bw_time = np.full(N, 255.0, np.float32)
    no_t = np.full(N, np.inf, np.float32)                       # nearest-opportunity minutes
    no_term = np.zeros(N, np.float32)
    for i in range(0, C, NOR_CHUNK):
        blk = term[i:i + NOR_CHUNK]
        tb = t[i:i + NOR_CHUNK]
        prod *= np.prod(1.0 - blk, axis=0)
        opp = blk > 0.0                                         # centre actually has opportunity here
        win = np.where(opp & (tb <= B_WINDOW), blk, -1.0)
        a = win.argmax(axis=0); mx = win[a, arange]
        u = mx > bw_term
        bw_term[u] = mx[u]; bw_time[u] = tb[a, arange][u].astype(np.float32)
        topp = np.where(opp, tb.astype(np.float32), np.inf)
        b = topp.argmin(axis=0); mn = topp[b, arange]
        u = mn < no_t
        no_t[u] = mn[u]; no_term[u] = blk[b, arange][u]
    onor = 1.0 - prod                                           # ceiling, ≥ max(B, native)
    has_win = bw_term >= 0.0                                    # real opportunity within the window
    has_opp = no_t < HORIZON                                    # any opportunity reachable at all
    bterm = np.where(has_win, bw_term, np.where(has_opp, no_term, 0.0))
    mc = np.where(has_win, bw_time, np.where(has_opp, no_t, 255.0))
    use_nat = (native >= bterm) & (native > 0.0)               # you're in a bigger local market
    present = (mc < HORIZON) | (native > 0.0)
    mval = np.where(use_nat, 0.0, mc)
    bval = np.maximum(bterm, native)
    M = np.where(present, np.clip(np.round(mval), 0, 254), 255).astype(np.uint8)
    B = np.where(present, np.clip(np.round(bval * 254), 0, 254), 255).astype(np.uint8)
    O = np.where(present, np.clip(np.round(onor * 254), 0, 254), 255).astype(np.uint8)
    return M, B, O
