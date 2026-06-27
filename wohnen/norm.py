"""Normalization helpers; all scores end up in [0, 1], higher = better."""

import numpy as np
import pandas as pd


def clip01(x):
    return np.clip(x, 0.0, 1.0)


def anchor_norm(x, worst: float, best: float):
    """Linear map: worst -> 0, best -> 1 (works for inverted ranges)."""
    return clip01((np.asarray(x, dtype=float) - worst) / (best - worst))


def log_p99_norm(x):
    """log1p then divide by P99 of log1p (heavy-tailed densities)."""
    lx = np.log1p(np.asarray(x, dtype=float))
    p99 = np.nanpercentile(lx[lx > 0], 99) if (lx > 0).any() else 1.0
    return clip01(lx / max(p99, 1e-9))


def pctl_rank(s: pd.Series) -> pd.Series:
    return s.rank(pct=True)
