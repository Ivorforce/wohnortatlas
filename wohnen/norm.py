"""Normalization helpers; all scores end up in [0, 1], higher = better."""

import numpy as np


def clip01(x):
    return np.clip(x, 0.0, 1.0)


def anchor_norm(x, worst: float, best: float):
    """Linear map: worst -> 0, best -> 1 (works for inverted ranges)."""
    return clip01((np.asarray(x, dtype=float) - worst) / (best - worst))
