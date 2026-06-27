"""Cityness opportunity weight O — a pure function of catchment_pop (the smoothed 3-km
population mass from 07a). 04b samples it at the selected population peaks to weight each
center; 04f evaluates it at EVERY cell for the Anbindung "native value" self-term (a cell's
own cityness, counted at full reachability). One definition so the per-center and per-cell
weights can never drift apart.
"""

import numpy as np

C_HALF = 25_000        # o_any half-saturation: O = c / (c + C_HALF)
# o_gross smoothstep bounds, on CATCHMENT (3-km weighted pop ≈ local agglomeration, NOT
# inhabitants — catchment runs ~⅓–½ of a city's headcount and saturates for big cities). So
# these are catchment values, not population: ~20k catchment ≈ a dense Mittelstadt (ramp start),
# ~55k ≈ a solid Großstadt (saturated). The town→Großstadt transition is the gradient; metros
# all reach 1. (An earlier 38k/160k read these as inhabitant counts — too high as catchment, it
# zeroed real but spread-out Großstädte like Jena.)
GROSS_LO, GROSS_HI = 20_000, 55_000


def smoothstep(x, lo, hi):
    t = np.clip((np.asarray(x, float) - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def cityness_o(catchment_pop, tier):
    """Per-location cityness O for a tier: 'any' = c/(c+C_HALF) (includes towns),
    'gross' = smoothstep(c, GROSS_LO, GROSS_HI) (gated to metropolises)."""
    c = np.asarray(catchment_pop, float)
    if tier == "any":
        return c / (c + C_HALF)
    if tier == "gross":
        return smoothstep(c, GROSS_LO, GROSS_HI)
    raise ValueError(f"unknown cityness tier {tier!r}")
