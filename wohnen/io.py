"""IO helpers shared across pipeline scripts."""

from pathlib import Path

import pandas as pd


def write_parquet_if_changed(df: pd.DataFrame, path, sort_cols=None) -> bool:
    """Write df to `path` only if its content differs from the file already there;
    return True if written, False if left untouched.

    Used for the minimal inputs to the expensive r5py routing steps (population.parquet,
    freizeit_spots.parquet). make keys on mtime, so a no-op rewrite of one of these would
    needlessly re-route. Preserving the old file's mtime when the values are identical lets
    a broad upstream layer (demographics, pois) regenerate without invalidating routing —
    the route only re-runs when the routed values actually change.

    Comparison is value-based (dtype + NaN-aware via DataFrame.equals), column-order
    insensitive, and row-order insensitive when `sort_cols` is given. Any read/compare
    failure falls through to a write (the safe default).
    """
    path = Path(path)
    df = df.reset_index(drop=True)
    if path.exists():
        try:
            old = pd.read_parquet(path)
            if set(old.columns) == set(df.columns):
                a, b = old, df[list(old.columns)]
                if sort_cols:
                    a = a.sort_values(sort_cols).reset_index(drop=True)
                    b = b.sort_values(sort_cols).reset_index(drop=True)
                if a.equals(b):
                    return False
        except Exception:
            pass
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return True
