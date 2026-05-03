"""Quantify how much each Stage 4 condition steered behaviour relative to vanilla.

`steering_rate` returns the percentage of valid candidates whose feature value
moved in the *advised* direction relative to vanilla's median. The advised
direction is taken from the Stage 1 Spearman analysis and supplied by the
caller — neutral feedback uses no explicit direction, so we measure whether
the implicit signal still lands.
"""
from __future__ import annotations

import pandas as pd

_DIRECTIONS = ('up', 'down')


def steering_rate(df: pd.DataFrame, *, feature: str,
                  condition: str, vanilla: str,
                  direction: str) -> float:
    """Return the percentage of `condition`'s candidates whose `feature` value
    is on the advised side of vanilla's median.

    df must have columns 'condition' and f'bm_{feature}'.
    """
    if direction not in _DIRECTIONS:
        raise ValueError(f'direction must be one of {_DIRECTIONS}, got {direction!r}')

    col = f'bm_{feature}'
    vanilla_median = df.loc[df['condition'] == vanilla, col].dropna().median()
    cand = df.loc[df['condition'] == condition, col].dropna()
    if len(cand) == 0:
        return float('nan')

    if direction == 'up':
        moved = (cand > vanilla_median).sum()
    else:
        moved = (cand < vanilla_median).sum()
    return 100.0 * moved / len(cand)
