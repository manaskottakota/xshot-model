"""Temporal season splits."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SeasonSplitMasks:
    train: pd.Series
    val: pd.Series
    test: pd.Series


def masks_from_seasons(
    seasons: pd.Series,
    *,
    train: set[str],
    val: set[str],
    test: set[str],
) -> SeasonSplitMasks:
    s = seasons.astype(str)
    tr = s.isin(train)
    va = s.isin(val)
    te = s.isin(test)
    disjoint = not (tr & va).any() and not (tr & te).any() and not (va & te).any()
    if not disjoint:
        raise ValueError("train/val/test season sets must be disjoint")
    if not (tr | va | te).all():
        dropped = (~(tr | va | te)).sum()
        raise ValueError(f"{dropped} rows have seasons outside train/val/test split")
    return SeasonSplitMasks(train=tr, val=va, test=te)
