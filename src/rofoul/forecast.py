"""Turning predicted fouling paths into days-to-clean and alert levels."""
from __future__ import annotations

import math

import numpy as np


def days_to_threshold(level: float, path: np.ndarray, threshold: float, max_extrapolation_days: float = 180.0
                      ) -> tuple[float | None, bool]:
    """Days until `path` first reaches `threshold`.

    level : current indicator value (day 0)
    path  : predicted values for h = 1..H days
    Returns (days, beyond_horizon).  days is fractional (linear interpolation
    between days), 0 if already at/above threshold, None if no crossing is
    expected within `max_extrapolation_days`.  Beyond the horizon the last
    7-day slope of the path is extrapolated linearly (flagged).
    """
    if level is None or not math.isfinite(level):
        return None, False
    if level >= threshold:
        return 0.0, False
    p = np.asarray(path, dtype=float)
    full = np.concatenate([[level], p])
    above = np.flatnonzero(full >= threshold)
    if len(above):
        k = int(above[0])
        y0, y1 = full[k - 1], full[k]
        frac = (threshold - y0) / (y1 - y0) if y1 > y0 else 1.0
        return round(k - 1 + float(np.clip(frac, 0, 1)), 1), False
    H = len(p)
    tail = p[-8:] if H >= 8 else full
    slope = (tail[-1] - tail[0]) / max(len(tail) - 1, 1)
    if not math.isfinite(slope) or slope <= 1e-3:
        return None, True
    days = H + (threshold - p[-1]) / slope
    if days > max_extrapolation_days:
        return None, True
    return round(float(days), 1), True


def alert_level(central: float | None, earliest: float | None, red_days: float, amber_days: float) -> str:
    """GREEN / AMBER / RED.  The pessimistic (earliest) estimate can raise GREEN to AMBER."""
    def lvl(d):
        if d is None:
            return 0
        if d <= red_days:
            return 2
        if d <= amber_days:
            return 1
        return 0
    c, e = lvl(central), lvl(earliest)
    level = max(c, min(e, 1)) if e > c else c
    return ["GREEN", "AMBER", "RED"][level]
