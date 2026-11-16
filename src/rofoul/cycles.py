"""Cleaning-cycle segmentation and fouling indicators.

A cleaning-in-place (CIP) resets reversible fouling, so fouling is measured
relative to the post-clean baseline of the current cycle:

    npf_decline_pct = 100 * (1 - NPF / NPF_baseline)
    dp_rise_pct     = 100 * (nDP / nDP_baseline - 1)

Baselines are causal: on day t they only use cycle data up to day t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _detect_jumps(npf: np.ndarray, cfg: dict) -> np.ndarray:
    """Return boolean array: True on days that start a new cycle (auto-detected CIP)."""
    ccfg = cfg["cycles"]
    jump = 1.0 + ccfg["detect_jump_pct"] / 100.0
    confirm = int(ccfg["confirm_days"])
    min_len = int(ccfg["min_cycle_days"])
    n = len(npf)
    starts = np.zeros(n, dtype=bool)
    valid_idx = np.flatnonzero(np.isfinite(npf))
    last_start_pos = -10**9
    for k, i in enumerate(valid_idx):
        if k < 3 or k - last_start_pos < min_len:
            continue
        prev = npf[valid_idx[k - 3:k]]
        ref = np.median(prev)
        if npf[i] < jump * ref:
            continue
        nxt = valid_idx[k + 1:k + 1 + confirm]
        if len(nxt) < confirm:  # cannot confirm yet (end of data)
            continue
        if np.all(npf[nxt] >= jump * ref * 0.995):
            starts[i] = True
            last_start_pos = k
    return starts


def unconfirmed_jump_at_end(npf: np.ndarray, cfg: dict) -> bool:
    """True if the last valid day looks like a cleaning that cannot be confirmed yet."""
    v = npf[np.isfinite(npf)]
    if len(v) < 4:
        return False
    return bool(v[-1] >= (1 + cfg["cycles"]["detect_jump_pct"] / 100.0) * np.median(v[-4:-1]))


def add_cycles(daily: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    parts = []
    for tid, g in daily.groupby("train_id", sort=True):
        parts.append(_add_cycles_one(g.sort_values("date").reset_index(drop=True), cfg))
    return pd.concat(parts, ignore_index=True)


def _add_cycles_one(g: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    npf = g["npf"].to_numpy(dtype=float)
    n = len(g)
    valid = np.isfinite(npf)
    if "cip" in g and np.nansum(g["cip"].to_numpy(dtype=float)) > 0:
        cip = g["cip"].fillna(0).to_numpy() > 0
        starts = np.zeros(n, dtype=bool)
        pending = False
        for i in range(n):
            if cip[i]:
                pending = True
                continue
            if pending and valid[i]:
                starts[i] = True
                pending = False
        # union with auto-detected jumps (catches unlogged cleanings / membrane replacements)
        auto = _detect_jumps(npf, cfg)
        logged = np.flatnonzero(starts)
        for i in np.flatnonzero(auto):
            if len(logged) == 0 or np.min(np.abs(logged - i)) > 3:
                starts[i] = True
        source = "cip_column+auto"
    else:
        starts = _detect_jumps(npf, cfg)
        source = "auto_detected"
    first_valid = np.flatnonzero(valid)
    if len(first_valid):
        starts[first_valid[0]] = True  # data start opens cycle 0 (true start unknown)
    cycle_id = np.cumsum(starts) - 1
    cycle_id[: first_valid[0] if len(first_valid) else n] = -1

    g = g.copy()
    g["cycle_start"] = starts
    g["cycle_id"] = cycle_id
    g["cycle_source"] = source
    start_dates = g["date"].where(starts).ffill()
    g["days_since_clean"] = (g["date"] - start_dates).dt.days
    if "cip" in g:
        # the cleaning day itself mixes pre/post-clean data -> do not use it
        for c in ("npf", "ndp_norm", "nsp"):
            g.loc[g["cip"].fillna(0) > 0, c] = np.nan

    nb = int(cfg["cycles"]["baseline_days"])
    # partial days (e.g. the day a cleaning ended) mix conditions -> not used for the baseline
    if "coverage_hours" in g:
        cov = g["coverage_hours"].to_numpy(dtype=float)
        full_day = cov >= 0.75 * np.nanmedian(cov[cov > 0]) if np.any(cov > 0) else np.ones(n, bool)
    else:
        full_day = np.ones(n, dtype=bool)
    for col, out_base in (("npf", "npf_baseline"), ("ndp_norm", "dp_baseline"), ("nsp", "nsp_baseline")):
        base = np.full(n, np.nan)
        for cid in np.unique(cycle_id[cycle_id >= 0]):
            idx = np.flatnonzero(cycle_id == cid)
            vals = g[col].to_numpy(dtype=float)[idx]
            seen: list[float] = []
            for j, v in zip(idx, vals):
                # causal fall-back: after nb+2 days accept partial days too
                use = full_day[j] or (j - idx[0]) >= nb + 2
                if np.isfinite(v) and use and len(seen) < nb:
                    seen.append(v)
                base[j] = np.median(seen) if seen else np.nan
        g[out_base] = base
    g["npf_decline_pct"] = 100.0 * (1.0 - g["npf"] / g["npf_baseline"])
    g["dp_rise_pct"] = 100.0 * (g["ndp_norm"] / g["dp_baseline"] - 1.0)
    g["nsp_rise_pct"] = 100.0 * (g["nsp"] / g["nsp_baseline"] - 1.0)
    first_npf = g["npf"].dropna()
    ref0 = float(first_npf.iloc[: nb].median()) if len(first_npf) else np.nan
    g["npf_vs_start_pct"] = 100.0 * (1.0 - g["npf"] / ref0)
    return g
