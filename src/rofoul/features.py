"""Leakage-free feature construction for direct multi-horizon forecasting.

For every forecast origin t (a day) we only use daily values <= t.  Each
origin is expanded into H rows (h = 1..H); the learning target is the change
of the fouling indicator between the current level and day t+h:

    target(t, h) = y_smooth(t+h) - level(t)

where level(t) is the median of the last 3 days (<= t) and y_smooth is a
centred 3-day median (used only as a label, never as an input).  Targets that
cross a cleaning are dropped (the process is reset, so they are censored).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGETS = {"npf": "npf_decline_pct", "dp": "dp_rise_pct"}

ORIGIN_FEATURES = [
    "dsc", "cycle0",
    "npf_lvl", "npf_s7", "npf_s14", "npf_curv", "npf_n14",
    "dp_lvl", "dp_s7", "dp_s14", "dp_curv", "dp_n14",
    "nsp_lvl",
    "temp_7", "temp_s14", "bio_factor", "tds_7", "rec_7", "flux_ratio",
    "prev_rate_npf", "prev_rate_dp", "prev_len", "miss_frac_14",
]


def horizon_features(prefix: str) -> list[str]:
    return ORIGIN_FEATURES + ["h", f"lin_inc_{prefix}"]


def _slope(x: np.ndarray, y: np.ndarray, min_pts: int) -> float:
    m = np.isfinite(y)
    if m.sum() < min_pts:
        return np.nan
    xx, yy = x[m], y[m]
    xx = xx - xx.mean()
    den = (xx * xx).sum()
    return float((xx * (yy - yy.mean())).sum() / den) if den > 0 else np.nan


def origin_table(g: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """One row per day (origin) for a single train with all origin features."""
    g = g.sort_values("date").reset_index(drop=True)
    n = len(g)
    cyc = g["cycle_id"].to_numpy()
    dsc = g["days_since_clean"].to_numpy(dtype=float)
    y = {p: g[col].to_numpy(dtype=float) for p, col in TARGETS.items()}
    nsp = g["nsp_rise_pct"].to_numpy(dtype=float) if "nsp_rise_pct" in g else np.full(n, np.nan)
    temp = g["feed_temp"].to_numpy(dtype=float)
    tds = g["feed_tds"].to_numpy(dtype=float) / 1000.0
    rec = g["recovery"].to_numpy(dtype=float)
    qp = g["permeate_flow"].to_numpy(dtype=float)
    valid = np.isfinite(y["npf"])
    win = int(cfg["forecast"]["trend_window_days"])

    # previous-cycle statistics (known once the cycle is over)
    prev_stats: dict[int, tuple[float, float, float]] = {}
    for cid in np.unique(cyc[cyc >= 0]):
        idx = np.flatnonzero(cyc == cid)
        length = float(len(idx))
        tail = idx[-5:]
        end_npf = np.nanmedian(y["npf"][tail]) if np.isfinite(y["npf"][tail]).any() else np.nan
        end_dp = np.nanmedian(y["dp"][tail]) if np.isfinite(y["dp"][tail]).any() else np.nan
        prev_stats[int(cid)] = (end_npf / max(length, 1.0), end_dp / max(length, 1.0), length)

    rows = []
    qp_hist_med = pd.Series(qp).expanding(min_periods=1).median().to_numpy()
    for i in range(n):
        c = cyc[i]
        if c < 0:
            continue
        lo = i
        while lo > 0 and cyc[lo - 1] == c and i - (lo - 1) < 28:
            lo -= 1
        r = {"row": i, "dsc": dsc[i], "cycle0": float(c == 0)}
        x = np.arange(i - lo + 1, dtype=float)
        for p, arr in y.items():
            seg = arr[lo:i + 1]
            last3 = seg[-3:]
            r[f"{p}_lvl"] = float(np.nanmedian(last3)) if np.isfinite(last3).any() else np.nan
            s7 = _slope(x[-7:], seg[-7:], 4)
            s14 = _slope(x[-win:], seg[-win:], 6)
            r[f"{p}_s7"], r[f"{p}_s14"] = s7, s14
            r[f"{p}_curv"] = s7 - s14 if np.isfinite(s7) and np.isfinite(s14) else np.nan
            r[f"{p}_n14"] = float(np.isfinite(seg[-14:]).sum())
        r["nsp_lvl"] = float(np.nanmedian(nsp[max(lo, i - 2):i + 1])) if np.isfinite(nsp[max(lo, i - 2):i + 1]).any() else np.nan
        tw = temp[max(0, i - 6):i + 1]
        r["temp_7"] = float(np.nanmean(tw)) if np.isfinite(tw).any() else np.nan
        r["temp_s14"] = _slope(np.arange(min(i + 1, 14), dtype=float), temp[max(0, i - 13):i + 1], 5)
        r["bio_factor"] = 1.08 ** (r["temp_7"] - 25.0) if np.isfinite(r["temp_7"]) else np.nan
        for name, arr in (("tds_7", tds), ("rec_7", rec)):
            w = arr[max(0, i - 6):i + 1]
            r[name] = float(np.nanmean(w)) if np.isfinite(w).any() else np.nan
        qw = qp[max(0, i - 6):i + 1]
        r["flux_ratio"] = float(np.nanmean(qw) / qp_hist_med[i]) if np.isfinite(qw).any() and qp_hist_med[i] > 0 else np.nan
        if c - 1 in prev_stats and c - 1 > 0:  # cycle 0 has an unknown start -> its rate is unreliable
            r["prev_rate_npf"], r["prev_rate_dp"], r["prev_len"] = prev_stats[c - 1]
        else:
            r["prev_rate_npf"] = r["prev_rate_dp"] = r["prev_len"] = np.nan
        r["miss_frac_14"] = 1.0 - float(valid[max(0, i - 13):i + 1].mean())
        rows.append(r)
    ot = pd.DataFrame(rows)
    if ot.empty:
        return ot
    ot["date"] = g["date"].to_numpy()[ot["row"].to_numpy()]
    ot["train_id"] = g["train_id"].iloc[0]
    ot["cycle_id"] = cyc[ot["row"].to_numpy()]
    return ot


def smoothed_targets(g: pd.DataFrame) -> dict[str, np.ndarray]:
    """Centred 3-day median of each indicator within a cycle (label only)."""
    g = g.sort_values("date").reset_index(drop=True)
    out = {}
    for p, col in TARGETS.items():
        s = g[col].groupby(g["cycle_id"]).transform(lambda v: v.rolling(3, center=True, min_periods=1).median())
        out[p] = s.to_numpy(dtype=float)
    return out


def expand_horizons(ot: pd.DataFrame, g: pd.DataFrame | None, horizon: int, with_targets: bool) -> pd.DataFrame:
    """Expand origins into origin x horizon rows (+targets when g is given)."""
    if ot.empty:
        return ot
    hs = np.arange(1, horizon + 1)
    rep = ot.loc[ot.index.repeat(horizon)].reset_index(drop=True)
    rep["h"] = np.tile(hs, len(ot)).astype(float)
    for p in TARGETS:
        slope = rep[f"{p}_s14"].fillna(rep[f"{p}_s7"]).fillna(0.0)
        rep[f"lin_inc_{p}"] = slope * rep["h"]
    if with_targets and g is not None:
        g = g.sort_values("date").reset_index(drop=True)
        ys = smoothed_targets(g)
        cyc = g["cycle_id"].to_numpy()
        tgt_row = rep["row"].to_numpy() + rep["h"].to_numpy().astype(int)
        inside = tgt_row < len(g)
        tr = np.where(inside, tgt_row, 0)
        same_cycle = inside & (cyc[tr] == rep["cycle_id"].to_numpy())
        for p in TARGETS:
            yt = np.where(same_cycle, ys[p][tr], np.nan)
            rep[f"y_{p}"] = yt
            rep[f"target_{p}"] = yt - rep[f"{p}_lvl"].to_numpy()
        rep["target_date"] = rep["date"] + pd.to_timedelta(rep["h"], unit="D")
    return rep


def build_training_table(daily: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    H = int(cfg["forecast"]["horizon_days"])
    parts = []
    for tid, g in daily.groupby("train_id", sort=True):
        g = g.sort_values("date").reset_index(drop=True)
        ot = origin_table(g, cfg)
        if ot.empty:
            continue
        parts.append(expand_horizons(ot, g, H, with_targets=True))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
