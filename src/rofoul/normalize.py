"""Normalization of RO operating data (ASTM D4516-style).

Raw permeate flow, salt passage and pressure drop move with temperature,
pressure and salinity even when the membrane does not change.  Normalization
re-expresses every reading at fixed reference conditions so that what is left
is membrane condition (fouling, scaling, damage).

    TCF      = exp(U (1/298.15 - 1/(273.15+T)))      (FilmTec; U=2640 T>=25, 3020 T<25)
             = 1.03**(T-25)                           (ASTM simplification)
    C_fc     = C_f * ln(1/(1-R)) / R                  (log-mean feed-concentrate TDS)
    pi(C,T)  = k * C * (T+273.15)/298.15              [bar]
    NDP      = P_f - DP/2 - P_p - (pi_fc*CP - pi_p)       (CP = exp(J/k), optional, off by default)
    NPF      = Q_p * (NDP_ref / NDP) * (TCF_ref / TCF)
    NSP      = SP * (Q_p/Q_p,ref) * (TCF_ref/TCF) * (C_fc,ref/C_fc) * (C_f/C_f,ref)
    nDP      = DP * (Q_avg,ref / Q_avg)**n * (mu(25C)/mu(T))**m
               (Q_avg = (Q_f + Q_c)/2; m = 0.3 from the Schock & Miquel (1987) spacer
               friction factor f ~ Re^-0.3; n = 1.5 vendor default)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

T_REF_K = 298.15


def tcf(temp_c, method: str = "filmtec"):
    """Temperature correction factor relative to 25 degC (TCF(25) = 1)."""
    t = np.asarray(temp_c, dtype=float)
    if method == "astm":
        return 1.03 ** (t - 25.0)
    if method != "filmtec":
        raise ValueError(f"Unknown tcf_method '{method}' (use 'filmtec' or 'astm')")
    inv = 1.0 / T_REF_K - 1.0 / (273.15 + t)
    return np.where(t >= 25.0, np.exp(2640.0 * inv), np.exp(3020.0 * inv))


def water_viscosity(temp_c):
    """Dynamic viscosity of water [Pa s] (Vogel equation)."""
    t = np.asarray(temp_c, dtype=float)
    return 2.414e-5 * 10 ** (247.8 / (t + 133.15))


def feed_brine_concentration(feed_tds, recovery, method: str = "log"):
    c = np.asarray(feed_tds, dtype=float)
    r = np.clip(np.asarray(recovery, dtype=float), 1e-6, 0.99)
    if method == "log":
        return c * np.log(1.0 / (1.0 - r)) / r
    if method == "arithmetic":
        return c * (1.0 + 1.0 / (1.0 - r)) / 2.0
    raise ValueError(f"Unknown feed_brine_average '{method}'")


def osmotic_pressure(tds_mgl, temp_c, coeff: float):
    return coeff * np.asarray(tds_mgl, dtype=float) * (273.15 + np.asarray(temp_c, dtype=float)) / T_REF_K


def operating_terms(df: pd.DataFrame, ncfg: dict) -> pd.DataFrame:
    """Per-row physical terms (no reference needed)."""
    out = pd.DataFrame(index=df.index)
    rec = df["permeate_flow"] / df["feed_flow"]
    c_fc = feed_brine_concentration(df["feed_tds"], rec, ncfg["feed_brine_average"])
    if "permeate_tds" in df and df["permeate_tds"].notna().any():
        c_p = df["permeate_tds"].to_numpy(dtype=float)
        has_sp = True
    else:
        c_p = df["feed_tds"].to_numpy(dtype=float) * (1 - ncfg["assumed_rejection_if_no_permeate_tds"])
        has_sp = False
    k = ncfg["osmotic_coeff_bar_per_mgl"]
    pi_fc = osmotic_pressure(c_fc, df["feed_temp"], k)
    cp_cfg = ncfg.get("concentration_polarization") or {}
    if cp_cfg.get("enabled"):
        if not cp_cfg.get("membrane_area_m2"):
            raise ValueError("normalization.concentration_polarization.enabled requires membrane_area_m2")
        flux_lmh = df["permeate_flow"] * 1000.0 / float(cp_cfg["membrane_area_m2"])
        pi_fc = pi_fc * np.exp(flux_lmh / float(cp_cfg.get("k_lmh", 150.0)))
    pi_p = osmotic_pressure(c_p, df["feed_temp"], k)
    out["recovery"] = rec
    out["c_fc"] = c_fc
    out["pi_fc"] = pi_fc
    out["ndp"] = df["feed_pressure"] - df["diff_pressure"] / 2.0 - df["permeate_pressure"] - (pi_fc - pi_p)
    out["tcf"] = tcf(df["feed_temp"], ncfg["tcf_method"])
    out["q_avg"] = (df["feed_flow"] + df["concentrate_flow"]) / 2.0
    out["salt_passage"] = (c_p / df["feed_tds"]) if has_sp else np.nan
    return out


def compute_reference(df: pd.DataFrame, terms: pd.DataFrame, ncfg: dict) -> dict:
    """Reference ("standard") conditions for one train."""
    rcfg = ncfg["reference"]
    ref: dict = {"temp_c": float(rcfg.get("temp_c", 25.0))}
    ref["tcf"] = float(tcf(ref["temp_c"], ncfg["tcf_method"]))
    if rcfg.get("mode", "auto") == "fixed":
        missing = [k for k in ("ndp_bar", "permeate_flow_m3h", "avg_feed_brine_flow_m3h", "feed_tds_mgl", "recovery")
                   if rcfg.get(k) is None]
        if missing:
            raise ValueError(f"normalization.reference.mode=fixed requires: {', '.join(missing)}")
        ref.update(ndp=float(rcfg["ndp_bar"]), q_p=float(rcfg["permeate_flow_m3h"]),
                   q_avg=float(rcfg["avg_feed_brine_flow_m3h"]), c_f=float(rcfg["feed_tds_mgl"]))
        ref["c_fc"] = float(feed_brine_concentration(ref["c_f"], float(rcfg["recovery"]), ncfg["feed_brine_average"]))
        return ref
    days = int(rcfg.get("auto_days", 7))
    t0 = df["timestamp"].min()
    first = df["timestamp"] < t0 + pd.Timedelta(days=days)
    sel = first & terms["ndp"].notna() & (terms["ndp"] > 0)
    if sel.sum() < 3:
        sel = terms["ndp"].notna() & (terms["ndp"] > 0)
    ref.update(
        ndp=float(terms.loc[sel, "ndp"].median()),
        q_p=float(df.loc[sel, "permeate_flow"].median()),
        q_avg=float(terms.loc[sel, "q_avg"].median()),
        c_f=float(df.loc[sel, "feed_tds"].median()),
        c_fc=float(terms.loc[sel, "c_fc"].median()),
    )
    return ref


def normalize(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Add normalized columns to a canonical (validated) frame.

    Returns (frame, references_by_train).
    """
    ncfg = cfg["normalization"]
    out_parts, refs = [], {}
    for tid, g in df.groupby("train_id", sort=True):
        terms = operating_terms(g, ncfg)
        bad = terms["ndp"] <= 0.1
        ref = compute_reference(g, terms.mask(bad), ncfg)
        refs[str(tid)] = ref
        o = g.copy()
        o["recovery"] = terms["recovery"]
        o["ndp_bar"] = terms["ndp"].where(~bad)
        o["tcf"] = terms["tcf"]
        o["npf"] = g["permeate_flow"] * (ref["ndp"] / o["ndp_bar"]) * (ref["tcf"] / terms["tcf"])
        visc = (water_viscosity(25.0) / water_viscosity(g["feed_temp"])) ** ncfg.get("dp_viscosity_exponent", 0.0)
        o["ndp_norm"] = g["diff_pressure"] * (ref["q_avg"] / terms["q_avg"]) ** ncfg["dp_flow_exponent"] * visc
        o["nsp"] = (terms["salt_passage"] * (g["permeate_flow"] / ref["q_p"]) * (ref["tcf"] / terms["tcf"])
                    * (ref["c_fc"] / terms["c_fc"]) * (g["feed_tds"] / ref["c_f"])) * 100.0
        out_parts.append(o)
    return pd.concat(out_parts).sort_values(["train_id", "timestamp"]), refs


def to_daily(norm: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Aggregate normalized data to daily medians on a continuous calendar per train.

    Days with less than `min_coverage_hours_per_day` of data are set to NaN.
    """
    min_cov = cfg["validation"]["min_coverage_hours_per_day"]
    cols = ["npf", "ndp_norm", "nsp", "feed_temp", "feed_tds", "recovery", "permeate_flow",
            "feed_pressure", "diff_pressure", "ndp_bar", "tcf"]
    parts = []
    for tid, g in norm.groupby("train_id", sort=True):
        g = g.set_index("timestamp").sort_index()
        dt_h = g.index.to_series().diff().dt.total_seconds().div(3600)
        step = float(np.nanmedian(dt_h)) if len(g) > 1 else 24.0
        step = min(max(step, 1 / 60), 24.0)
        day = g.index.normalize()
        agg = g[cols].groupby(day).median()
        counts = g["npf"].notna().groupby(day).sum()
        cov_h = counts * step
        agg.loc[cov_h.reindex(agg.index) < min_cov, cols] = np.nan
        if "cip" in g:
            agg["cip"] = g["cip"].groupby(day).max()
        full = pd.date_range(agg.index.min(), agg.index.max(), freq="D")
        agg = agg.reindex(full)
        agg.index.name = "date"
        agg["coverage_hours"] = cov_h.reindex(full).fillna(0.0)
        agg["train_id"] = str(tid)
        parts.append(agg.reset_index())
    return pd.concat(parts, ignore_index=True)
