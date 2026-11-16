"""Figures and failure analysis for the technical report.

Usage (after `rofoul synth` and `rofoul backtest`):
    python scripts/analyze.py --data data/synthetic --reports reports --figs docs/figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from rofoul.config import load_config  # noqa: E402
from rofoul.features import expand_horizons, origin_table  # noqa: E402
from rofoul.models import GBMForecaster  # noqa: E402
from rofoul.pipeline import prepare_daily  # noqa: E402

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 150})
C = {"gbm": "#1f5fa8", "linear_trend": "#d17a00", "persistence": "#8a8a8a", "truth": "#2b2b2b", "thr": "#b8322a"}


def fig_normalization(daily, truth, out, daily_cp=None):
    g = daily[(daily.train_id == "A") & (daily.date >= "2023-01-01") & (daily.date < "2024-01-01")]
    t = truth[(truth.train_id == "A")].set_index("date").reindex(g["date"])
    fig, ax = plt.subplots(2, 1, figsize=(7.2, 3.0), sharex=True, gridspec_kw={"height_ratios": [1, 1.6]})
    a = ax[0]
    a.plot(g.date, g.permeate_flow, color="#999", lw=0.8, label="raw permeate flow (m³/h)")
    a2 = a.twinx()
    a2.plot(g.date, g.feed_pressure, color="#c55", lw=0.8, label="feed pressure (bar)")
    a2.set_ylabel("bar", color="#c55"); a.set_ylabel("m³/h")
    a.set_title("Raw signals (Train A, 2023): flow is held constant, pressure compensates fouling + temperature",
                loc="left", fontsize=8)
    b = ax[1]
    b.plot(g.date, g.npf_decline_pct, color=C["gbm"], lw=1, label="NPF decline vs post-clean baseline (measured)")
    if daily_cp is not None:
        gc = daily_cp[(daily_cp.train_id == "A")].set_index("date").reindex(g["date"])
        b.plot(g.date, gc["npf_decline_pct"].to_numpy(), color="#5aa05a", lw=0.8,
               label="same, with CP correction (optional)")
    b.plot(g.date, t["true_npf_decline_pct"].to_numpy(), color=C["truth"], lw=0.8, ls="--",
           label="true permeability loss (simulator)")
    b.axhline(10, color=C["thr"], lw=0.8, ls=":", label="cleaning threshold 10 %")
    for d in g.date[g.cycle_start.fillna(False).astype(bool)]:
        b.axvline(d, color="#bbb", lw=0.5)
    b.set_ylabel("%"); b.legend(loc="upper left", fontsize=6.2, ncol=2, frameon=False)
    b.set_title("After normalization: saw-tooth fouling cycles; grey lines = detected cleanings", loc="left", fontsize=8)
    b.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    fig.tight_layout(); fig.savefig(out / "fig1_normalization.png"); plt.close(fig)


def fig_forecast_example(daily, cfg, model, out):
    g = daily[daily.train_id == "C"].sort_values("date").reset_index(drop=True)
    ot = origin_table(g, cfg)
    # pick a test-period origin ~ 2/3 through a cycle that reached the threshold
    cand = ot[(ot.date >= "2024-07-01") & (ot.dsc >= 25) & (ot.npf_lvl.between(5.5, 7.5))]
    o = cand.iloc[len(cand) // 2]
    X = expand_horizons(ot.loc[[o.name]], None, 30, with_targets=False)
    q = model.predict(X, "npf")
    lvl = o.npf_lvl
    dates = o.date + pd.to_timedelta(np.arange(1, 31), unit="D")
    hist = g[(g.date > o.date - pd.Timedelta(days=60)) & (g.date <= o.date) & (g.cycle_id == o.cycle_id)]
    fut = g[(g.date > o.date) & (g.date <= o.date + pd.Timedelta(days=30)) & (g.cycle_id == o.cycle_id)]
    fig, a = plt.subplots(figsize=(7.2, 2.5))
    a.plot(hist.date, hist.npf_decline_pct, color=C["truth"], lw=1, label="observed (history)")
    a.plot(fut.date, fut.npf_decline_pct, color=C["truth"], lw=1, ls="--", alpha=0.6, label="observed (future, unseen)")
    a.fill_between(dates, lvl + q[0.1], lvl + q[0.9], color=C["gbm"], alpha=0.15, label="GBM p10-p90 (conformal)")
    a.plot(dates, lvl + q[0.5], color=C["gbm"], lw=1.2, label="GBM median")
    a.plot(dates, lvl + q["decision"], color=C["gbm"], lw=1, ls=":", label=f"decision path (q={model.decision_quantile})")
    a.plot(dates, lvl + X["lin_inc_npf"], color=C["linear_trend"], lw=1, label="linear-trend baseline")
    a.axhline(10, color=C["thr"], lw=0.8, ls=":")
    a.axvline(o.date, color="#999", lw=0.6)
    a.set_ylabel("NPF decline %")
    a.set_title(f"Forecast from {o.date.date()} (Train C, test period, day {o.dsc:.0f} of cycle)", loc="left", fontsize=8)
    a.legend(fontsize=6.3, ncol=3, frameon=False, loc="upper left")
    a.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    fig.tight_layout(); fig.savefig(out / "fig2_forecast_example.png"); plt.close(fig)


def fig_metrics(metrics, days, out):
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.3))
    H = [1, 3, 7, 14, 21, 30]
    for a, key, title in ((ax[0], "npf_decline_pct", "NPF decline: MAE (pp)"), (ax[1], "dp_rise_pct", "nDP rise: MAE (pp)")):
        for m in ("persistence", "linear_trend", "gbm"):
            a.plot(H, [metrics["forecast_mae"][key][m][f"h{h}"] for h in H], marker="o", ms=2.5, color=C[m], label=m)
        a.set_xlabel("horizon (days)"); a.set_title(title, loc="left", fontsize=8)
    ax[0].legend(fontsize=6.5, frameon=False)
    d = days[days.true_days.notna() & (days.true_days > 0) & (days.true_days <= 30)]
    a = ax[2]
    for m in ("linear_trend", "gbm"):
        p = d[f"{m}_days"].replace(np.inf, 60).clip(upper=60)
        err = (p - d.true_days)
        bins = pd.cut(d.true_days, [0, 5, 10, 15, 20, 25, 30])
        mae = err.abs().groupby(bins, observed=False).mean()
        a.plot([5, 10, 15, 20, 25, 30], mae.to_numpy(), marker="o", ms=2.5, color=C[m], label=m)
    a.set_xlabel("true days to threshold"); a.set_title("Days-to-clean: MAE (days)", loc="left", fontsize=8)
    fig.tight_layout(); fig.savefig(out / "fig3_metrics.png"); plt.close(fig)


def failure_analysis(rows, days, truth, daily):
    """Error by operating condition (GBM, h = 14 days, NPF; and days-to-clean)."""
    t = truth.set_index(["train_id", "date"])
    r = rows[(rows.h == 14) & rows.target_npf.notna()].copy()
    r["err"] = (r.gbm_npf - r.target_npf)
    r["err_lin"] = (r.linear_trend_npf - r.target_npf)
    r["err_dp"] = (r.gbm_dp - r.target_dp)

    def window_max(col, h=14):
        s = truth.pivot(index="date", columns="train_id", values=col)
        fwd = s[::-1].rolling(h, min_periods=1).max()[::-1].shift(-1)  # max over (t, t+h]
        return fwd.stack().rename(col).swaplevel().sort_index()

    def flag(df, col, thr):
        m = window_max(col).reindex(pd.MultiIndex.from_arrays([df.train_id, df.date])).to_numpy(dtype=float)
        return np.nan_to_num(m, nan=0.0) > thr

    now_bloom = t["bloom_mult"].reindex(pd.MultiIndex.from_arrays([r.train_id, r.date])).to_numpy() > 1.2
    r["bloom_starts_in_window"] = flag(r, "bloom_mult", 1.2) & ~now_bloom
    r["bloom_ongoing"] = now_bloom
    r["pretreatment_upset"] = flag(r, "upset_mult", 1.2)
    r["scaling_event"] = flag(r, "scaling", 0.5)
    r["pump_limited"] = flag(r, "pump_limited_h", 0)
    drift = (r.train_id == "D") & r.date.between("2024-04-10", "2024-06-09")
    r["transmitter_drift"] = drift
    r["early_cycle_dsc_3_7"] = r.dsc.between(3, 7)
    r["late_cycle_dsc_gt_40"] = r.dsc > 40
    r["hot_feed"] = t["temp_c"].reindex(pd.MultiIndex.from_arrays([r.train_id, r.date])).to_numpy() > 31
    segs = ["bloom_starts_in_window", "bloom_ongoing", "pretreatment_upset", "scaling_event", "pump_limited",
            "transmitter_drift", "early_cycle_dsc_3_7", "late_cycle_dsc_gt_40", "hot_feed"]
    base = r[~r[segs[:6]].any(axis=1)]
    out = {"all": _seg_stats(r), "no_event": _seg_stats(base)}
    for s in segs:
        out[s] = _seg_stats(r[r[s]])
    # days-to-clean error split by whether an unforeseeable event happens before the true crossing
    d = days[days.true_days.notna() & (days.true_days > 0) & (days.true_days <= 30)].copy()
    ev = np.zeros(len(d), dtype=bool)
    piv = {c: truth.pivot(index="date", columns="train_id", values=c) for c in ("bloom_mult", "upset_mult", "scaling")}
    for i, (tid, dt, td) in enumerate(zip(d.train_id, d.date, d.true_days)):
        sl = slice(dt + pd.Timedelta(days=1), dt + pd.Timedelta(days=int(td)))
        ev[i] = bool((piv["bloom_mult"].loc[sl, tid] > 1.2).any() or (piv["upset_mult"].loc[sl, tid] > 1.2).any()
                     or (piv["scaling"].loc[sl, tid] > 0.5).any())
    d["event_before_crossing"] = ev
    dd = {}
    for name, m in (("event_before_crossing", ev), ("no_event", ~ev)):
        x = d[m]
        row = {"n": int(len(x))}
        for mdl in ("gbm", "linear_trend"):
            e = x[f"{mdl}_days"].replace(np.inf, np.nan) - x.true_days
            row[f"{mdl}_mae_days"] = round(float(e.abs().mean()), 1)
            row[f"{mdl}_bias_days"] = round(float(e.mean()), 1)
        dd[name] = row
    for tid in sorted(d.train_id.unique()):
        x = d[d.train_id == tid]
        e = x["gbm_days"].replace(np.inf, np.nan) - x.true_days
        el = x["linear_trend_days"].replace(np.inf, np.nan) - x.true_days
        dd[f"train_{tid}"] = {"n": int(len(x)), "gbm_mae_days": round(float(e.abs().mean()), 1),
                              "linear_trend_mae_days": round(float(el.abs().mean()), 1)}
    out["days_to_clean_by_condition"] = dd
    return out


def _seg_stats(df):
    if len(df) == 0:
        return {"n": 0}
    return {"n": int(len(df)), "gbm_mae_npf_h14": round(float(df.err.abs().mean()), 2),
            "gbm_bias_npf_h14": round(float(df.err.mean()), 2),
            "linear_mae_npf_h14": round(float(df.err_lin.abs().mean()), 2),
            "gbm_mae_dp_h14": round(float(df.err_dp.abs().mean()), 2)}


def temperature_artifact(daily, truth):
    m = daily.merge(truth, on=["train_id", "date"])
    m = m[m.npf.notna()]
    r = m.npf / m.true_A_rel
    r = r / r.groupby(m.train_id).transform("median")
    return round(float(r[m.feed_temp > 31].mean() / r[m.feed_temp < 22].mean()), 3)


def normalization_error(daily, truth):
    m = daily.merge(truth, on=["train_id", "date"])
    ok = m.npf_decline_pct.notna()
    e = (m.npf_decline_pct - m.true_npf_decline_pct)[ok]
    e2 = (m.dp_rise_pct - m.true_dp_rise_pct)[m.dp_rise_pct.notna()]
    drift = (m.train_id == "D") & m.date.between("2024-04-10", "2024-06-09")
    e3 = (m.dp_rise_pct - m.true_dp_rise_pct)[drift & m.dp_rise_pct.notna()]
    raw_cv = m.groupby("train_id").permeate_flow.std() / m.groupby("train_id").permeate_flow.mean()
    return {"npf_decline_mae_pp": round(float(e.abs().mean()), 2), "npf_decline_bias_pp": round(float(e.mean()), 2),
            "dp_rise_mae_pp": round(float(e2.abs().mean()), 2),
            "dp_rise_mae_during_transmitter_drift_pp": round(float(e3.abs().mean()), 2),
            "corr_npf_vs_true_permeability": round(float(np.corrcoef(m.npf[ok], m.true_A_rel[ok])[0, 1]), 3),
            "corr_raw_flow_vs_true_permeability": round(float(np.corrcoef(m.permeate_flow[ok], m.true_A_rel[ok])[0, 1]), 3),
            "raw_permeate_flow_cv": round(float(raw_cv.mean()), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synthetic")
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--figs", default="docs/figures")
    ap.add_argument("--model", default="models/v1")
    a = ap.parse_args()
    data, rep, figs = Path(a.data), Path(a.reports), Path(a.figs)
    figs.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    daily, _, _ = prepare_daily(data / "plant_data.csv", cfg)
    truth = pd.read_csv(data / "ground_truth_daily.csv", parse_dates=["date"])
    for c in ("scaling", "replacement", "cip"):
        truth[c] = truth[c].astype(str).str.lower().isin(["true", "1"]).astype(float)
    rows = pd.read_csv(rep / "backtest_rows.csv.gz", parse_dates=["date"])
    days = pd.read_csv(rep / "backtest_days.csv", parse_dates=["date"])
    metrics = json.loads((rep / "backtest_metrics.json").read_text())
    # the example forecast uses a model trained only on data before the test fold
    from rofoul.evaluate import train_model
    from rofoul.features import build_training_table
    fold_model = train_model(build_training_table(daily, cfg), cfg, before=pd.Timestamp("2024-07-01"))
    cp_cfg = load_config(overrides={"normalization": {"concentration_polarization": {
        "enabled": True, "membrane_area_m2": 10000, "k_lmh": 150}}})
    daily_cp, _, _ = prepare_daily(data / "plant_data.csv", cp_cfg)
    fig_normalization(daily, truth, figs, daily_cp)
    fig_forecast_example(daily, cfg, fold_model, figs)
    fig_metrics(metrics, days, figs)
    fa = failure_analysis(rows, days, truth, daily)
    ne = normalization_error(daily, truth)
    ne["npf_ratio_hot_over_cold_default"] = temperature_artifact(daily, truth)
    ne_cp = normalization_error(daily_cp, truth)
    ne["with_cp_correction_k150"] = {"npf_decline_mae_pp": ne_cp["npf_decline_mae_pp"],
                                     "npf_decline_bias_pp": ne_cp["npf_decline_bias_pp"],
                                     "npf_ratio_hot_over_cold": temperature_artifact(daily_cp, truth)}
    # cleaning detection without the CIP log
    (rep / "failure_analysis.json").write_text(json.dumps({"forecast_error_by_condition": fa,
                                                           "normalization_vs_ground_truth": ne}, indent=2))
    print(json.dumps({"failure": fa, "normalization": ne}, indent=1))


if __name__ == "__main__":
    main()
