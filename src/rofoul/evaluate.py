"""Training and leakage-free temporal back-testing.

Back-test design (rolling origin, expanding window):
  * for each fold [start, end) the model is trained ONLY on samples whose
    target date (origin + h) is before `start` (purged: no label overlaps the
    test period);
  * it is evaluated on every daily origin inside [start, end);
  * normalization references are computed from the first days of each train,
    cycle baselines and all features are causal (see features.py);
  * optionally one train is held out of training entirely (unseen asset).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import TARGETS, build_training_table, smoothed_targets
from .forecast import days_to_threshold
from .models import GBMForecaster, LinearTrendModel, PersistenceModel

THR_KEY = {"npf": "npf_decline_pct", "dp": "dp_rise_pct"}
REPORT_H = [1, 3, 7, 14, 21, 30]


def default_folds(daily: pd.DataFrame, n_folds: int = 3, test_months: int = 6) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    end = daily["date"].max() + pd.Timedelta(days=1)
    folds = []
    for k in range(n_folds, 0, -1):
        s = end - pd.DateOffset(months=test_months * k)
        e = s + pd.DateOffset(months=test_months)
        folds.append((pd.Timestamp(s).normalize(), pd.Timestamp(e).normalize()))
    return folds


def train_model(table: pd.DataFrame, cfg: dict, before: pd.Timestamp | None = None,
                exclude_trains: tuple = ()) -> GBMForecaster:
    t = table
    m = t["dsc"] >= cfg["training"]["min_days_since_clean"]
    if before is not None:
        m &= t["target_date"] < before
    if exclude_trains:
        m &= ~t["train_id"].isin(exclude_trains)
    params = {k: cfg["training"][k] for k in ("max_iter", "learning_rate", "max_leaf_nodes", "min_samples_leaf",
                                              "l2_regularization", "random_state")}
    params["origin_stride"] = int(cfg["training"].get("origin_stride", 1))
    return GBMForecaster(quantiles=cfg["forecast"]["quantiles"], params=params).fit(
        t[m], thresholds=cfg["thresholds"], calibrate=cfg["training"].get("calibrate", True))


def true_days_to_threshold(daily: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """For every (train, date): observed days until either indicator crosses its threshold
    within the same cycle.  NaN + censored=True if the cycle ended (cleaning) first."""
    out = []
    thr = {p: cfg["thresholds"][THR_KEY[p]] for p in TARGETS}
    for tid, g in daily.groupby("train_id", sort=True):
        g = g.sort_values("date").reset_index(drop=True)
        ys = smoothed_targets(g)
        cyc = g["cycle_id"].to_numpy()
        n = len(g)
        cross = np.zeros(n, dtype=bool)
        for p in TARGETS:
            cross |= np.nan_to_num(ys[p], nan=-1e9) >= thr[p]
        days = np.full(n, np.nan)
        censored = np.zeros(n, dtype=bool)
        nxt = None
        for i in range(n - 1, -1, -1):  # backward scan: next crossing in the same cycle
            if i + 1 < n and cyc[i + 1] != cyc[i]:
                nxt = None
            if cross[i]:
                nxt = i
            if nxt is not None:
                days[i] = nxt - i
            else:
                censored[i] = True
        last_cycle = cyc == cyc[-1]
        out.append(pd.DataFrame({"train_id": tid, "date": g["date"], "true_days": days,
                                 "censored": censored & ~last_cycle, "open_end": censored & last_cycle}))
    return pd.concat(out, ignore_index=True)


def _paths_days(test: pd.DataFrame, preds: dict, cfg: dict, H: int) -> pd.DataFrame:
    """Per-origin days-to-clean from per-row predicted increments (central + band)."""
    rows = []
    thr = {p: cfg["thresholds"][THR_KEY[p]] for p in TARGETS}
    mx = cfg["forecast"]["max_extrapolation_days"]
    key = ["train_id", "date"]
    grp = test.groupby(key, sort=False).indices
    for (tid, date), idx in grp.items():
        idx = np.asarray(idx)
        order = np.argsort(test["h"].to_numpy()[idx])
        idx = idx[order]
        if len(idx) != H:
            continue
        rec = {"train_id": tid, "date": date, "dsc": float(test["dsc"].to_numpy()[idx[0]])}
        for model_name, pm in preds.items():
            best = {}
            for qn in ("med", "mid", "hi"):
                ds = []
                for p in TARGETS:
                    lvl = float(test[f"{p}_lvl"].to_numpy()[idx[0]])
                    inc = pm[p][qn][idx]
                    d, _ = days_to_threshold(lvl, lvl + inc, thr[p], mx)
                    ds.append(np.inf if d is None else d)
                best[qn] = min(ds)
            rec[f"{model_name}_days"] = best["mid"]
            rec[f"{model_name}_days_earliest"] = best["hi"]
            rec[f"{model_name}_days_latest"] = best["med"]
        rows.append(rec)
    return pd.DataFrame(rows)


def backtest(daily: pd.DataFrame, cfg: dict, folds=None, holdout_train: str | None = None,
             table: pd.DataFrame | None = None) -> dict:
    H = int(cfg["forecast"]["horizon_days"])
    table = build_training_table(daily, cfg) if table is None else table
    folds = folds or default_folds(daily)
    truth_days = true_days_to_threshold(daily, cfg)
    min_dsc = cfg["training"]["min_days_since_clean"]
    all_rows, all_days = [], []
    for k, (s, e) in enumerate(folds):
        excl = (holdout_train,) if holdout_train else ()
        model = train_model(table, cfg, before=s, exclude_trains=excl)
        test = table[(table["date"] >= s) & (table["date"] < e) & (table["dsc"] >= min_dsc)]
        if holdout_train:
            test = test[test["train_id"] == holdout_train]
        test = test.reset_index(drop=True)
        models = {"persistence": PersistenceModel(), "linear_trend": LinearTrendModel(), "gbm": model}
        preds = {}
        for name, mdl in models.items():
            preds[name] = {}
            for p in TARGETS:
                q = mdl.predict(test, p)
                qs = sorted(k for k in q if not isinstance(k, str))
                # mid = decision path (days-to-clean), med = median path (forecast accuracy)
                preds[name][p] = {"lo": q[qs[0]], "med": q[0.5], "mid": q["decision"], "hi": q[qs[-1]]}
        r = test[["train_id", "date", "h", "dsc", "cycle_id"] + [f"{p}_lvl" for p in TARGETS]
                 + [f"target_{p}" for p in TARGETS]].copy()
        r["fold"] = k
        for name in models:
            for p in TARGETS:
                r[f"{name}_{p}"] = preds[name][p]["med"]
        for p in TARGETS:
            r[f"gbm_{p}_lo"] = preds["gbm"][p]["lo"]
            r[f"gbm_{p}_hi"] = preds["gbm"][p]["hi"]
        all_rows.append(r)
        dd = _paths_days(test, preds, cfg, H)
        dd["fold"] = k
        all_days.append(dd)
    rows = pd.concat(all_rows, ignore_index=True)
    days = pd.concat(all_days, ignore_index=True).merge(truth_days, on=["train_id", "date"], how="left")
    return {"rows": rows, "days": days, "folds": [(str(s.date()), str(e.date())) for s, e in folds],
            "metrics": summarize(rows, days, cfg)}


def summarize(rows: pd.DataFrame, days: pd.DataFrame, cfg: dict) -> dict:
    out: dict = {"forecast_mae": {}, "interval_coverage_p10_p90": {}, "days_to_clean": {}, "alert_red": {}}
    for p in TARGETS:
        ok = rows[f"target_{p}"].notna()
        r = rows[ok]
        tab = {}
        for name in ("persistence", "linear_trend", "gbm"):
            err = (r[f"{name}_{p}"] - r[f"target_{p}"]).abs()
            tab[name] = {f"h{h}": round(float(err[r["h"] == h].mean()), 3) for h in REPORT_H}
            tab[name]["all"] = round(float(err.mean()), 3)
        out["forecast_mae"][THR_KEY[p]] = tab
        inside = (r[f"target_{p}"] >= r[f"gbm_{p}_lo"]) & (r[f"target_{p}"] <= r[f"gbm_{p}_hi"])
        out["interval_coverage_p10_p90"][THR_KEY[p]] = round(float(inside.mean()), 3)
    red = cfg["alerts"]["red_days"]
    d = days[days["true_days"].notna() & (days["true_days"] > 0)]
    for name in ("linear_trend", "gbm"):
        pred = d[f"{name}_days"].replace(np.inf, np.nan)
        sel = d["true_days"] <= cfg["forecast"]["horizon_days"]
        err = (pred - d["true_days"])[sel]
        out["days_to_clean"][name] = {
            "n_origins": int(sel.sum()),
            "mae_days": round(float(err.abs().mean()), 2),
            "median_abs_err_days": round(float(err.abs().median()), 2),
            "bias_days": round(float(err.mean()), 2),
            "no_crossing_predicted_frac": round(float(pred[sel].isna().mean()), 3),
        }
        # alert: crossing predicted within `red` days
        labelled = days[(days["true_days"].notna()) | days["censored"]]
        y_true = labelled["true_days"].fillna(np.inf) <= red
        y_pred = labelled[f"{name}_days"].fillna(np.inf) <= red
        tp = int((y_true & y_pred).sum()); fp = int((~y_true & y_pred).sum()); fn = int((y_true & ~y_pred).sum())
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        out["alert_red"][name] = {"precision": round(prec, 3), "recall": round(rec, 3),
                                  "f1": round(2 * prec * rec / (prec + rec), 3) if prec + rec else float("nan"),
                                  "tp": tp, "fp": fp, "fn": fn}
    if "gbm_days_earliest" in d:
        sel = d["true_days"] <= cfg["forecast"]["horizon_days"]
        lo = d["gbm_days_earliest"].replace(np.inf, 1e9)
        hi = d["gbm_days_latest"].replace(np.inf, 1e9)
        cov = ((d["true_days"] >= lo) & (d["true_days"] <= hi))[sel]
        out["days_to_clean"]["gbm"]["band_coverage"] = round(float(cov.mean()), 3)
    return out
