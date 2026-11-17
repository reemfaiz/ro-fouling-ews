"""End-to-end pipeline: CSV -> validated -> normalized -> daily -> cycles -> forecast."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import resolve_model_path
from .cycles import add_cycles, unconfirmed_jump_at_end
from .features import TARGETS, expand_horizons, origin_table
from .forecast import alert_level, days_to_threshold
from .models import GBMForecaster, LinearTrendModel
from .normalize import normalize, to_daily
from .validation import InputValidationError, ValidationReport, load_and_validate

log = logging.getLogger(__name__)

DAILY_OUTPUT_COLS = [
    "train_id", "date", "feed_temp", "feed_tds", "recovery", "permeate_flow", "feed_pressure", "diff_pressure",
    "ndp_bar", "tcf", "npf", "ndp_norm", "nsp", "cycle_id", "days_since_clean", "cycle_start",
    "npf_baseline", "npf_decline_pct", "dp_baseline", "dp_rise_pct", "nsp_rise_pct", "npf_vs_start_pct",
    "coverage_hours",
]
THRESHOLD_KEY = {"npf": "npf_decline_pct", "dp": "dp_rise_pct"}


@dataclass
class PipelineResult:
    daily: pd.DataFrame
    forecast: pd.DataFrame
    summary: list[dict]
    report: ValidationReport
    references: dict
    model_info: dict = field(default_factory=dict)

    def to_dict(self, include_daily: bool = True, include_forecast: bool = True) -> dict:
        def records(df):
            d = df.copy()
            for c in d.columns:
                if pd.api.types.is_datetime64_any_dtype(d[c]):
                    d[c] = d[c].dt.strftime("%Y-%m-%d")
            d = d.astype(object).where(pd.notna(d), None)
            return d.to_dict(orient="records")
        out = {"summary": self.summary, "validation": self.report.to_dict(), "model": self.model_info,
               "references": self.references}
        if include_forecast:
            out["forecast"] = records(self.forecast)
        if include_daily:
            out["normalized_daily"] = records(self.daily)
        return out


def prepare_daily(source, cfg: dict) -> tuple[pd.DataFrame, ValidationReport, dict]:
    df, report = load_and_validate(source, cfg)
    norm, refs = normalize(df, cfg)
    daily = add_cycles(to_daily(norm, cfg), cfg)
    return daily, report, refs


def load_model(cfg: dict, report: ValidationReport | None = None):
    path = resolve_model_path(cfg)
    try:
        model = GBMForecaster.load(path)
    except FileNotFoundError as exc:
        if report is not None:
            report.warn("model_missing", f"{exc} Falling back to the linear-trend baseline.")
        return None
    except Exception as exc:  # e.g. pickle written by an incompatible scikit-learn version
        msg = (f"Model at {path} could not be loaded ({type(exc).__name__}: {exc}). Falling back to the "
               "linear-trend baseline. Rebuild it on this machine with: rofoul synth --out data/synthetic && "
               f"rofoul train data/synthetic/plant_data.csv --out {cfg['model']['path']}")
        log.warning(msg)
        if report is not None:
            report.warn("model_unloadable", msg)
        return None
    import sklearn
    trained = model.metadata.get("sklearn_version")
    if trained and trained != sklearn.__version__ and report is not None:
        report.warn("model_version_mismatch",
                    f"Model was trained with scikit-learn {trained}, running {sklearn.__version__}. "
                    "Predictions should still work; retrain (`rofoul train`) if they look wrong.")
    return model


def _ood_warnings(origin: pd.Series, model: GBMForecaster) -> list[str]:
    out = []
    labels = {"temp_7": "7-day feed temperature (degC)", "tds_7": "feed TDS (g/L)", "rec_7": "recovery",
              "flux_ratio": "flux vs history", "npf_s14": "NPF-decline slope (%/day)",
              "dp_s14": "DP-rise slope (%/day)", "npf_lvl": "NPF decline (%)", "dp_lvl": "DP rise (%)",
              "dsc": "days since cleaning"}
    for f, lab in labels.items():
        if f not in model.feature_ranges or f not in origin or not np.isfinite(origin[f]):
            continue
        lo, hi = model.feature_ranges[f]
        span = hi - lo
        if origin[f] < lo - 0.1 * span or origin[f] > hi + 0.1 * span:
            out.append(f"{lab} = {origin[f]:.3g} is outside the training range [{lo:.3g}, {hi:.3g}]; "
                       "forecast is an extrapolation - treat with caution or retrain on this plant.")
    return out


def forecast_train(g: pd.DataFrame, cfg: dict, model: GBMForecaster | None, file_end: pd.Timestamp
                   ) -> tuple[pd.DataFrame, dict]:
    tid = str(g["train_id"].iloc[0])
    fc = cfg["forecast"]
    th = cfg["thresholds"]
    H = int(fc["horizon_days"])
    warnings: list[str] = []
    g = g.sort_values("date").reset_index(drop=True)
    valid = g["npf"].notna()
    n_valid = int(valid.sum())
    summary: dict = {"train_id": tid, "status": "ok", "warnings": warnings}
    if n_valid < cfg["validation"]["min_history_days"]:
        summary.update(status="insufficient_history",
                       message=f"Only {n_valid} valid day(s); {cfg['validation']['min_history_days']} needed.")
        return pd.DataFrame(), summary
    last = int(np.flatnonzero(valid.to_numpy())[-1])
    last_date = g.loc[last, "date"]
    ot = origin_table(g.iloc[: last + 1], cfg)
    origin = ot.iloc[-1]
    X = expand_horizons(ot.iloc[[-1]], None, H, with_targets=False)

    if (file_end.normalize() - last_date).days > cfg["validation"]["stale_data_days"]:
        warnings.append(f"Last valid day {last_date.date()} is {(file_end.normalize() - last_date).days} days "
                        "before the end of the file; forecast starts from stale data.")
    dsc = float(origin["dsc"])
    if dsc < fc["min_cycle_days_for_forecast"]:
        warnings.append(f"Only {dsc:.0f} day(s) since the last cleaning: post-clean baseline still forming; "
                        "low confidence.")
    if origin["cycle0"] == 1.0:
        warnings.append("No cleaning seen in the file: fouling is measured relative to the first days of the file, "
                        "not to a post-clean baseline. Provide a `cip` column or start the file after a cleaning.")
    if unconfirmed_jump_at_end(g["npf"].iloc[: last + 1].to_numpy(dtype=float), cfg):
        warnings.append("NPF jumped on the last day - possibly a cleaning that cannot be confirmed yet.")
    if model is not None:
        warnings += _ood_warnings(origin, model)

    dates = last_date + pd.to_timedelta(np.arange(1, H + 1), unit="D")
    fdf = pd.DataFrame({"train_id": tid, "origin_date": last_date, "date": dates, "horizon_days": np.arange(1, H + 1)})
    per_ind = {}
    base_model = LinearTrendModel()
    for p, col in TARGETS.items():
        level = float(origin[f"{p}_lvl"])
        thr = float(th[THRESHOLD_KEY[p]])
        base_path = level + base_model.predict(X, p)[0.5]
        fdf[f"{col}_baseline"] = base_path
        if model is not None and np.isfinite(level):
            q = model.predict(X, p)
            qs = sorted(k for k in q if not isinstance(k, str))
            for qq in qs:
                fdf[f"{col}_p{int(round(qq * 100))}"] = level + q[qq]
            fdf[f"{col}_decision"] = level + q["decision"]
            mid_path = level + q["decision"]          # calibrated decision path -> estimate
            hi_path = level + q[qs[-1]]               # pessimistic (widened p90) -> earliest
            lo_path = level + q[0.5]                  # median path -> latest
        else:
            lo_path = mid_path = hi_path = base_path
            fdf[f"{col}_p50"] = base_path
        mx = fc["max_extrapolation_days"]
        central, beyond = days_to_threshold(level, mid_path, thr, mx)
        earliest, _ = days_to_threshold(level, hi_path, thr, mx)
        latest, _ = days_to_threshold(level, lo_path, thr, mx)
        base_days, _ = days_to_threshold(level, base_path, thr, mx)
        per_ind[p] = dict(level=level, threshold=thr, central=central, earliest=earliest, latest=latest,
                          beyond_horizon=beyond, baseline=base_days)

    def _min(key):
        vals = [(v[key], p) for p, v in per_ind.items() if v[key] is not None]
        return min(vals) if vals else (None, None)

    central, limiting = _min("central")
    earliest, _ = _min("earliest")
    latest, _ = _min("latest")
    base_days, _ = _min("baseline")
    alert = alert_level(central, earliest, cfg["alerts"]["red_days"], cfg["alerts"]["amber_days"])
    beyond = bool(per_ind[limiting]["beyond_horizon"]) if limiting else True
    if central == 0.0:
        msg = f"Cleaning threshold already exceeded ({THRESHOLD_KEY[limiting]}). Clean now."
    elif central is None:
        msg = f"No threshold crossing expected within {fc['max_extrapolation_days']} days."
    else:
        msg = (f"Cleaning expected in ~{central:.0f} days (limiting: {THRESHOLD_KEY[limiting]}"
               f"{', extrapolated beyond forecast horizon' if beyond else ''}).")
    cur = g.loc[last]
    summary.update(
        last_data_date=str(last_date.date()),
        days_since_clean=dsc,
        cycle_source=str(g["cycle_source"].iloc[0]),
        current={
            "npf_m3h": _f(cur["npf"]), "npf_decline_pct": _f(origin["npf_lvl"]),
            "ndp_norm_bar": _f(cur["ndp_norm"]), "dp_rise_pct": _f(origin["dp_lvl"]),
            "nsp_rise_pct": _f(cur.get("nsp_rise_pct")), "feed_temp_c": _f(cur["feed_temp"]),
        },
        thresholds={"npf_decline_pct": th["npf_decline_pct"], "dp_rise_pct": th["dp_rise_pct"]},
        days_to_clean={
            "estimate": central, "earliest_p90_fouling": earliest, "latest_p10_fouling": latest,
            "limiting_indicator": THRESHOLD_KEY.get(limiting) if limiting else None,
            "beyond_forecast_horizon": beyond,
            "estimated_date": str((last_date + pd.Timedelta(days=central)).date()) if central is not None else None,
            "per_indicator": {THRESHOLD_KEY[p]: v for p, v in per_ind.items()},
        },
        baseline_linear_days_to_clean=base_days,
        alert=alert,
        message=msg,
        model_used="gbm" if model is not None else "linear_trend (fallback)",
    )
    return fdf, summary


def _f(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, 3) if np.isfinite(x) else None


def run(source, cfg: dict, model: GBMForecaster | None | str = "auto") -> PipelineResult:
    daily, report, refs = prepare_daily(source, cfg)
    if isinstance(model, str) and model == "auto":
        model = load_model(cfg, report)
    file_end = daily["date"].max()
    fparts, summaries = [], []
    for tid, g in daily.groupby("train_id", sort=True):
        fdf, s = forecast_train(g, cfg, model, file_end)
        summaries.append(s)
        if len(fdf):
            fparts.append(fdf)
    if not fparts:
        report.error("no_forecastable_train",
                     "No train has enough history to forecast "
                     f"(need >= {cfg['validation']['min_history_days']} valid days).")
        raise InputValidationError(report)
    info = {}
    if model is not None:
        info = {k: model.metadata.get(k) for k in ("version", "created_utc", "path", "sha256", "training_data")}
    out_daily = daily[[c for c in DAILY_OUTPUT_COLS if c in daily.columns]].copy()
    return PipelineResult(out_daily, pd.concat(fparts, ignore_index=True), summaries, report, refs, info)
