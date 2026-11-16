"""Forecasting models: two simple baselines and a quantile gradient-boosting model.

All models predict the *increment* of a fouling indicator h days ahead,
relative to the current level, for every row of a horizon table.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor

from . import __version__
from .features import TARGETS, horizon_features


class PersistenceModel:
    """Fouling stays where it is (increment = 0)."""
    name = "persistence"
    quantiles = (0.5,)

    def predict(self, X: pd.DataFrame, prefix: str) -> dict[float, np.ndarray]:
        z = np.zeros(len(X))
        return {0.5: z, "decision": z}


class LinearTrendModel:
    """Industry-style baseline: extrapolate the last 14-day linear trend of the cycle."""
    name = "linear_trend"
    quantiles = (0.5,)

    def predict(self, X: pd.DataFrame, prefix: str) -> dict[float, np.ndarray]:
        v = X[f"lin_inc_{prefix}"].fillna(0.0).to_numpy(dtype=float)
        return {0.5: v, "decision": v}


class GBMForecaster:
    """Direct multi-horizon quantile gradient boosting with physics-motivated features.

    One HistGradientBoostingRegressor per (indicator, quantile).  The model is
    constrained to be non-decreasing in the horizon h and in the linear-trend
    extrapolation (fouling does not reverse within a cleaning cycle).

    Calibration (done on the LAST part of the training period only, never on
    test data):
      * decision quantile q*: the indicator observed day-by-day is noisy, so it
        first crosses the threshold earlier than the median path does
        (first-passage effect).  q* is the quantile path whose threshold
        crossing has ~zero median error on the calibration window.
      * conformal widening of the p10/p90 band per horizon (split-conformal
        quantile regression) so the band reaches its nominal 80 % coverage.
    """
    name = "gbm"
    DECISION_GRID = (0.5, 0.6, 0.7, 0.8, 0.9)

    def __init__(self, quantiles=(0.1, 0.5, 0.9), params: dict | None = None):
        self.quantiles = tuple(float(q) for q in quantiles)
        self.params = params or {}
        self.models: dict[str, dict[float, HistGradientBoostingRegressor]] = {}
        self.feature_ranges: dict[str, list[float]] = {}
        self.decision_quantile: float = 0.5
        self.widen: dict[str, list[float]] = {}
        self.calibration: dict = {}
        self.metadata: dict = {}

    def _make(self, q: float, features: list[str], prefix: str, iter_factor: float = 1.0) -> HistGradientBoostingRegressor:
        mono = [1 if f in ("h", f"lin_inc_{prefix}") else 0 for f in features]
        p = self.params
        return HistGradientBoostingRegressor(
            loss="quantile", quantile=q,
            max_iter=max(20, int(int(p.get("max_iter", 400)) * iter_factor)), learning_rate=float(p.get("learning_rate", 0.05)),
            max_leaf_nodes=int(p.get("max_leaf_nodes", 31)), min_samples_leaf=int(p.get("min_samples_leaf", 40)),
            l2_regularization=float(p.get("l2_regularization", 1.0)),
            monotonic_cst=mono, random_state=int(p.get("random_state", 42)), early_stopping=False,
        )

    def _fit_quantiles(self, table: pd.DataFrame, quantiles, iter_factor: float = 1.0) -> dict:
        models = {}
        stride = int(self.params.get("origin_stride", 1))
        if stride > 1:  # neighbouring daily origins are highly correlated -> thin them
            day_no = (pd.to_datetime(table["date"]) - pd.Timestamp("2000-01-01")).dt.days
            table = table[(day_no % stride) == 0]
        for prefix in TARGETS:
            feats = horizon_features(prefix)
            m = table[f"target_{prefix}"].notna() & table[f"{prefix}_lvl"].notna()
            X = table.loc[m, feats]
            y = table.loc[m, f"target_{prefix}"].to_numpy(dtype=float)
            if len(X) < 200:
                raise ValueError(f"Not enough training rows for {prefix} ({len(X)})")
            models[prefix] = {q: self._make(q, feats, prefix, iter_factor).fit(X, y) for q in sorted(set(quantiles))}
        return models

    def fit(self, table: pd.DataFrame, thresholds: dict | None = None, calibrate: bool = True,
            cal_fraction: float = 0.25) -> "GBMForecaster":
        if calibrate and thresholds:
            self._calibrate(table, thresholds, cal_fraction)
        qs = set(self.quantiles) | {self.decision_quantile}
        self.models = self._fit_quantiles(table, qs)
        cols = sorted(set(sum((horizon_features(p) for p in TARGETS), [])))
        for c in cols:
            v = table[c].to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            if len(v):
                self.feature_ranges[c] = [float(np.percentile(v, 1)), float(np.percentile(v, 99))]
        return self

    def _calibrate(self, table: pd.DataFrame, thresholds: dict, cal_fraction: float) -> None:
        from .forecast import days_to_threshold  # local import to avoid a cycle

        dates = np.sort(table["date"].unique())
        cal_start = dates[int(len(dates) * (1 - cal_fraction))]
        fit_part = table[table["target_date"] < cal_start]
        cal_part = table[table["date"] >= cal_start].reset_index(drop=True)
        if fit_part[["target_npf"]].notna().sum().iloc[0] < 2000 or len(cal_part) < 300:
            self.calibration = {"skipped": "not enough data for a calibration split"}
            return
        grid = sorted(set(self.DECISION_GRID) | set(self.quantiles))
        models = self._fit_quantiles(fit_part, grid, iter_factor=0.5)  # lighter models suffice here
        H = int(cal_part["h"].max())
        # --- conformal widening of the outer band, per indicator and horizon
        lo_q, hi_q = min(self.quantiles), max(self.quantiles)
        nominal = hi_q - lo_q
        for prefix in TARGETS:
            feats = horizon_features(prefix)
            ok = cal_part[f"target_{prefix}"].notna().to_numpy()
            Xc = cal_part.loc[ok, feats]
            y = cal_part.loc[ok, f"target_{prefix}"].to_numpy(dtype=float)
            lo = models[prefix][lo_q].predict(Xc)
            hi = models[prefix][hi_q].predict(Xc)
            score = np.maximum(lo - y, y - hi)
            hh = Xc["h"].to_numpy().astype(int)
            w = []
            for h in range(1, H + 1):
                sc = score[hh == h]
                if len(sc) < 20:
                    w.append(w[-1] if w else 0.0)
                    continue
                level = min(1.0, nominal * (1 + 1 / len(sc)))
                w.append(max(0.0, float(np.quantile(sc, level))))
            self.widen[prefix] = w
        # --- decision quantile from days-to-threshold on calibration origins
        thr = {"npf": thresholds["npf_decline_pct"], "dp": thresholds["dp_rise_pct"]}
        preds = {q: {p: models[p][q].predict(cal_part[horizon_features(p)]) for p in TARGETS} for q in grid}
        errs = {q: [] for q in self.DECISION_GRID}
        for _, idx in cal_part.groupby(["train_id", "date"], sort=False).indices.items():
            idx = np.asarray(idx)[np.argsort(cal_part["h"].to_numpy()[idx])]
            if len(idx) != H:
                continue
            # observed first crossing (either indicator) within the horizon, same cycle
            true_d = np.inf
            for p in TARGETS:
                lvl = cal_part[f"{p}_lvl"].iat[idx[0]]
                if not np.isfinite(lvl) or lvl >= thr[p]:
                    true_d = -1
                    break
                yv = cal_part[f"y_{p}"].to_numpy()[idx]
                hit = np.flatnonzero(np.nan_to_num(yv, nan=-1e9) >= thr[p])
                if len(hit):
                    true_d = min(true_d, float(hit[0] + 1))
            if not np.isfinite(true_d) or true_d <= 0:
                continue
            for q in self.DECISION_GRID:
                best = np.inf
                for p in TARGETS:
                    lvl = float(cal_part[f"{p}_lvl"].iat[idx[0]])
                    d, _ = days_to_threshold(lvl, lvl + preds[q][p][idx], thr[p], 365)
                    best = min(best, np.inf if d is None else d)
                errs[q].append((best if np.isfinite(best) else 60.0) - true_d)
        stats = {q: (float(np.median(e)), float(np.mean(np.abs(e))), len(e)) for q, e in errs.items() if e}
        if stats:
            self.decision_quantile = min(stats, key=lambda q: abs(stats[q][0]))
        self.calibration = {
            "calibration_start": str(pd.Timestamp(cal_start).date()),
            "decision_quantile": self.decision_quantile,
            "decision_grid_median_err_mae_n": {str(q): v for q, v in stats.items()},
            "band_widening_by_h": {p: [round(x, 3) for x in w] for p, w in self.widen.items()},
        }

    def predict(self, X: pd.DataFrame, prefix: str) -> dict:
        """Quantile increments {0.1, 0.5, 0.9 (conformal-widened), 'decision': q* path}."""
        feats = horizon_features(prefix)
        Xf = X[feats]
        raw = {q: m.predict(Xf) for q, m in self.models[prefix].items()}
        qs = list(self.quantiles)
        stacked = np.sort(np.vstack([raw[q] for q in qs]), axis=0)
        out = {q: stacked[k].copy() for k, q in enumerate(qs)}
        w = self.widen.get(prefix)
        if w:
            hidx = np.clip(X["h"].to_numpy().astype(int) - 1, 0, len(w) - 1)
            wv = np.asarray(w)[hidx]
            out[qs[0]] -= wv
            out[qs[-1]] += wv
        out["decision"] = np.maximum(raw[self.decision_quantile], out[0.5]) if 0.5 in out else raw[self.decision_quantile]
        return out

    # -- persistence ----------------------------------------------------------
    def save(self, directory: str | Path, extra_metadata: dict | None = None) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        meta = {
            "model_type": "GBMForecaster",
            "package_version": __version__,
            "sklearn_version": sklearn.__version__,
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "quantiles": list(self.quantiles),
            "targets": TARGETS,
            "features": {p: horizon_features(p) for p in TARGETS},
            "params": self.params,
            "feature_ranges": self.feature_ranges,
        }
        meta.update(self.metadata)
        meta.update(extra_metadata or {})
        self.metadata = meta
        meta["calibration"] = self.calibration
        meta["decision_quantile"] = self.decision_quantile
        joblib.dump({"models": self.models, "quantiles": self.quantiles, "params": self.params,
                     "feature_ranges": self.feature_ranges, "decision_quantile": self.decision_quantile,
                     "widen": self.widen, "calibration": self.calibration, "metadata": meta},
                    d / "model.joblib", compress=3)
        digest = hashlib.sha256((d / "model.joblib").read_bytes()).hexdigest()
        meta["sha256"] = digest
        (d / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
        return d

    @classmethod
    def load(cls, directory: str | Path) -> "GBMForecaster":
        d = Path(directory)
        f = d / "model.joblib"
        if not f.exists():
            raise FileNotFoundError(
                f"No model found at {f}. Train one with `rofoul train <data.csv> --out {d}` "
                "or point model.path / ROFOUL_MODEL_PATH at an existing model directory.")
        blob = joblib.load(f)
        obj = cls(quantiles=blob["quantiles"], params=blob["params"])
        obj.models = blob["models"]
        obj.feature_ranges = blob.get("feature_ranges", {})
        obj.decision_quantile = blob.get("decision_quantile", 0.5)
        obj.widen = blob.get("widen", {})
        obj.calibration = blob.get("calibration", {})
        obj.metadata = blob.get("metadata", {})
        obj.metadata["path"] = str(d.resolve())
        return obj
