"""Input loading, column mapping, unit conversion and data-quality checks.

`load_and_validate()` turns an arbitrary plant CSV into a clean, canonical
DataFrame (internal units) plus a `ValidationReport`.  Problems that make the
data unusable raise `InputValidationError` with human-readable messages;
problems that can be repaired are fixed and reported as warnings.
"""
from __future__ import annotations

import io
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .units import FIELD_QUANTITY, UnitError, canonical_unit, split_header_unit, supported_units, to_internal

NUMERIC_FIELDS = [
    "feed_temp", "feed_pressure", "concentrate_pressure", "diff_pressure", "permeate_pressure",
    "feed_flow", "permeate_flow", "concentrate_flow", "feed_tds", "feed_conductivity",
    "permeate_tds", "permeate_conductivity",
]

ALIASES: dict[str, list[str]] = {
    "timestamp": ["timestamp", "datetime", "date", "time", "ts", "dateandtime", "datetimeutc", "localtime"],
    "train_id": ["trainid", "train", "unit", "unitid", "rack", "skid", "line", "trainname"],
    "feed_temp": ["feedtemp", "feedtemperature", "temperature", "temp", "watertemp", "feedwatertemperature",
                  "tfeed", "feedwatertemp"],
    "feed_pressure": ["feedpressure", "pfeed", "hppdischargepressure", "inletpressure", "feedpress",
                      "rofeedpressure", "membranefeedpressure"],
    "concentrate_pressure": ["concentratepressure", "brinepressure", "rejectpressure", "pconc", "pbrine",
                             "concpressure", "pconcentrate"],
    "diff_pressure": ["diffpressure", "differentialpressure", "dp", "deltap", "pressuredrop", "feedconcdp",
                      "feedtoconcentratedp", "dpfeedconc"],
    "permeate_pressure": ["permeatepressure", "productpressure", "ppermeate", "permpressure", "permeatebackpressure"],
    "feed_flow": ["feedflow", "qfeed", "feedflowrate", "inletflow", "rofeedflow"],
    "permeate_flow": ["permeateflow", "productflow", "qpermeate", "permflow", "permeateflowrate",
                      "productwaterflow", "qp"],
    "concentrate_flow": ["concentrateflow", "brineflow", "rejectflow", "qconc", "qbrine", "concflow"],
    "feed_tds": ["feedtds", "tdsfeed", "feedsalinity", "salinity", "feedtdsmgl"],
    "feed_conductivity": ["feedconductivity", "feedcond", "feedec", "conductivityfeed", "ecfeed"],
    "permeate_tds": ["permeatetds", "producttds", "tdspermeate", "permeatesalinity"],
    "permeate_conductivity": ["permeateconductivity", "permeatecond", "permeateec", "productconductivity", "ecpermeate"],
    "cip": ["cip", "cleaning", "cipflag", "cleaningevent", "cipevent", "cleaningflag"],
}

# At least one field of each group must be present.
REQUIRED_GROUPS: list[tuple[str, ...]] = [
    ("timestamp",),
    ("feed_temp",),
    ("feed_pressure",),
    ("permeate_flow",),
    ("concentrate_pressure", "diff_pressure"),
    ("feed_flow", "concentrate_flow"),
    ("feed_tds", "feed_conductivity"),
]


@dataclass
class Issue:
    code: str
    message: str
    column: str | None = None
    train_id: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    errors: list[Issue] = field(default_factory=list)
    warnings: list[Issue] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def error(self, code, message, **kw):
        self.errors.append(Issue(code, message, **kw))

    def warn(self, code, message, **kw):
        self.warnings.append(Issue(code, message, **kw))

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "errors": [asdict(e) for e in self.errors],
            "warnings": [asdict(w) for w in self.warnings],
            "info": self.info,
        }

    def format(self) -> str:
        lines = []
        for e in self.errors:
            lines.append(f"ERROR   [{e.code}] {e.message}")
        for w in self.warnings:
            lines.append(f"WARNING [{w.code}] {w.message}")
        if not lines:
            lines.append("OK: no problems found")
        return "\n".join(lines)


class InputValidationError(ValueError):
    def __init__(self, report: ValidationReport):
        self.report = report
        super().__init__("Input validation failed:\n" + report.format())


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


# ---------------------------------------------------------------------------
# Column mapping
# ---------------------------------------------------------------------------
def map_columns(headers: list[str], cfg: dict, report: ValidationReport) -> dict[str, tuple[str, str | None]]:
    """Return canonical field -> (csv header, unit token)."""
    colcfg = cfg["input"]["columns"]
    parsed = {h: split_header_unit(h) for h in headers}
    used: set[str] = set()
    mapping: dict[str, tuple[str, str | None]] = {}

    def assign(fld, header, unit_from_header):
        cfg_unit = colcfg.get(fld, {}).get("unit")
        unit = unit_from_header or cfg_unit
        if unit_from_header and cfg_unit and canonical_unit(cfg_unit) != unit_from_header:
            report.info.setdefault("unit_overrides", {})[fld] = (
                f"header '{header}' declares {unit_from_header}; overriding config unit {cfg_unit}")
        mapping[fld] = (header, unit)
        used.add(header)

    for fld in ALIASES:
        want = colcfg.get(fld, {}).get("name") or fld
        # 1) exact configured name (case-insensitive)
        hit = next((h for h in headers if h not in used and h.strip().lower() == str(want).lower()), None)
        if hit is not None:
            base, unit = parsed[hit]
            # unit suffix inside the configured name is part of the name, don't treat as override
            assign(fld, hit, None if _norm(hit) == _norm(want) and colcfg.get(fld, {}).get("unit") else unit)
            continue
        # 2) normalized base of configured name, 3) aliases
        want_base = _norm(split_header_unit(str(want))[0])
        keys = [want_base] + ALIASES[fld]
        for key in keys:
            hit = next((h for h in headers if h not in used and _norm(parsed[h][0]) == key), None)
            if hit is not None:
                assign(fld, hit, parsed[hit][1])
                break
    return mapping


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def read_csv_any(source) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        return source.copy()
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    if isinstance(source, (str, Path)) and not Path(source).exists():
        raise FileNotFoundError(f"Input file not found: {source}")
    try:
        df = pd.read_csv(source, sep=None, engine="python", encoding_errors="replace")
    except Exception as exc:  # pragma: no cover - pandas error types vary
        raise ValueError(f"Could not parse CSV: {exc}") from exc
    df.columns = [str(c).strip().lstrip("﻿") for c in df.columns]
    return df


def _parse_timestamps(raw: pd.Series, cfg: dict) -> pd.Series:
    fmt = cfg["input"].get("timestamp_format")
    dayfirst = bool(cfg["input"].get("dayfirst", False))
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        try:
            ts = pd.to_datetime(raw, format=fmt, errors="coerce", utc=False, dayfirst=dayfirst)
        except (ValueError, TypeError):
            ts = pd.to_datetime(raw, format="mixed", errors="coerce", utc=True, dayfirst=dayfirst)
        if ts.isna().mean() > 0.5 and fmt is None:
            ts2 = pd.to_datetime(raw, format="mixed", errors="coerce", utc=True, dayfirst=dayfirst)
            if ts2.notna().sum() > ts.notna().sum():
                ts = ts2
    if getattr(ts.dt, "tz", None) is not None:
        tz = cfg["input"].get("timezone")
        ts = ts.dt.tz_convert(tz) if tz else ts.dt.tz_convert("UTC")
        ts = ts.dt.tz_localize(None)
    return ts


def _plausibility_hint(field_name: str, values: pd.Series, unit: str | None, cfg: dict) -> str:
    """Suggest a unit that would make the values plausible."""
    quantity = FIELD_QUANTITY[field_name]
    rng_key = {"temperature": "feed_temp", "flow": "flow", "concentration": "feed_tds"}.get(quantity)
    if quantity == "pressure":
        rng_key = "diff_pressure" if field_name == "diff_pressure" else (
            "permeate_pressure" if field_name == "permeate_pressure" else "feed_pressure")
    if rng_key is None:
        return ""
    lo, hi = cfg["validation"]["ranges"][rng_key]
    med = float(np.nanmedian(values))
    for cand in supported_units(field_name):
        if cand == unit:
            continue
        conv = float(to_internal(np.array([med]), field_name, cand)[0])
        if lo <= conv <= hi:
            return (f" Values look like {cand}: set input.columns.{field_name}.unit: {cand} in the config "
                    f"or put the unit in the header, e.g. '{field_name} ({cand})'.")
    return ""


def load_and_validate(source, cfg: dict) -> tuple[pd.DataFrame, ValidationReport]:
    report = ValidationReport()
    vcfg = cfg["validation"]
    try:
        raw = read_csv_any(source)
    except (ValueError, FileNotFoundError) as exc:
        report.error("unreadable_file", str(exc))
        raise InputValidationError(report)
    if raw.empty or len(raw.columns) < 2:
        report.error("empty_file", "The CSV has no data rows or fewer than two columns.")
        raise InputValidationError(report)
    report.info["rows_in"] = int(len(raw))

    # -- column mapping -----------------------------------------------------
    try:
        mapping = map_columns(list(raw.columns), cfg, report)
    except UnitError as exc:
        report.error("bad_unit", str(exc))
        raise InputValidationError(report)
    if "timestamp" not in mapping:
        lower = {c.lower(): c for c in raw.columns}
        if "date" in lower and "time" in lower:
            raw["__ts"] = raw[lower["date"]].astype(str) + " " + raw[lower["time"]].astype(str)
            mapping["timestamp"] = ("__ts", None)
    report.info["column_mapping"] = {k: {"column": v[0], "unit": v[1]} for k, v in mapping.items()}

    for group in REQUIRED_GROUPS:
        if not any(g in mapping for g in group):
            want = " or ".join(f"'{cfg['input']['columns'].get(g, {}).get('name', g)}'" for g in group)
            aliases = ", ".join(ALIASES[group[0]][:4])
            report.error(
                "missing_column",
                f"Required column missing: {want} (field '{'/'.join(group)}'). "
                f"Accepted header aliases include: {aliases}. Map your header in input.columns of the config.",
                column="/".join(group),
            )
    if report.errors:
        report.info["columns_found"] = list(raw.columns)
        raise InputValidationError(report)

    # -- build canonical frame -----------------------------------------------
    df = pd.DataFrame(index=raw.index)
    df["timestamp"] = _parse_timestamps(raw[mapping["timestamp"][0]], cfg)
    if "train_id" in mapping:
        df["train_id"] = raw[mapping["train_id"][0]].astype(str).str.strip()
        df.loc[df["train_id"].isin(["", "nan", "None"]), "train_id"] = cfg["input"]["default_train_id"]
    else:
        df["train_id"] = str(cfg["input"]["default_train_id"])

    for fld in NUMERIC_FIELDS:
        if fld not in mapping:
            continue
        col, unit = mapping[fld]
        vals = pd.to_numeric(raw[col], errors="coerce")
        n_bad = int(vals.isna().sum() - raw[col].isna().sum())
        if n_bad > 0:
            frac = n_bad / len(raw)
            msg = f"{n_bad} non-numeric value(s) in '{col}' ({frac:.1%}); treated as missing."
            (report.error if frac > vcfg["max_bad_row_fraction"] else report.warn)("non_numeric", msg, column=col)
        try:
            df[fld] = to_internal(vals.astype(float), fld, unit)
        except UnitError as exc:
            report.error("bad_unit", f"Column '{col}': {exc}", column=col)
    if "cip" in mapping:
        c = raw[mapping["cip"][0]]
        df["cip"] = pd.to_numeric(c, errors="coerce").fillna(0).clip(0, 1).astype(int)
    if report.errors:
        raise InputValidationError(report)

    n_badts = int(df["timestamp"].isna().sum())
    if n_badts:
        frac = n_badts / len(df)
        msg = f"{n_badts} timestamp(s) could not be parsed ({frac:.1%}); those rows were dropped."
        if frac > vcfg["max_bad_row_fraction"]:
            report.error("bad_timestamp", msg + " Use ISO 8601, e.g. 2024-05-01 13:00, or set input.timestamp_format.")
            raise InputValidationError(report)
        report.warn("bad_timestamp", msg)
        df = df[df["timestamp"].notna()]

    # -- derived fields --------------------------------------------------------
    k_cond = float(cfg["input"]["conductivity_to_tds_factor"])
    if "feed_tds" not in df and "feed_conductivity" in df:
        df["feed_tds"] = df["feed_conductivity"] * k_cond
        report.info["feed_tds_from_conductivity_factor"] = k_cond
    if "permeate_tds" not in df and "permeate_conductivity" in df:
        df["permeate_tds"] = df["permeate_conductivity"] * k_cond
    if "permeate_pressure" not in df:
        df["permeate_pressure"] = 0.0
        report.warn("assumed_permeate_pressure",
                    "No permeate pressure column; 0 bar(g) assumed. NDP will be slightly overestimated.")
    if "diff_pressure" not in df:
        df["diff_pressure"] = df["feed_pressure"] - df["concentrate_pressure"]
    if "feed_flow" not in df:
        df["feed_flow"] = df["permeate_flow"] + df["concentrate_flow"]
    if "concentrate_flow" not in df:
        df["concentrate_flow"] = df["feed_flow"] - df["permeate_flow"]

    # -- unit plausibility (median based, catches psi-vs-bar, F-vs-C, g/L-vs-mg/L)
    rng = vcfg["ranges"]
    checks = [("feed_temp", rng["feed_temp"]), ("feed_pressure", rng["feed_pressure"]),
              ("diff_pressure", rng["diff_pressure"]), ("permeate_pressure", rng["permeate_pressure"]),
              ("permeate_flow", rng["flow"]), ("feed_flow", rng["flow"]), ("feed_tds", rng["feed_tds"])]
    for fld, (lo, hi) in checks:
        if fld not in df or df[fld].notna().sum() == 0:
            continue
        med = float(df[fld].median())
        if not (lo <= med <= hi):
            src = mapping.get(fld) or mapping.get("feed_conductivity" if fld == "feed_tds" else fld) or (fld, None)
            raw_vals = pd.to_numeric(raw[src[0]], errors="coerce") if src[0] in raw else df[fld]
            unit_used = src[1] or cfg["input"]["columns"].get(fld, {}).get("unit")
            hint = _plausibility_hint(fld, raw_vals, canonical_unit(unit_used) if unit_used else None, cfg) \
                if fld in FIELD_QUANTITY and src[0] in raw else ""
            report.error(
                "implausible_units",
                f"'{src[0]}' has median {med:.4g} after conversion from {unit_used}, outside the plausible "
                f"range [{lo}, {hi}] for {fld}.{hint}",
                column=src[0],
            )
    if report.errors:
        raise InputValidationError(report)

    df = _clean_rows(df, cfg, report)
    _gap_report(df, cfg, report)
    report.info["rows_out"] = int(len(df))
    report.info["trains"] = sorted(df["train_id"].unique().tolist())
    if len(df) == 0:
        report.error("no_valid_rows", "No valid operating rows remain after cleaning.")
        raise InputValidationError(report)
    return df.reset_index(drop=True), report


def _clean_rows(df: pd.DataFrame, cfg: dict, report: ValidationReport) -> pd.DataFrame:
    vcfg = cfg["validation"]
    n0 = len(df)
    df = df.sort_values(["train_id", "timestamp"])
    # duplicates -> mean
    dup = df.duplicated(["train_id", "timestamp"], keep=False)
    if dup.any():
        n_dup = int(df.duplicated(["train_id", "timestamp"]).sum())
        report.warn("duplicate_timestamps", f"{n_dup} duplicate timestamp row(s) were averaged.")
        agg = {c: ("max" if c == "cip" else "mean") for c in df.columns if c not in ("train_id", "timestamp")}
        df = df.groupby(["train_id", "timestamp"], as_index=False).agg(agg)

    # offline rows (train stopped): permeate flow < 5% of the train's median
    med_qp = df.groupby("train_id")["permeate_flow"].transform("median")
    offline = df["permeate_flow"].fillna(0) < 0.05 * med_qp
    if offline.any():
        report.info["offline_rows_removed"] = int(offline.sum())
        if "cip" in df:
            # keep the cleaning log, drop the (meaningless) process values while offline
            keep = offline & (df["cip"] > 0)
            df.loc[keep, [c for c in NUMERIC_FIELDS if c in df]] = np.nan
            offline = offline & ~keep
        df = df[~offline]

    # stuck sensors: long runs of identical values -> NaN
    for fld in ["feed_pressure", "permeate_flow", "feed_temp", "feed_tds", "diff_pressure", "concentrate_pressure"]:
        if fld not in df:
            continue
        for tid, g in df.groupby("train_id"):
            s = g[fld]
            run_id = (s.diff() != 0).cumsum()
            if len(g) < 3:
                continue
            ts = g["timestamp"]
            spans = ts.groupby(run_id).agg(["min", "max", "size"])
            long_runs = spans[((spans["max"] - spans["min"]).dt.total_seconds() / 3600
                               >= vcfg["stuck_sensor_hours"]) & (spans["size"] >= 6)]
            if len(long_runs):
                bad = run_id.isin(long_runs.index)
                df.loc[bad[bad].index, fld] = np.nan
                first = long_runs.iloc[0]
                report.warn("stuck_sensor",
                            f"Train {tid}: '{fld}' was constant for >= {vcfg['stuck_sensor_hours']} h in "
                            f"{len(long_runs)} period(s) (first {first['min']} -> {first['max']}); values ignored.",
                            column=fld, train_id=str(tid))

    # spikes: isolated points far from a centred 5-sample median.  A centred
    # window (2 samples ahead) is used so that genuine level shifts (cleanings,
    # set-point changes) are NOT removed; the look-ahead only decides whether a
    # single reading is kept, and daily medians are used downstream anyway.
    z = vcfg["outlier_mad_z"]
    n_spikes = 0
    for fld in ["feed_pressure", "diff_pressure", "permeate_flow", "feed_flow", "feed_temp", "feed_tds"]:
        if fld not in df:
            continue
        g = df.groupby("train_id")[fld]
        med = g.transform(lambda s: s.rolling(5, center=True, min_periods=3).median())
        resid = (df[fld] - med).abs()
        mad = resid.groupby(df["train_id"]).transform(lambda s: s.rolling(96, min_periods=10).median())
        scale = 1.4826 * mad + 1e-3 * med.abs() + 1e-9
        spike = (resid / scale > z) & med.notna()
        n_spikes += int(spike.sum())
        df.loc[spike, fld] = np.nan
    if n_spikes:
        report.warn("spikes_removed", f"{n_spikes} isolated spike value(s) (robust z > {z}) were removed.")

    # physical consistency
    req = ["feed_temp", "feed_pressure", "diff_pressure", "permeate_flow", "feed_flow", "feed_tds"]
    rec = df["permeate_flow"] / df["feed_flow"]
    lo, hi = vcfg["ranges"]["recovery"]
    bad_phys = (rec <= lo) | (rec >= hi) | (df["diff_pressure"] <= 0) | (df["permeate_flow"] >= df["feed_flow"])
    n_phys = int(bad_phys.sum())
    if n_phys:
        report.warn("inconsistent_rows",
                    f"{n_phys} row(s) physically inconsistent (recovery outside [{lo}, {hi}], "
                    "permeate >= feed flow or DP <= 0) were removed.")
        df = df[~bad_phys]
    n_missing = int(df[req].isna().any(axis=1).sum())
    if n_missing:
        report.info["rows_with_missing_required_values"] = n_missing
    dropped = n0 - len(df.dropna(subset=req))
    if n0 and dropped / n0 > vcfg["max_bad_row_fraction"] + 0.3:
        report.error("too_many_bad_rows",
                     f"{dropped} of {n0} rows ({dropped / n0:.0%}) are unusable (offline, missing or inconsistent).")
        raise InputValidationError(report)
    return df


def _gap_report(df: pd.DataFrame, cfg: dict, report: ValidationReport) -> None:
    vcfg = cfg["validation"]
    gaps_out = {}
    for tid, g in df.groupby("train_id"):
        ts = g["timestamp"].sort_values()
        dt_h = ts.diff().dt.total_seconds() / 3600
        step = float(np.nanmedian(dt_h)) if len(dt_h) > 1 else np.nan
        big = dt_h[dt_h > max(vcfg["max_gap_hours"], 1.5 * (step if np.isfinite(step) else 0))]
        days = ts.dt.normalize().nunique()
        gaps_out[str(tid)] = {
            "sampling_interval_hours": None if not np.isfinite(step) else round(step, 3),
            "start": str(ts.iloc[0]) if len(ts) else None,
            "end": str(ts.iloc[-1]) if len(ts) else None,
            "days_with_data": int(days),
            "n_gaps": int(len(big)),
            "longest_gap_hours": float(big.max()) if len(big) else 0.0,
        }
        if len(big):
            top = big.sort_values(ascending=False).head(3)
            desc = "; ".join(f"{ts.loc[i] - pd.Timedelta(hours=float(h))} (+{h:.0f} h)" for i, h in top.items())
            n_long = int((big > 24 * vcfg["long_gap_days"]).sum())
            report.warn("data_gaps",
                        f"Train {tid}: {len(big)} gap(s) longer than {vcfg['max_gap_hours']} h "
                        f"(longest {big.max():.0f} h; {n_long} longer than {vcfg['long_gap_days']} days). "
                        f"Largest start at: {desc}.", train_id=str(tid))
        if days < vcfg["min_history_days"]:
            report.warn("short_history",
                        f"Train {tid}: only {days} day(s) of data; at least {vcfg['min_history_days']} are needed "
                        "for a forecast. Normalized values are still returned.", train_id=str(tid))
    report.info["trains_summary"] = gaps_out
