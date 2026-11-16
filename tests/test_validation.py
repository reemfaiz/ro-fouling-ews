import numpy as np
import pandas as pd
import pytest

from rofoul.validation import InputValidationError, load_and_validate
from conftest import make_rows


def codes(report):
    return {e.code for e in report.errors}, {w.code for w in report.warnings}


def test_clean_input_passes(cfg):
    df, rep = load_and_validate(make_rows(), cfg)
    assert rep.ok
    assert set(["feed_temp", "diff_pressure", "concentrate_flow"]).issubset(df.columns)
    assert df["diff_pressure"].median() == pytest.approx(1.5, abs=0.2)


def test_missing_required_column_message(cfg):
    df = make_rows().drop(columns=["permeate_flow_m3h"])
    with pytest.raises(InputValidationError) as ei:
        load_and_validate(df, cfg)
    err = ei.value.report.errors[0]
    assert err.code == "missing_column"
    assert "permeate_flow" in err.message and "alias" in err.message


def test_either_or_columns(cfg):
    df = make_rows().drop(columns=["concentrate_pressure_bar"])
    with pytest.raises(InputValidationError) as ei:
        load_and_validate(df, cfg)
    assert "concentrate_pressure/diff_pressure" in ei.value.report.errors[0].column
    df["diff_pressure_bar"] = 1.5 + np.random.default_rng(1).normal(0, 0.01, len(df))
    out, rep = load_and_validate(df, cfg)  # DP column alone is enough
    assert out["diff_pressure"].notna().all()


def test_fahrenheit_under_celsius_header_is_caught_with_hint(cfg):
    df = make_rows()
    df["feed_temp_c"] = df["feed_temp_c"] * 9 / 5 + 32
    with pytest.raises(InputValidationError) as ei:
        load_and_validate(df, cfg)
    msg = ei.value.report.errors[0].message
    assert ei.value.report.errors[0].code == "implausible_units"
    assert "degF" in msg


def test_psi_under_bar_header_is_caught(cfg):
    df = make_rows()
    for c in ["feed_pressure_bar", "concentrate_pressure_bar"]:
        df[c] = df[c] / 0.0689475729
    with pytest.raises(InputValidationError) as ei:
        load_and_validate(df, cfg)
    assert any("psi" in e.message for e in ei.value.report.errors)


def test_units_in_header_are_converted(cfg):
    df = make_rows()
    us = pd.DataFrame({
        "Date/Time": df["timestamp"].dt.strftime("%m/%d/%Y %H:%M"),
        "Feed Temperature (°F)": df["feed_temp_c"] * 9 / 5 + 32,
        "Feed Pressure (psi)": df["feed_pressure_bar"] / 0.0689475729,
        "Concentrate Pressure (psi)": df["concentrate_pressure_bar"] / 0.0689475729,
        "Feed Flow (gpm)": df["feed_flow_m3h"] / 0.227124707,
        "Permeate Flow (gpm)": df["permeate_flow_m3h"] / 0.227124707,
        "Feed Conductivity (mS/cm)": df["feed_tds_mgl"] / 0.7 / 1000,
    })
    out, rep = load_and_validate(us, cfg)
    assert rep.ok
    assert out["feed_pressure"].median() == pytest.approx(60, rel=0.01)
    assert out["feed_temp"].median() == pytest.approx(df["feed_temp_c"].median(), rel=0.01)
    assert out["feed_tds"].median() == pytest.approx(40000, rel=0.01)
    assert "assumed_permeate_pressure" in codes(rep)[1]


def test_non_numeric_and_duplicates_and_gaps(cfg):
    df = make_rows()
    df["feed_pressure_bar"] = df["feed_pressure_bar"].astype(object)
    df.loc[3, "feed_pressure_bar"] = "n/a"
    df = pd.concat([df, df.iloc[[10]]])                                  # duplicate row
    df = df[~df["timestamp"].between("2024-01-10", "2024-01-12")]        # 2-day gap
    out, rep = load_and_validate(df, cfg)
    errs, warns = codes(rep)
    assert not errs
    assert {"non_numeric", "duplicate_timestamps", "data_gaps"} <= warns


def test_stuck_sensor_flagged(cfg):
    df = make_rows()
    df.loc[100:160, "feed_pressure_bar"] = 61.0
    out, rep = load_and_validate(df, cfg)
    assert "stuck_sensor" in codes(rep)[1]
    assert out["feed_pressure"].isna().sum() >= 60


def test_mostly_garbage_timestamps_rejected(cfg):
    df = make_rows()
    df["timestamp"] = df["timestamp"].astype(str)
    df.loc[: len(df) // 2, "timestamp"] = "not a date"
    with pytest.raises(InputValidationError) as ei:
        load_and_validate(df, cfg)
    assert ei.value.report.errors[0].code == "bad_timestamp"


def test_empty_file(cfg, tmp_path):
    p = tmp_path / "empty.csv"
    p.write_text("timestamp,feed_temp_c\n")
    with pytest.raises(InputValidationError):
        load_and_validate(p, cfg)


def test_short_history_warning(cfg):
    out, rep = load_and_validate(make_rows(n_hours=24 * 5), cfg)
    assert "short_history" in codes(rep)[1]
