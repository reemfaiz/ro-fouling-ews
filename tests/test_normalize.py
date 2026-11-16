import numpy as np
import pandas as pd
import pytest

from rofoul.normalize import (feed_brine_concentration, normalize, osmotic_pressure, tcf, to_daily,
                              water_viscosity)
from rofoul.validation import load_and_validate
from conftest import make_rows


def test_tcf_reference_and_monotonic():
    assert tcf(25.0) == pytest.approx(1.0)
    assert tcf(25.0, "astm") == pytest.approx(1.0)
    assert tcf(30.0, "astm") == pytest.approx(1.03 ** 5)
    t = np.linspace(5, 40, 50)
    assert np.all(np.diff(tcf(t)) > 0)
    # FilmTec constants: 3020 below 25 C, 2640 above
    assert tcf(15.0) == pytest.approx(np.exp(3020 * (1 / 298.15 - 1 / 288.15)))
    assert tcf(35.0) == pytest.approx(np.exp(2640 * (1 / 298.15 - 1 / 308.15)))


def test_hand_calculated_ndp():
    """Worked example: Pf=60, DP=2, Pp=1, TDS=40 g/L, R=0.4, 25 C, permeate 200 mg/L."""
    c_fc = feed_brine_concentration(40000, 0.4)
    assert c_fc == pytest.approx(40000 * np.log(1 / 0.6) / 0.4)
    pi_fc = osmotic_pressure(c_fc, 25.0, 0.00077)
    pi_p = osmotic_pressure(200, 25.0, 0.00077)
    ndp = 60 - 2 / 2 - 1 - (pi_fc - pi_p)
    assert ndp == pytest.approx(18.82, abs=0.02)


def _clean_membrane_frame(cfg, n=24 * 40, seed=0):
    """A plant that obeys exactly the physics assumed by the normalization, with
    constant membrane permeability but varying temperature, salinity, set-points."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="h")
    T = 20 + 10 * rng.random(n)
    Cf = 38000 + 6000 * rng.random(n)
    R = 0.35 + 0.1 * rng.random(n)
    Qp = 80 + 40 * rng.random(n)
    Qf = Qp / R
    Qc = Qf - Qp
    A = 5.0  # m3/h/bar at 25 C
    k = cfg["normalization"]["osmotic_coeff_bar_per_mgl"]
    dp = 1.5 * (((Qf + Qc) / 2) / 200) ** cfg["normalization"]["dp_flow_exponent"] \
        * (water_viscosity(T) / water_viscosity(25.0)) ** cfg["normalization"]["dp_viscosity_exponent"]
    Cp = 0.004 * Cf
    pi_fc = osmotic_pressure(feed_brine_concentration(Cf, R), T, k)
    ndp = Qp / (A * tcf(T))
    Pf = ndp + dp / 2 + 1.0 + (pi_fc - osmotic_pressure(Cp, T, k))
    return pd.DataFrame({"timestamp": ts, "train_id": "T1", "feed_temp_c": T, "feed_pressure_bar": Pf,
                         "concentrate_pressure_bar": Pf - dp, "permeate_pressure_bar": 1.0, "feed_flow_m3h": Qf,
                         "permeate_flow_m3h": Qp, "feed_tds_mgl": Cf, "permeate_tds_mgl": Cp})


def test_npf_and_ndp_invariant_for_clean_membrane(cfg):
    df, _ = load_and_validate(_clean_membrane_frame(cfg), cfg)
    norm, refs = normalize(df, cfg)
    npf = norm["npf"].to_numpy()
    raw = norm["permeate_flow"].to_numpy()
    assert np.isfinite(npf).mean() > 0.99               # (i.i.d. random inputs may trip the spike filter)
    assert np.nanstd(raw) / np.nanmean(raw) > 0.1       # raw flow varies a lot ...
    assert np.nanstd(npf) / np.nanmean(npf) < 1e-9      # ... normalized flow does not
    ndp_norm = norm["ndp_norm"].to_numpy()
    assert np.nanstd(ndp_norm) / np.nanmean(ndp_norm) < 1e-9


def test_npf_tracks_permeability_loss(cfg):
    df = _clean_membrane_frame(cfg)
    # 10 % permeability loss in the second half: plant needs more pressure for the same flow
    half = len(df) // 2
    T = df["feed_temp_c"].to_numpy()
    extra_ndp = (df["permeate_flow_m3h"] / (5.0 * tcf(T))) * (1 / 0.9 - 1)
    df.loc[half:, "feed_pressure_bar"] += extra_ndp[half:]
    df.loc[half:, "concentrate_pressure_bar"] += extra_ndp[half:]
    d, _ = load_and_validate(df, cfg)
    norm, _ = normalize(d, cfg)
    ratio = norm["npf"].iloc[half:].median() / norm["npf"].iloc[:half].median()
    assert ratio == pytest.approx(0.9, rel=1e-6)


def test_reference_fixed_mode(cfg):
    import copy
    c = copy.deepcopy(cfg)
    c["normalization"]["reference"].update(mode="fixed", ndp_bar=20.0, permeate_flow_m3h=100.0,
                                           avg_feed_brine_flow_m3h=200.0, feed_tds_mgl=40000.0, recovery=0.4)
    df, _ = load_and_validate(make_rows(), c)
    norm, refs = normalize(df, c)
    assert refs["T1"]["ndp"] == 20.0
    assert norm["npf"].notna().all()


def test_daily_aggregation_coverage(cfg):
    df = make_rows(n_hours=24 * 5)
    df = df[~((df["timestamp"] >= "2024-01-03 02:00") & (df["timestamp"] < "2024-01-04 00:00"))]  # day 3: 2 h only
    d, _ = load_and_validate(df, cfg)
    norm, _ = normalize(d, cfg)
    daily = to_daily(norm, cfg)
    assert len(daily) == 5
    assert daily.loc[daily["date"] == "2024-01-03", "npf"].isna().all()
    assert daily.loc[daily["date"] != "2024-01-03", "npf"].notna().all()


def test_concentration_polarization_option(cfg):
    import copy
    c = copy.deepcopy(cfg)
    df, _ = load_and_validate(make_rows(), c)
    base, _ = normalize(df, c)
    c["normalization"]["concentration_polarization"].update(enabled=True, membrane_area_m2=5000, k_lmh=150)
    cp, _ = normalize(df, c)
    # CP raises the feed-side osmotic pressure -> lower NDP for every row
    assert (cp["ndp_bar"] < base["ndp_bar"]).all()
    c["normalization"]["concentration_polarization"]["membrane_area_m2"] = None
    with pytest.raises(ValueError):
        normalize(df, c)


def test_dayfirst_option(cfg):
    import copy
    c = copy.deepcopy(cfg)
    c["input"]["dayfirst"] = True
    df = make_rows(n_hours=24 * 3)
    df["timestamp"] = df["timestamp"].dt.strftime("%d/%m/%Y %H:%M")
    out, _ = load_and_validate(df, c)
    assert out["timestamp"].min() == pd.Timestamp("2024-01-01")
    assert out["timestamp"].max() == pd.Timestamp("2024-01-03 23:00")
