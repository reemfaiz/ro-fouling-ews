import numpy as np
import pandas as pd
import pytest

from rofoul.features import build_training_table, expand_horizons, origin_table
from rofoul.forecast import alert_level, days_to_threshold
from rofoul.models import GBMForecaster, LinearTrendModel, PersistenceModel
from rofoul.pipeline import prepare_daily


def test_days_to_threshold_interpolates():
    path = 5 + 0.5 * np.arange(1, 31)  # reaches 10 at h = 10
    d, beyond = days_to_threshold(5.0, path, 10.0)
    assert d == pytest.approx(10.0) and not beyond


def test_days_to_threshold_edge_cases():
    assert days_to_threshold(12.0, np.full(30, 13.0), 10.0) == (0.0, False)      # already exceeded
    assert days_to_threshold(5.0, np.full(30, 5.0), 10.0) == (None, True)        # flat -> never
    d, beyond = days_to_threshold(1.0, 1 + 0.1 * np.arange(1, 31), 10.0)          # 0.1 %/day -> day 90
    assert beyond and d == pytest.approx(90.0, abs=0.5)
    assert days_to_threshold(1.0, 1 + 0.01 * np.arange(1, 31), 10.0, 180) == (None, True)
    assert days_to_threshold(float("nan"), np.ones(30), 10.0) == (None, False)


def test_alert_levels():
    assert alert_level(3, 2, 7, 14) == "RED"
    assert alert_level(10, 8, 7, 14) == "AMBER"
    assert alert_level(30, 20, 7, 14) == "GREEN"
    assert alert_level(30, 5, 7, 14) == "AMBER"   # pessimistic estimate escalates GREEN -> AMBER
    assert alert_level(None, None, 7, 14) == "GREEN"


def test_baselines():
    X = pd.DataFrame({"lin_inc_npf": [0.5, 1.0, np.nan]})
    assert PersistenceModel().predict(X, "npf")[0.5].tolist() == [0, 0, 0]
    assert LinearTrendModel().predict(X, "npf")[0.5].tolist() == [0.5, 1.0, 0.0]


def test_gbm_quantiles_ordered_monotone_and_roundtrip(tiny_model_dir, sim, cfg):
    model = GBMForecaster.load(tiny_model_dir)
    daily, _, _ = prepare_daily(sim[0], cfg)
    ot = origin_table(daily[daily["train_id"] == "B"], cfg).iloc[[120]]
    X = expand_horizons(ot, None, 30, with_targets=False)
    q = model.predict(X, "npf")
    assert np.all(q[0.1] <= q[0.5] + 1e-9) and np.all(q[0.5] <= q[0.9] + 1e-9)
    assert np.all(np.diff(q[0.5]) >= -1e-9)        # non-decreasing in horizon (monotonic constraint)
    assert np.all(q["decision"] >= q[0.5] - 1e-9)
    # save / load gives identical predictions
    model2 = GBMForecaster.load(tiny_model_dir)
    np.testing.assert_allclose(model2.predict(X, "dp")[0.5], model.predict(X, "dp")[0.5])


def test_gbm_beats_persistence_in_sample(tiny_model_dir, sim, cfg):
    model = GBMForecaster.load(tiny_model_dir)
    daily, _, _ = prepare_daily(sim[0], cfg)
    tab = build_training_table(daily, cfg)
    tab = tab[tab["target_npf"].notna() & (tab["h"] == 14)]
    err_gbm = np.abs(model.predict(tab, "npf")[0.5] - tab["target_npf"]).mean()
    err_pers = np.abs(tab["target_npf"]).mean()
    assert err_gbm < err_pers
