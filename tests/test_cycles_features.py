import numpy as np
import pandas as pd
import pytest

from rofoul.cycles import add_cycles
from rofoul.features import TARGETS, build_training_table, expand_horizons, origin_table
from rofoul.pipeline import prepare_daily


def _daily(npf, dp=None, cip=None):
    n = len(npf)
    d = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=n, freq="D"), "train_id": "T1",
                      "npf": npf, "ndp_norm": dp if dp is not None else np.full(n, 1.5),
                      "nsp": np.full(n, 0.3), "feed_temp": 25.0, "feed_tds": 40000.0, "recovery": 0.4,
                      "permeate_flow": 100.0})
    if cip is not None:
        d["cip"] = cip
    return d


def test_auto_detects_cleaning_jump(cfg):
    npf = np.r_[np.linspace(100, 88, 40), np.linspace(99, 90, 30)]
    g = add_cycles(_daily(npf), cfg)
    assert g["cycle_start"].sum() == 2
    assert g.loc[g["cycle_start"], "date"].iloc[1] == pd.Timestamp("2024-01-01") + pd.Timedelta(days=40)
    assert g["cycle_source"].iloc[0] == "auto_detected"


def test_cip_column_defines_cycles(cfg):
    npf = np.r_[np.linspace(100, 92, 30), [np.nan], np.linspace(99, 95, 20)]
    cip = np.zeros(51); cip[30] = 1
    g = add_cycles(_daily(npf, cip=cip), cfg)
    starts = g.index[g["cycle_start"]].tolist()
    assert starts == [0, 31]
    assert g.loc[31, "days_since_clean"] == 0


def test_baseline_is_causal_and_decline_correct(cfg):
    npf = np.linspace(100, 80, 30)
    g = add_cycles(_daily(npf), cfg)
    # day 0 baseline only uses day 0
    assert g.loc[0, "npf_baseline"] == pytest.approx(100.0)
    # after baseline_days the baseline is frozen to the median of the first 3 days
    assert g.loc[20, "npf_baseline"] == pytest.approx(np.median(npf[:3]))
    assert g.loc[29, "npf_decline_pct"] == pytest.approx(100 * (1 - 80 / np.median(npf[:3])))


def test_features_do_not_use_the_future(cfg, sim):
    daily, _, _ = prepare_daily(sim[0], cfg)
    g = daily[daily["train_id"] == "A"].reset_index(drop=True)
    t = 150
    full = origin_table(g, cfg)
    cut = origin_table(g.iloc[: t + 1], cfg)
    a = full[full["row"] == t].drop(columns=["row"]).iloc[0]
    b = cut[cut["row"] == t].drop(columns=["row"]).iloc[0]
    pd.testing.assert_series_equal(a, b, check_names=False)


def test_targets_never_cross_a_cleaning(cfg, sim):
    daily, _, _ = prepare_daily(sim[0], cfg)
    tab = build_training_table(daily, cfg)
    g = daily.set_index(["train_id", "date"])["cycle_id"]
    lab = tab[tab["target_npf"].notna()].sample(2000, random_state=0)
    tgt_cycle = g.reindex(pd.MultiIndex.from_arrays([lab["train_id"], lab["target_date"]])).to_numpy()
    assert (tgt_cycle == lab["cycle_id"].to_numpy()).all()
    assert (lab["target_date"] > lab["date"]).all()


def test_expand_horizons_linear_increment(cfg, sim):
    daily, _, _ = prepare_daily(sim[0], cfg)
    ot = origin_table(daily[daily["train_id"] == "A"], cfg).iloc[[100]]
    X = expand_horizons(ot, None, 30, with_targets=False)
    assert len(X) == 30 and X["h"].tolist() == list(range(1, 31))
    s = ot["npf_s14"].iloc[0]
    assert X["lin_inc_npf"].iloc[9] == pytest.approx(10 * s)
