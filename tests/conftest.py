import numpy as np
import pandas as pd
import pytest

from rofoul.config import load_config
from rofoul.synthetic import simulate


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def sim():
    """Small synthetic plant: 2 trains x 240 days, hourly."""
    data, truth = simulate(start="2023-01-01", days=240, trains=("A", "B"), seed=5)
    return data, truth


@pytest.fixture(scope="session")
def sim_csv(sim, tmp_path_factory):
    p = tmp_path_factory.mktemp("data") / "plant.csv"
    sim[0].to_csv(p, index=False)
    return p


@pytest.fixture(scope="session")
def tiny_model_dir(sim, cfg, tmp_path_factory):
    """A quickly-trained model so tests do not depend on models/v1."""
    from rofoul.evaluate import train_model
    from rofoul.features import build_training_table
    from rofoul.pipeline import prepare_daily

    c = load_config(overrides={"training": {"max_iter": 40, "calibrate": False}})
    daily, _, _ = prepare_daily(sim[0], c)
    model = train_model(build_training_table(daily, c), c)
    d = tmp_path_factory.mktemp("model") / "vtest"
    model.save(d, {"version": "vtest"})
    return d


@pytest.fixture()
def cfg_tiny(tiny_model_dir):
    return load_config(overrides={"model": {"path": str(tiny_model_dir)}})


def make_rows(n_hours=24 * 30, start="2024-01-01", **over):
    """Simple consistent hourly frame in canonical columns."""
    ts = pd.date_range(start, periods=n_hours, freq="h")
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "timestamp": ts, "train_id": "T1",
        "feed_temp_c": 25.0 + 3 * np.sin(np.arange(n_hours) / 200),
        "feed_pressure_bar": 60.0, "concentrate_pressure_bar": 58.5, "permeate_pressure_bar": 1.0,
        "feed_flow_m3h": 250.0, "permeate_flow_m3h": 100.0, "feed_tds_mgl": 40000.0, "permeate_tds_mgl": 150.0,
    })
    for c in df.columns[3:]:  # small measurement noise so no channel looks "stuck"
        df[c] = df[c] * (1 + rng.normal(0, 0.002, n_hours))
    for k, v in over.items():
        df[k] = v
    return df
