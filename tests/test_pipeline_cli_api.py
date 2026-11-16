import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rofoul.api import create_app
from rofoul.cli import main
from rofoul.pipeline import run

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample" / "plant_sample.csv"
SAMPLE_US = ROOT / "data" / "sample" / "plant_sample_us_units.csv"
BAD = ROOT / "data" / "sample" / "bad_input.csv"


def test_pipeline_end_to_end(sim_csv, cfg_tiny):
    res = run(sim_csv, cfg_tiny)
    assert {s["train_id"] for s in res.summary} == {"A", "B"}
    for s in res.summary:
        assert s["status"] == "ok"
        assert s["alert"] in {"GREEN", "AMBER", "RED"}
        assert "estimate" in s["days_to_clean"]
    assert len(res.forecast) == 2 * cfg_tiny["forecast"]["horizon_days"]
    assert {"npf_decline_pct_p50", "dp_rise_pct_p90", "npf_decline_pct_baseline"} <= set(res.forecast.columns)
    json.dumps(res.to_dict(), default=str)  # serialisable


def test_pipeline_falls_back_without_model(sim_csv, cfg):
    res = run(sim_csv, cfg, model=None)
    assert res.summary[0]["model_used"].startswith("linear_trend")


def test_cli_predict_writes_outputs(tmp_path, tiny_model_dir):
    rc = main(["predict", str(SAMPLE), "-o", str(tmp_path), "--model", str(tiny_model_dir)])
    assert rc == 0
    for f in ("normalized_daily.csv", "forecast.csv", "summary.json"):
        assert (tmp_path / f).exists()
    summary = json.loads((tmp_path / "summary.json").read_text())["summary"]
    assert summary[0]["train_id"] == "P1"


def test_cli_us_units_file_matches_si_file(tmp_path, tiny_model_dir):
    a, b = tmp_path / "a", tmp_path / "b"
    assert main(["predict", str(SAMPLE), "-o", str(a), "--model", str(tiny_model_dir)]) == 0
    assert main(["predict", str(SAMPLE_US), "-o", str(b), "--model", str(tiny_model_dir)]) == 0
    sa = json.loads((a / "summary.json").read_text())["summary"][0]["current"]
    sb = json.loads((b / "summary.json").read_text())["summary"][0]["current"]
    assert sb["npf_decline_pct"] == pytest.approx(sa["npf_decline_pct"], abs=0.6)


def test_cli_validate_bad_file_exit_code(capsys):
    rc = main(["validate", str(BAD)])
    out = capsys.readouterr().out
    assert rc == 2
    assert "missing_column" in out and "permeate_flow" in out


def test_cli_missing_file():
    assert main(["validate", "does_not_exist.csv"]) == 2


@pytest.fixture()
def client(cfg_tiny):
    return TestClient(create_app(cfg_tiny))


def test_api_health_and_model(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["model_loaded"] is True
    assert client.get("/model").json()["version"] == "vtest"


def test_api_predict(client):
    with open(SAMPLE, "rb") as fh:
        r = client.post("/predict?include_daily=true", files={"file": ("plant.csv", fh, "text/csv")})
    assert r.status_code == 200
    body = r.json()
    assert body["summary"][0]["alert"] in {"GREEN", "AMBER", "RED"}
    assert len(body["forecast"]) == 30 and len(body["normalized_daily"]) > 100


def test_api_rejects_bad_file_with_clear_errors(client):
    with open(BAD, "rb") as fh:
        r = client.post("/predict", files={"file": ("bad.csv", fh, "text/csv")})
    assert r.status_code == 422
    errs = r.json()["errors"]
    assert errs and errs[0]["code"] == "missing_column"


def test_api_normalize_csv(client):
    with open(SAMPLE, "rb") as fh:
        r = client.post("/normalize?format=csv", files={"file": ("plant.csv", fh, "text/csv")})
    assert r.status_code == 200 and r.text.startswith("train_id,date")


def test_api_empty_upload(client):
    r = client.post("/predict", files={"file": ("e.csv", b"", "text/csv")})
    assert r.status_code == 422
