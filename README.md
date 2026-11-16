# rofoul — early warning of membrane fouling in RO desalination

`rofoul` takes a CSV of reverse-osmosis (RO) plant operating data and returns:

1. **Normalized performance** (ASTM D4516-style): normalized permeate flow (NPF), normalized
   feed-channel pressure drop (nDP) and normalized salt passage (NSP). These are corrected for
   temperature, pressure, salinity and recovery.
2. **A fouling forecast** for the next 30 days, as a median with a p10–p90 band: % NPF decline and
   % nDP rise relative to the post-cleaning baseline.
3. **Estimated days until cleaning (CIP) is needed**, with a range, the limiting indicator and a
   GREEN / AMBER / RED alert.

You can use it as a **CLI** or as a **REST API**. It ships as an installable Python package and as
a Docker image. The model is a versioned artefact selected in a config file, so you can retrain
and swap it without changing any code.

> Methodology and results: [`docs/TECHNICAL_REPORT.pdf`](docs/TECHNICAL_REPORT.pdf) (2 pages) ·
> Data sources: [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md) · AI usage: [`docs/AI_USAGE_LOG.md`](docs/AI_USAGE_LOG.md)

---

## 1. Setup

Choose **one** of the two options.

### Option A: Docker (one command)

```bash
docker compose up --build          # REST API on http://localhost:8000  (docs at /docs)
```

To run the CLI inside the same image on a file in the current directory:

```bash
docker build -t rofoul:1.0.0 .
docker run --rm -v "$PWD:/work" rofoul:1.0.0 predict /work/plant.csv -o /work/out
```

### Option B: Python package (Python ≥ 3.10)

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install .                                           # or: pip install -e ".[dev]" for tests
rofoul --help
```

## 2. Quick start

```bash
rofoul validate data/sample/plant_sample.csv           # data-quality report only
rofoul predict  data/sample/plant_sample.csv -o out    # full pipeline -> out/
rofoul serve                                           # REST API on :8000
```

Example output of `rofoul predict`:

```
train   last day     NPF decl % DP rise %  days->clean  range         alert   message
P1      2025-04-30          6.7       6.7            9  6-10          AMBER   Cleaning expected in ~9 days (limiting: npf_decline_pct).
        ! flux vs history = 0.846 is outside the training range [0.899, 1.18]; forecast is an extrapolation - ...
```

`out/` then contains:

| file | content |
|---|---|
| `normalized_daily.csv` | one row per train per day: NPF, nDP, NSP, cycle id, days since cleaning, `npf_decline_pct`, `dp_rise_pct`, … |
| `forecast.csv` | 30 rows per train: `*_p10 / *_p50 / *_p90` quantiles, `*_decision` path, and `*_baseline` (linear trend) for both indicators |
| `summary.json` | per train: current state, `days_to_clean` (`estimate`, `earliest_p90_fouling`, `latest_p10_fouling`, limiting indicator, estimated date), alert, warnings, plus the full validation report and model metadata |

The vendor-style file `data/sample/plant_sample_us_units.csv` (psi, °F, gpm, µS/cm, different
headers, no train column) works without any configuration, because units are read from the
headers. `data/sample/bad_input.csv` shows the validation errors.

## 3. REST API

| method | path | purpose |
|---|---|---|
| GET | `/health` | liveness, model loaded / version |
| GET | `/model` | metadata of the loaded model (training data, calibration, hash) |
| POST | `/model/reload` | reload config + model from disk (hot swap after retraining) |
| POST | `/validate` | data-quality report |
| POST | `/normalize` | normalized daily series (`?format=csv` for CSV) |
| POST | `/predict` | summary + forecast (+ `?include_daily=true` for the normalized series) |

```bash
curl -F "file=@data/sample/plant_sample.csv" http://localhost:8000/predict
```

Invalid input returns **HTTP 422** with the same structured report as the CLI:

```json
{"ok": false,
 "errors": [{"code": "missing_column",
             "message": "Required column missing: 'permeate_flow_m3h' (field 'permeate_flow'). Accepted header aliases include: permeateflow, productflow, ..."}],
 "warnings": [...]}
```

## 4. Input format

You need one CSV with one row per timestamp (and per train). Any sampling interval from minutes
to daily works; the data is aggregated to daily medians.

| field | default header | unit (default) | required |
|---|---|---|---|
| timestamp | `timestamp` | ISO 8601 recommended | **yes** |
| train / unit id | `train_id` | – | no (default `T1`) |
| feed temperature | `feed_temp_c` | degC (degF, K) | **yes** |
| feed pressure | `feed_pressure_bar` | bar (psi, kPa, MPa) | **yes** |
| concentrate pressure **or** feed–concentrate ΔP | `concentrate_pressure_bar` / `diff_pressure_bar` | bar | **one of** |
| permeate pressure | `permeate_pressure_bar` | bar | no (0 assumed, warning) |
| permeate flow | `permeate_flow_m3h` | m3/h (m3/d, gpm, L/s, L/min, MGD) | **yes** |
| feed flow **or** concentrate flow | `feed_flow_m3h` / `concentrate_flow_m3h` | m3/h | **one of** |
| feed TDS **or** feed conductivity | `feed_tds_mgl` / `feed_cond_uscm` | mg/L (g/L, ppm) / µS/cm (mS/cm) | **one of** |
| permeate TDS / conductivity | `permeate_tds_mgl` / `permeate_cond_uscm` | mg/L / µS/cm | no (needed for salt passage) |
| cleaning log | `cip` | 0/1 | no (cleanings are auto-detected; a log is better) |

**How headers are matched:** first the configured name, then common aliases (e.g. `Feed Pressure`,
`Product Flow`, `Brine Pressure`, `DP`, `Salinity`). Matching ignores case and punctuation.

**How units are chosen:** a unit in the header, such as `Feed Pressure (psi)` or `feed_temp_degF`,
takes precedence. Otherwise the unit from the config is used. Each column's median is checked
against a plausible physical range after conversion. Wrong units (°F under a °C header, psi
under bar, g/L under mg/L) are rejected with a suggested fix.

### What validation does

| problem | handling |
|---|---|
| missing required column / unknown unit | **error** listing the accepted names / units |
| implausible values (wrong units) | **error** with a hint, e.g. "Values look like degF: set `input.columns.feed_temp.unit: degF`" |
| unparseable timestamps / non-numeric cells | dropped with a warning (**error** above 20 % of rows) |
| duplicate timestamps | averaged, warning |
| train offline (permeate flow ≈ 0) | rows removed (the CIP log on those rows is kept) |
| stuck sensor (identical values ≥ 24 h) | values ignored, warning |
| isolated spikes (robust z > 8) | removed, warning |
| physically inconsistent rows (recovery outside 5–95 %, ΔP ≤ 0) | removed, warning |
| gaps > 6 h | warning with the largest gaps; days with < 4 h of data become missing |
| < 21 days of history for a train | that train is normalized but not forecast |

Exit codes: `0` ok, `2` validation failed, `1` other error.

## 5. Configuration, retraining and model swap

All defaults live in [`src/rofoul/default.yaml`](src/rofoul/default.yaml). That file documents
column mapping, units, validation limits, normalization constants, cleaning thresholds, forecast
horizon, alert levels and training hyper-parameters. Put plant-specific overrides in a small
YAML file such as [`config/plant.yaml`](config/plant.yaml) and pass it with `--config` or the
`ROFOUL_CONFIG` env var. It is deep-merged over the defaults.

```bash
rofoul predict plant.csv --config config/plant.yaml
```

**Retrain and swap the model without changing code:**

```bash
rofoul train my_history.csv --out models/v2            # ~30 s for 4 trains x 3 years; writes model.joblib + metadata.json
rofoul backtest my_history.csv --out reports/v2        # check it against the baselines first
# activate it: set  model.path: models/v2  in config/plant.yaml   (or ROFOUL_MODEL_PATH=models/v2)
curl -X POST http://localhost:8000/model/reload        # running API picks it up
rofoul info                                            # shows the active model's metadata
```

If no model can be loaded, the system still works. It falls back to the linear-trend baseline and
says so in `model_used` and in a warning.

**Regenerate the synthetic dataset and the shipped model:**

```bash
rofoul synth --out data/synthetic                      # 4 trains x 3 years, hourly (seeded)
rofoul train data/synthetic/plant_data.csv --out models/v1
rofoul backtest data/synthetic/plant_data.csv --out reports
rofoul backtest data/synthetic/plant_data.csv --out reports --holdout-train D
python scripts/analyze.py                              # figures + failure analysis
python scripts/extra_checks.py && python scripts/build_report.py   # report numbers + PDF (needs pandoc, xelatex)
```

## 6. Method in one paragraph

Hourly data is validated, converted to SI units and normalized:
`NPF = Qp·(NDP_ref/NDP)·(TCF_ref/TCF)`, with NDP computed from the log-mean feed-concentrate
osmotic pressure. ΔP is corrected for flow (exponent 1.5) and viscosity (exponent 0.3). The data
is then aggregated to daily medians and split into cleaning cycles, using the CIP log and/or
detected NPF jumps. Fouling is measured against each cycle's post-clean baseline. A direct
multi-horizon **quantile gradient-boosting** model predicts the change in each indicator 1–30
days ahead. Its inputs are causal features: current level, 7/14-day slopes, curvature, days since
cleaning, a biofouling temperature factor, salinity, recovery, flux and the previous cycle's
fouling rate. The model is constrained to be monotone in the horizon. The decision path quantile
and a conformal widening of the p10–p90 band are calibrated on the last 25 % of the training
period. Days-to-clean is the first day the decision path crosses **NPF −10 %** or **nDP +15 %**,
the manufacturer cleaning criteria (Hydranautics TSB107; DuPont FilmTec). Validation is a
rolling-origin back-test with purged training windows, compared with persistence and a 14-day
linear-trend baseline. Details and results are in the technical report.

## 7. Tests

```bash
pip install -e ".[dev]"
pytest                      # 51 tests, ~1-2 min
```

The tests cover unit conversion and header parsing, and the normalization equations: a
hand-calculated NDP, exact invariance of NPF/nDP for a clean membrane under varying T/P/TDS/flow,
and a recovered 10 % permeability loss. They also cover:

- validation messages and causal cycle baselines;
- **no-leakage checks**: features at day t are identical whether or not future data exists, and
  labels never cross a cleaning;
- days-to-threshold and alert logic;
- model quantile ordering, monotonicity and save/load;
- CLI exit codes and the REST endpoints.

## 8. Known limitations (read before using on a real plant)

* **Trained on synthetic data.** No suitable public long-term RO dataset was found (see
  `docs/DATA_SOURCES.md`). The shipped model `models/v1` is trained on a physics-based simulator
  of a 4-train seawater RO plant. On a real plant, use the baseline and normalization outputs at
  first, and **retrain on that plant's history** (`rofoul train`) before trusting the days-to-clean
  numbers. Inputs outside the training range produce a warning.
* **Unpredictable onsets.** Sudden events such as algal blooms, pretreatment failures and
  antiscalant-dosing failures cannot be predicted from operating data alone. The model reacts
  only after the slope changes. Feed-quality inputs (SDI, turbidity, chlorophyll) would be needed.
* **Sensor faults look like fouling.** A drifting concentrate-pressure transmitter produces a
  false ΔP rise. Stuck sensors and spikes are filtered; slow drift is not.
* **Start of file.** If the file does not start right after a cleaning and has no `cip` column,
  fouling is measured against the first days of the file, so the decline is underestimated (a
  warning is given).
* **Early cycle.** Forecasts in the first 3 days after a cleaning have low confidence because the
  baseline is still forming.
* **One train = one unit.** Normalization is per train and per whole train, not per stage or per
  vessel. Two-stage systems with inter-stage data should be split into stages.
* **Thresholds are generic.** NPF −10 % / nDP +15 % follow manufacturer guidance. Change them in
  the config if your plant has its own criteria.
* **Normalization simplifications.** Osmotic pressure is linear in TDS. Concentration
  polarization is ignored. TCF uses FilmTec constants. These create small systematic errors,
  quantified against the simulator truth in the report.

## 9. Repository layout

```
src/rofoul/        package: validation, units, normalize, cycles, features, models, forecast,
                   pipeline, evaluate, synthetic, cli, api, default.yaml
config/plant.yaml  plant-specific override example (used by docker compose)
models/v1/         shipped model (model.joblib + metadata.json)
data/sample/       small example inputs (SI, US units, deliberately broken)
reports/           back-test metrics, per-origin results, failure analysis
docs/              technical report, data-source log, AI usage log, figures
scripts/           sample generation, figures / failure analysis
tests/             pytest suite
```
