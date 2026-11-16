# AI usage log

## Tools used

| tool | used for |
|---|---|
| **Claude (Anthropic), Opus 5.5 model, agent mode in the Claude app.** It had web search and a sandboxed Linux workspace (Python, git, pytest). | Searching for and screening public datasets and literature. Drafting the code (package, synthetic generator, back-test, CLI, API, Dockerfile), the tests, the README, this log and the technical report. Running the tests and back-tests in the sandbox. |
| Web search (through the assistant) | Finding RO datasets on Mendeley, Zenodo, OpenEI/NAWI, Kaggle and data.gov; ASTM D4516 and manufacturer notes (TSB107, TSB111, FilmTec) on normalization and cleaning criteria. |

> Section to complete by the candidate: list any other tools (IDE assistant, ChatGPT, etc.) and
> what you reviewed, changed or re-ran yourself.

## Most important prompts

1. **Initial task prompt** (original in Arabic, translated): *"I have a task at work: I've been
   asked to build a system. I'll send you the details. Do it in the best way possible, and do it
   in English."* The full brief was attached: early-warning model for RO membrane fouling,
   normalization, forecasting and days-to-clean, temporal validation against a baseline, an
   honest failure analysis, packaging (Docker/package, REST/CLI, validation, tests, versioned
   model and config, README) and this log.
2. *(to add: the main follow-up prompts used while reviewing and changing the work)*
3. *(to add)*

## Where the AI output was wrong or needed correction, and how it was caught

All of these happened during this build and are fixed in the code.

| # | Problem in the AI-generated work | How it was caught | Fix |
|---|---|---|---|
| 1 | Daily aggregation used `min(min_coverage, sampling_step)`. For hourly data this accepted days with only 1 h of data. | Code review of `to_daily` | Compare coverage hours with the configured minimum; regression test `test_daily_aggregation_coverage` |
| 2 | The offline-row filter also deleted the rows carrying the CIP log (the train is offline during cleaning), so cleanings logged in the file were lost. | The detected cycle count did not match the simulator's ground truth | Keep CIP rows, with process values set to missing; cycles now match the truth exactly |
| 3 | The first spike filter used a trailing median. It would have deleted the real step change in pressure after every cleaning or set-point change. | Design review: a trailing window cannot tell a spike from a level shift | Centred 5-sample median; only single readings are removed |
| 4 | The stuck-sensor rule flagged 3 identical readings, which would wrongly reject rounded daily data. | Thinking about daily-resolution inputs; a unit test with constant synthetic values tripped it | Require ≥ 24 h **and** ≥ 6 samples |
| 5 | First back-test: days-to-clean was biased by about +8 days, and the p10–p90 band covered only 57–66 % instead of 80 %. | Back-test metrics, checked against the observed threshold crossings | Diagnosed as (a) a first-passage effect: a noisy indicator crosses the threshold before the median path does; (b) over-confident quantiles. Added a decision quantile and split-conformal band widening, **both calibrated only on the last 25 % of each training window**, never on test folds. Final back-test: bias +1.7 days; band coverage 0.72–0.74, which is still below the nominal 0.80 and is reported as a limitation. |
| 5b | Raw `NPF / true permeability` varied by about 9 % between hot and cold days. | Comparison with the simulator truth | Not a code bug. Concentration polarization is ignored in standard normalization, and in SWRO NDP is a small difference of large pressures. Added an optional CP correction (off by default); the effect is quantified in the report. |
| 5c | Without a CIP log, the post-clean baseline included the partly-cleaned day, so the same data in two unit systems gave declines 0.9 pp apart. | Test comparing the SI file with the US-units file | The baseline uses only full-coverage days, with a causal fall-back |
| 6 | The synthetic generator's operator-cleaning logic contained leftover placeholder code (an undefined variable inside a dead branch). | Reading the generated code before running it | Rewrote the perceived-ΔP rule |
| 7 | JSON export crashed on pandas-3 string columns (`np.issubdtype` on `StringDtype`). | Unit tests (`test_pipeline_end_to_end`, `test_api_predict`) | Use `pd.api.types.is_datetime64_any_dtype` |
| 8 | Normalized ΔP ignored the temperature (viscosity) effect, giving a 3–4 percentage-point error against the simulator truth. | Comparison with the ground truth | Added a viscosity correction, (μ25/μT)^0.3, with the exponent from the Schock & Miquel spacer friction factor. It was **not** tuned to the simulator. |
| 9 | Test fixtures with perfectly constant values were rejected by the stuck-sensor check. | Failing tests | Fixtures add realistic measurement noise |
| 10 | The assistant could not open one paper PDF (the repository returned a login page), and the sandbox could not download any dataset. | Tool errors | Used the publisher's HTML page; assessed datasets from metadata only, and said so in `DATA_SOURCES.md` |
| 11 | **The Docker image could not be built in the AI sandbox** because Docker Hub is blocked there. | Build error | The `pip install` route was verified in a fresh virtual environment. **The Docker route must be checked on a machine with internet access before the live session.** |

## How the outputs were verified

- **About 50 unit and integration tests**, all passing. They include physics checks: a
  hand-calculated NDP; NPF and nDP constant to 1e-9 for a clean membrane under varying T, P, TDS
  and flow; a 10 % permeability loss recovered exactly. Leakage checks: features at day t are the
  same with or without future data, and labels never cross a cleaning. API and CLI behaviour on
  good and broken files is also tested.
- **Ground truth.** Normalized NPF decline was compared with the simulator's true permeability
  loss, and cleaning detection with the true cleaning dates.
- **Rolling-origin back-test** with purged training windows, against two baselines. A second run
  holds out a whole train.
- **Fresh-environment install.** The repository was cloned to a new directory, installed with
  `pip install .` into a new virtual environment, and the commands in the README were run.

## Judgement calls the AI made that a reviewer should challenge

- Choosing synthetic data over a partial real dataset (#1 or #4 in `DATA_SOURCES.md`). The model
  therefore learns the simulator's fouling law. This is the biggest limitation.
- Using a single global decision quantile (0.9, at the top of the search grid) for both
  indicators.
- Cleaning thresholds of NPF −10 % and nDP +15 % (generic manufacturer guidance), with no
  salt-passage trigger because salt passage is not forecast.
