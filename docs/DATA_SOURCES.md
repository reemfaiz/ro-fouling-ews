# Data-source log

**Search date:** 16 Nov 2026

**What was searched:** web search, Kaggle, Zenodo, Mendeley Data, 4TU, OpenEI/NAWI Water DAMS,
data.gov, and the literature.

**What a usable dataset needs**, for the task as specified:

- **R1** full-scale or long-running pilot RO operation (months to years) with real fouling
- **R2** every variable needed for normalization: feed temperature, feed and concentrate (or ΔP)
  and permeate pressure, permeate and feed (or concentrate) flow, and feed salinity or conductivity
- **R3** several cleaning (CIP) cycles, so that fouling-to-threshold can be learned and validated
- **R4** a licence that allows reuse
- **R5** downloadable

> **How the assessment was done.** The build environment's network is restricted to package
> registries. The datasets below could not be downloaded from it (HTTP requests to openei.org,
> mendeley.com, zenodo.org and kaggle.com were refused), so each one was assessed from its landing
> page, metadata and README. If time allows, the most promising one for a real-data check of the
> normalization (#1) should be downloaded and run through `rofoul normalize` on an unrestricted
> machine.

| # | Source | What it is | R1 | R2 | R3 | Decision | Reason |
|---|---|---|---|---|---|---|---|
| 1 | NAWI Seedling: CCRO online actuator/sensor data (OpenEI submission 6474, data.gov; CC-BY-4.0) | Closed-circuit RO pilot: conventional, open-loop, Bayesian-opt. and MPC experiments | ✗ | ✓ (likely) | ✗ | **Rejected for modelling** (candidate for a normalization sanity check) | CCRO is a semi-batch process whose pressure cycles within minutes. It was built to *avoid* fouling. Runs are short control experiments with no cleaning cycles. |
| 2 | NREL: Experimental wave-desalination data with variable feed and pressure | Lab RO under steady / ramp / sinusoidal pressure | ✗ | ✓ | ✗ | Rejected | Dynamic-pressure lab tests; no long-term fouling. |
| 3 | Mendeley: Performance data of a SWRO from wave-powered desalination (Frost, 2023, doi:10.17632/hws49dsfvc.1) | Lab SWRO membrane, steady-state tests | ✗ | ✓ | ✗ | Rejected | Short lab experiments; no fouling or cleaning history. |
| 4 | Mendeley: MBR–RO oleochemical wastewater plant (540 daily records, 2024–25, doi:10.17632/53g7p9mv4c.1) | Full-scale, long-term | ✓ | ✗ | ? | Rejected | Only flows (industrial water, MBR product, RO product). No pressure, temperature or conductivity, so normalization is impossible. |
| 5 | Zenodo 4630769: UF performance data for SWRO feedwater (Cohen group) | UF pretreatment, short- and long-term | ✓ | ✗ (UF, not RO) | – | Rejected | Wrong process. Could be a future source of a feed-fouling-potential covariate. |
| 6 | NREL/OSTI: Reverse Osmosis Simulation Data (doi:10.7481/2478402) | CFD of spacer-filled feed channels | ✗ | ✗ | ✗ | Rejected | Simulation of channel hydrodynamics, not plant operation. |
| 7 | 4TU: Feedwater pH and ion rejection (BW30HR, doi:10.4121/20160329.v1) | Lab NaCl rejection vs pH | ✗ | ✗ | ✗ | Rejected | Single-condition lab data. |
| 8 | Kaggle (searched "reverse osmosis plant sensor dataset", "RO desalination") | – | – | – | – | Nothing suitable | Only unrelated or simulated IoT datasets turned up. |
| 9 | Chebil et al., *Water* 2024, 16, 1892: 5026 days of a full-scale BWRO plant (Tunisia) | Paper with ASTM D4516-standardized trends | ✓ | ✓ | ✗ (almost no cleaning) | **Used for plausibility only** | Data shown only as figures. Used to sanity-check the normalization approach and ΔP growth rates. |
| 10 | Beyer et al., *Membrane fouling and chemical cleaning in three full-scale RO plants* | Paper | ✓ | – | ✓ | **Used for assumptions** | Shows that CIP leaves an irreversible residue, so normalized ΔP and permeability do not fully recover. That is built into the simulator. |
| 11 | OCWD six-year RO dataset (Roehl, Georgia Water Resources Conf.) | 179 runs, 62 parameters | ✓ | ✓ | ✓ | Not available | Proprietary; not public. |
| 12 | ASTM D4516; Hydranautics TSB107 & RODataXL manual; LG Chem TSB111; DuPont FilmTec guidance | Standards / manufacturer notes | – | – | – | **Used for methods** | Normalization equations and cleaning criteria (clean at NPF −10 %, nDP +15 %, salt passage +5–10 %). |

## Conclusion

No public dataset met R1–R3 together. Long-term full-scale data with every normalization
variable and a cleaning history appears not to be published. Plants treat it as confidential.
As the brief allows, the model is developed and validated on a **physically-based synthetic
dataset**. The generator is `rofoul synth` and the code is `src/rofoul/synthetic.py`.

## Synthetic dataset: assumptions

The simulated plant is a 4-train seawater RO plant on a warm-sea intake (Gulf / Red Sea type),
logged hourly for 3 years (2022–2024), about 101,000 rows. Each train has 10,000 m² of membrane
area and a 150 m³/h design permeate flow (15 L/m²/h).

| aspect | assumption | basis |
|---|---|---|
| Transport | Solution-diffusion: Jw = A·TCF·NDP, Js = B·TCF_s·C_m | standard RO theory |
| Operation | Constant permeate-flow and recovery set-points (135–160 m³/h, 40–44 %) that change every ~1.5 / 4 months; feed pressure compensates; the HP pump is capped at 70 bar, after which flow drops | typical SWRO control |
| Seawater | 21–34 °C seasonal + weather noise; 40.5–43.5 g/L TDS | Gulf / Red Sea ranges |
| Fouling | Self-accelerating biofouling dF/dt = k0·1.08^(T−25)·(J/J0)²·E(t)·(1+8F); k0 differs per train | biofilm growth is temperature- and flux-dependent and accelerates |
| Events | Site-wide algal blooms (1–2 per year, starting Apr–Sep, 1–3 weeks, ×2.5–4.5 fouling rate); train-specific pretreatment upsets (×4–8, mostly spacer fouling); one antiscalant failure (scaling, Train B, Sep 2024) | common SWRO upsets |
| Spacer / ΔP | ΔP = ΔP0·(Qavg/Qavg0)^1.6·μ^0.3·(1+F_sp), with spacer fouling growing alongside biofouling | Schock & Miquel-type friction |
| Operators | Clean when perceived NPF loss is 10.5–15 %, ΔP rise is 15–22 %, or after 100–140 days, with a 0–4 day scheduling delay; CIP lasts 18–40 h | manufacturer guidance + practice |
| CIP effect | Removes 88–97 % of reversible fouling and 75–92 % of spacer fouling; 0.2–0.8 % irreversible loss per CIP; ageing −3 %/yr permeability, +8 %/yr salt permeability; membrane replacement in Train C (Oct 2023) | Beyer et al.; typical membrane life |
| Model mismatch | The simulator uses a different TCF constant (2700 vs 2640/3020), osmotic coefficient (0.00079 vs 0.00077), concentration polarization exp(J/180) and viscosity effects compared with the normalization | so that normalization error is realistic, not zero |
| Data defects | Gaussian sensor noise; 0.05 % spikes; 0.2 % missing cells; ~10 communication gaps per train per year (2–30 h) plus one 6-day outage; a stuck feed-pressure transmitter (Train B, 50 h); a slow concentrate-pressure transmitter drift of up to −0.5 bar over 60 days (Train D, Apr–Jun 2024) | typical SCADA problems |

The generator also writes `ground_truth_daily.csv`, which holds the true permeability, the
fouling states and the event flags. This makes it possible to measure the error of the
normalization itself (the report shows it) and to break the forecast error down by condition.

## Limitations of the synthetic data

- The fouling law is our own choice. A model trained on it learns that law, including the
  dependence on temperature and flux. Real plants have other foulant mixes (colloidal, organic,
  scaling), feed-quality drivers and operator habits. **The results show that the method works
  when the assumptions hold. They do not show how well it will do on a particular real plant.**
  Retraining on real plant history is required.
- Events are random but statistically stationary. Real plants change regime (new pretreatment,
  new membranes, a new intake).
- The simulator has a single stage and no per-vessel effects. Lead elements fouling faster than
  tail elements is not modelled.
- Operators in the simulator follow a simple, consistent cleaning rule. Real cleaning decisions
  are messier, so the "truth" for days-to-clean is noisier in reality.

## Cleaning applied (full synthetic set, `rofoul validate data/synthetic/plant_data.csv`)

| step | effect |
|---|---|
| Offline rows (permeate flow < 5 % of median) | 640 rows removed. The CIP-log flag was kept, with process values set to missing. |
| Stuck-sensor detection (constant ≥ 24 h) | 1 period found (Train B feed pressure, 7–9 Mar 2023); values ignored |
| Spike removal (robust z > 8 vs centred 5-sample median) | 58 values removed |
| Duplicate timestamps | none |
| Physically inconsistent rows | none |
| Gaps > 6 h | 23–29 per train reported; the longest was 145 h (Train A, Oct–Nov 2022) |
| Daily aggregation | medians; days with < 4 h of valid data are set to missing |
| Cleaning cycles | taken from the CIP log, combined with detected NPF jumps (this also catches the membrane replacement) |

Without the CIP log, the automatic detector found **81 of 82** cleanings and replacements to
within ±2 days, with **0 false detections**.
