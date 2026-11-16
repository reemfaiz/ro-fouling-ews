"""Fill docs/report_template.md with the numbers in reports/*.json and build the PDF.

    python scripts/build_report.py            # -> docs/TECHNICAL_REPORT.md + .pdf (needs pandoc + xelatex)
"""
from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    m = json.loads((ROOT / "reports/backtest_metrics.json").read_text())
    hd = json.loads((ROOT / "reports/backtest_metrics_holdout_D.json").read_text())
    fa = json.loads((ROOT / "reports/failure_analysis.json").read_text())
    extra = json.loads((ROOT / "reports/extra_checks.json").read_text())
    f = m["forecast_mae"]
    n, d = f["npf_decline_pct"], f["dp_rise_pct"]
    dc, al = m["days_to_clean"], m["alert_red"]
    ne = fa["normalization_vs_ground_truth"]
    fe = fa["forecast_error_by_condition"]
    dd = fe["days_to_clean_by_condition"]
    vals = dict(
        date=dt.date.today().strftime("%d %b %Y"),
        lin_mae=dc["linear_trend"]["mae_days"], gbm_mae=dc["gbm"]["mae_days"],
        lin_med=dc["linear_trend"]["median_abs_err_days"], gbm_med=dc["gbm"]["median_abs_err_days"],
        lin_bias=dc["linear_trend"]["bias_days"], gbm_bias=dc["gbm"]["bias_days"],
        n_days=dc["gbm"]["n_origins"],
        lin_h30=n["linear_trend"]["h30"], gbm_h30=n["gbm"]["h30"],
        lin_p=al["linear_trend"]["precision"], lin_r=al["linear_trend"]["recall"],
        gbm_p=al["gbm"]["precision"], gbm_r=al["gbm"]["recall"],
        cov_n=m["interval_coverage_p10_p90"]["npf_decline_pct"], cov_d=m["interval_coverage_p10_p90"]["dp_rise_pct"],
        hd_lin=hd["days_to_clean"]["linear_trend"]["mae_days"], hd_gbm=hd["days_to_clean"]["gbm"]["mae_days"],
        folds=", ".join(f"{a[:7]}→{b[:7]}" for a, b in m["folds"]),
        corr_raw=ne["corr_raw_flow_vs_true_permeability"], corr_npf=ne["corr_npf_vs_true_permeability"],
        norm_mae=ne["npf_decline_mae_pp"], norm_mae_cp=ne["with_cp_correction_k150"]["npf_decline_mae_pp"],
        artefact=round(100 * (1 - ne["npf_ratio_hot_over_cold_default"]), 1),
        artefact_cp=round(100 * (1 - ne["with_cp_correction_k150"]["npf_ratio_hot_over_cold"]), 1),
        ev_gbm=dd["event_before_crossing"]["gbm_mae_days"], ev_bias=dd["event_before_crossing"]["gbm_bias_days"],
        ev_lin=dd["event_before_crossing"]["linear_trend_mae_days"], ev_n=dd["event_before_crossing"]["n"],
        noev_gbm=dd["no_event"]["gbm_mae_days"],
        dp_norm=fe["no_event"]["gbm_mae_dp_h14"], dp_drift=fe["transmitter_drift"]["gbm_mae_dp_h14"],
        cip_tp=extra["cip_detection"]["tp"], cip_n=extra["cip_detection"]["tp"] + extra["cip_detection"]["fn"],
        n_tests=extra["n_tests"],
    )
    for model, short in (("persistence", "p"), ("linear_trend", "l"), ("gbm", "g")):
        for h in (7, 14, 30):
            vals[f"{short}_n{h}"] = n[model][f"h{h}"]
            vals[f"{short}_d{h}"] = d[model][f"h{h}"]
    text = (ROOT / "docs/report_template.md").read_text().format(**vals)
    out_md = ROOT / "docs/TECHNICAL_REPORT.md"
    out_md.write_text(text)
    cmd = ["pandoc", str(out_md), "-o", str(ROOT / "docs/TECHNICAL_REPORT.pdf"), "--pdf-engine=xelatex",
           f"--resource-path={ROOT / 'docs'}", "-H", str(ROOT / "docs/report_header.tex")]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        print(r.stderr, file=sys.stderr)
        sys.exit(r.returncode)
    print("written", out_md, "and TECHNICAL_REPORT.pdf")


if __name__ == "__main__":
    main()
