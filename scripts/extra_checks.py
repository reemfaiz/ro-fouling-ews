"""Extra checks quoted in the report: cleaning detection without a CIP log, number of tests.

    python scripts/extra_checks.py   -> reports/extra_checks.json
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

from rofoul.config import load_config
from rofoul.pipeline import prepare_daily

ROOT = Path(__file__).resolve().parents[1]

cfg = load_config()
data = pd.read_csv(ROOT / "data/synthetic/plant_data.csv")
data = data[data["cip"] == 0].drop(columns="cip")          # remove the cleaning log entirely
daily, _, _ = prepare_daily(data, cfg)
truth = pd.read_csv(ROOT / "data/synthetic/ground_truth_daily.csv", parse_dates=["date"])
for c in ("cip", "replacement"):
    truth[c] = truth[c].astype(str).str.lower().isin(["true", "1"])
tp = fp = fn = 0
for tid, g in daily.groupby("train_id"):
    det = g.loc[g["cycle_start"].astype(bool), "date"].iloc[1:]       # skip the data start
    t = truth[truth.train_id == tid].sort_values("date")
    blk = (t["cip"] | t["replacement"]).astype(int)
    true_starts = t.loc[blk.diff() == -1, "date"]
    for s in true_starts:
        hit = (abs((det - s).dt.days) <= 2).any()
        tp, fn = tp + hit, fn + (not hit)
    for s in det:
        fp += not (abs((true_starts - s).dt.days) <= 2).any()

out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", str(ROOT / "tests")],
                     capture_output=True, text=True).stdout
counts = [int(x) for x in re.findall(r"^\S+\.py: (\d+)$", out, flags=re.M)]
res = {"cip_detection": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tolerance_days": 2},
       "n_tests": sum(counts) if counts else None}
(ROOT / "reports/extra_checks.json").write_text(json.dumps(res, indent=2))
print(res)
