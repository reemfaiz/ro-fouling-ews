"""Create the small sample input files in data/sample/ (reproducible).

* plant_sample.csv           - one unseen train (different seed), canonical headers/units
* plant_sample_us_units.csv  - same data, US units, conductivity instead of TDS,
                               vendor-style headers, no train column, no cip column
* bad_input.csv              - deliberately broken file to demo validation errors
"""
from pathlib import Path

import numpy as np
import pandas as pd

from rofoul.synthetic import simulate

out = Path(__file__).resolve().parents[1] / "data" / "sample"
out.mkdir(parents=True, exist_ok=True)

data, truth = simulate(start="2025-01-01", days=170, trains=("P1",), seed=7)
data = data[data["timestamp"] < "2025-05-01"]  # stop mid-cycle so the forecast is informative
data.to_csv(out / "plant_sample.csv", index=False)

us = pd.DataFrame({
    "Date/Time": data["timestamp"].dt.strftime("%m/%d/%Y %H:%M"),
    "Feed Temperature (°F)": (data["feed_temp_c"] * 9 / 5 + 32).round(2),
    "Feed Pressure (psi)": (data["feed_pressure_bar"] / 0.0689475729).round(1),
    "Concentrate Pressure (psi)": (data["concentrate_pressure_bar"] / 0.0689475729).round(1),
    "Permeate Pressure (psi)": (data["permeate_pressure_bar"] / 0.0689475729).round(2),
    "Feed Flow (gpm)": (data["feed_flow_m3h"] / 0.227124707).round(1),
    "Permeate Flow (gpm)": (data["permeate_flow_m3h"] / 0.227124707).round(1),
    "Feed Conductivity (uS/cm)": (data["feed_tds_mgl"] / 0.70).round(0),
    "Permeate Conductivity (uS/cm)": (data["permeate_tds_mgl"] / 0.70).round(1),
})
us = us[data["permeate_flow_m3h"].fillna(1) > 1]  # SCADA export without offline rows / cip log
us.to_csv(out / "plant_sample_us_units.csv", index=False)

bad = data.head(24 * 10).drop(columns=["permeate_flow_m3h"]).copy()
bad["feed_temp_c"] = bad["feed_temp_c"] * 9 / 5 + 32          # Fahrenheit under a Celsius header
bad["feed_pressure_bar"] = bad["feed_pressure_bar"].astype(object)
bad.loc[bad.index[5], "feed_pressure_bar"] = "n/a"
bad.to_csv(out / "bad_input.csv", index=False)
print("written:", *sorted(p.name for p in out.iterdir()))
