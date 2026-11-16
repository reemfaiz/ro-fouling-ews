"""Physically-based synthetic SWRO plant data generator.

Why synthetic?  No public dataset combines long-term (multi-year) full-scale RO
operation, all variables needed for normalization (T, pressures, flows, TDS)
and a cleaning log (see docs/DATA_SOURCES.md).  The generator therefore
simulates a 4-train seawater RO plant from first principles so that the full
pipeline can be developed and *validated against a known ground truth*.

Physics (per train, hourly):
  * Solution-diffusion transport:  Jw = A_eff * TCF_w(T) * NDP,  Js = B * TCF_s(T) * C_m
  * Operating mode: constant permeate flow and recovery set-points (changed
    occasionally); feed pressure rises to compensate fouling, capped by the
    high-pressure pump (70 bar) after which permeate flow drops.
  * Concentration polarization CP = exp(Jw / k), k = 180 L/m2/h.
  * Feed-channel pressure drop  DP = DP0 * (Qavg/Qavg0)^1.6 * mu(T)^0.3 * (1 + F_sp)
  * Reversible (bio/organic) fouling, self-accelerating biofilm growth:
        dF/dt = k0 * 1.08^(T-25) * (J/J0)^2 * E(t) * (1 + 8 F)
    E(t) = intake events (algal blooms, site-wide) x pretreatment upsets (train).
  * Spacer (feed-channel) fouling:  dF_sp/dt = ks * 1.08^(T-25) * (J/J0) * E_sp(t) * (1 + 4 F_sp)
  * Conditioning after each cleaning: fast 1.5-3 % permeability drop (tau 1.5 d).
  * Operators clean (CIP) when their perceived NPF loss exceeds 10.5-15 %,
    DP rise exceeds 15-22 % or ~100-140 days elapsed, with 0-4 days scheduling
    delay.  CIP removes 88-97 % of reversible fouling, 75-92 % of spacer
    fouling; 0.2-0.8 % irreversible loss per CIP; membrane ageing 3 %/yr.
  * Deliberate model mismatch vs. the normalization in rofoul.normalize
    (different TCF constant, osmotic coefficient, CP and viscosity effects),
    as in a real plant.
  * Data defects: sensor noise, spikes, stuck transmitter, slow transmitter
    drift, communication gaps (hours to days), missing cells, an antiscalant
    failure (scaling), and a membrane replacement.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

AREA_M2 = 10_000.0
J0 = 15.0                      # design flux, L/m2/h
QAVG0 = (150 / 0.42 + (150 / 0.42 - 150)) / 2
P_MAX = 70.0
K_OSM_TRUE = 0.00079
K_CP = 180.0


def _tcf(T, U):
    return math.exp(U * (1 / 298.15 - 1 / (273.15 + T)))


@dataclass
class Event:
    start: float   # day index
    dur: float
    mult: float


def _site_series(n_days: int, start: pd.Timestamp, rng: np.random.Generator):
    doy = np.array([(start + pd.Timedelta(days=d)).dayofyear for d in range(n_days)])
    anom = np.zeros(n_days)
    tds_anom = np.zeros(n_days)
    for d in range(1, n_days):
        anom[d] = 0.9 * anom[d - 1] + rng.normal(0, 0.4)
        tds_anom[d] = 0.95 * tds_anom[d - 1] + rng.normal(0, 120)
    temp = 27.5 + 6.0 * np.sin(2 * np.pi * (doy - 110) / 365.25) + anom
    tds = 42_000 + 1_200 * np.sin(2 * np.pi * (doy - 200) / 365.25) + tds_anom
    # algal blooms: 1-2 per year, starting in the warm season (Apr-Sep)
    blooms = []
    for year in range(start.year, (start + pd.Timedelta(days=n_days)).year + 1):
        for _ in range(int(rng.integers(1, 3))):
            s0 = (pd.Timestamp(year=year, month=4, day=1) - start).days + rng.uniform(0, 180)
            if 10 <= s0 <= n_days - 10:
                blooms.append(Event(s0, rng.uniform(7, 21), rng.uniform(2.5, 4.5)))
    return temp, tds, blooms


def _bump(day: float, ev: Event) -> float:
    if ev.start <= day < ev.start + ev.dur:
        x = (day - ev.start) / ev.dur
        return 1 + (ev.mult - 1) * math.sin(math.pi * x)
    return 1.0


def _piecewise(n_days: int, mean_dur: float, choices, rng):
    vals = np.empty(n_days)
    d = 0
    while d < n_days:
        L = max(5, int(rng.exponential(mean_dur)))
        vals[d:d + L] = rng.choice(choices)
        d += L
    return vals


def simulate(start: str = "2022-01-01", days: int = 1096, trains=("A", "B", "C", "D"), seed: int = 42,
             include_cip_column: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (hourly SCADA-like data, daily ground truth)."""
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp(start)
    temp_d, tds_d, blooms = _site_series(days, t0, rng)
    hours = days * 24
    rows, truth = [], []
    special = {  # deterministic scenario events (day offsets from start)
        "replacement": ("C", 640),
        "scaling": ("B", min(days - 40, 980)),
        "stuck": ("B", 430),
        "drift": ("D", min(days - 120, 830)),
        "long_outage": ("A", 300),
    }
    for ti, tid in enumerate(trains):
        r = np.random.default_rng(seed * 100 + ti)
        A0 = r.uniform(1.05, 1.20)
        B0 = 0.03
        k0 = r.uniform(0.0009, 0.0015)
        ks = k0 * r.uniform(0.8, 1.2)
        dp0 = r.uniform(1.3, 1.7)
        qp_set = _piecewise(days, 45, [135, 145, 150, 155, 160], r)
        rec_set = _piecewise(days, 120, [0.40, 0.42, 0.44], r)
        upsets = [Event(r.uniform(20, days - 20), r.uniform(2, 5), r.uniform(4, 8)) for _ in range(r.poisson(days / 550) + 1)]
        outages = [(r.uniform(0, hours), r.uniform(2, 30)) for _ in range(int(10 * days / 365))]
        if special["long_outage"][0] == tid:
            outages.append((special["long_outage"][1] * 24, 6 * 24))
        # state
        A_irr, B, F, Fsp, Fsc = 1.0, B0, 0.0, 0.0, 0.0
        cond_amp = r.uniform(0.015, 0.03)
        cycle_start_h = 0.0
        A_base = None
        thr_n, thr_d, sched = r.uniform(10.5, 15), r.uniform(15, 22), r.uniform(100, 140)
        cip_due_h, cip_until_h = None, -1.0
        cip_hours: list[tuple[float, float]] = []
        fsp_after_clean = 0.0
        day_acc = {}
        for h in range(hours):
            day = h / 24.0
            d = int(day)
            hod = h % 24
            T = temp_d[d] + 0.3 * math.sin(2 * math.pi * (hod - 9) / 24)
            Cf = tds_d[d] + r.normal(0, 60)
            E_bloom = 1.0
            for ev in blooms:
                E_bloom *= _bump(day, ev)
            E_up = 1.0
            for ev in upsets:
                E_up *= _bump(day, ev)
            sc_tid, sc_day = special["scaling"]
            scaling = tid == sc_tid and sc_day <= day < sc_day + 8
            rp_tid, rp_day = special["replacement"]
            replacing = tid == rp_tid and rp_day <= day < rp_day + 3
            if replacing and abs(day - rp_day) < 1 / 48:
                A_irr, B, F, Fsp, Fsc = 1.04, B0, 0.0, 0.0, 0.0
                cycle_start_h, A_base, fsp_after_clean = h + 72, None, 0.0
            offline = h < cip_until_h or replacing
            # ---- ageing (continuous)
            A_irr *= math.exp(-0.03 / 365 / 24)
            B *= math.exp(0.08 / 365 / 24)
            s_days = max(0.0, (h - cycle_start_h) / 24)
            F_cond = cond_amp * (1 - math.exp(-s_days / 1.5))
            A_eff = A0 * A_irr * (1 - F) * (1 - F_cond) * (1 - Fsc)
            if not offline:
                Qp = qp_set[d]
                R = rec_set[d]
                Jw = Qp * 1000 / AREA_M2
                tw, ts = _tcf(T, 2700), _tcf(T, 3500)
                mu = math.exp(-0.0205 * (T - 25))
                pump_limited = False
                for _ in range(4):
                    Qf = Qp / R
                    Qc = Qf - Qp
                    Qavg = (Qf + Qc) / 2
                    dP = dp0 * (Qavg / QAVG0) ** 1.6 * mu ** 0.3 * (1 + Fsp)
                    C_fc = Cf * math.log(1 / (1 - R)) / R
                    CP = math.exp(Jw / K_CP)
                    Cp = B * (1.15 if scaling else 1.0) * ts * C_fc * CP / Jw
                    pi_m = K_OSM_TRUE * C_fc * CP * (T + 273.15) / 298.15
                    pi_p = K_OSM_TRUE * Cp * (T + 273.15) / 298.15
                    Pp = 1.2
                    Pf = Jw / (A_eff * tw) + pi_m - pi_p + dP / 2 + Pp
                    if Pf <= P_MAX:
                        break
                    pump_limited = True
                    Jw = max(1.0, A_eff * tw * (P_MAX - dP / 2 - Pp - (pi_m - pi_p)))
                    Qp = Jw * AREA_M2 / 1000
                    Pf = P_MAX
                # ---- fouling growth (only while running)
                dt = 1 / 24
                theta = 1.08 ** (T - 25)
                F += k0 * theta * (Jw / J0) ** 2 * E_bloom * E_up * (1 + 8 * F) * dt
                Fsp += ks * theta * (Jw / J0) * (E_bloom * E_up ** 2) * (1 + 4 * Fsp) * dt
                if scaling:
                    Fsc += 0.006 * dt
                F, Fsp, Fsc = min(F, 0.6), min(Fsp, 2.0), min(Fsc, 0.4)
                # ---- measurements
                in_outage = any(o0 <= h < o0 + od for o0, od in outages)
                if not in_outage:
                    meas = {
                        "timestamp": t0 + pd.Timedelta(hours=h),
                        "train_id": tid,
                        "feed_temp_c": T + r.normal(0, 0.1),
                        "feed_pressure_bar": Pf + r.normal(0, 0.10),
                        "concentrate_pressure_bar": Pf - dP + r.normal(0, 0.10),
                        "permeate_pressure_bar": Pp + r.normal(0, 0.03),
                        "feed_flow_m3h": Qf * (1 + r.normal(0, 0.007)),
                        "permeate_flow_m3h": Qp * (1 + r.normal(0, 0.007)),
                        "feed_tds_mgl": Cf * (1 + r.normal(0, 0.01)),
                        "permeate_tds_mgl": Cp * (1 + r.normal(0, 0.03)),
                        "cip": 0,
                    }
                    rows.append(meas)
                acc = day_acc.setdefault(d, {"pump_limited_h": 0, "running_h": 0})
                acc["pump_limited_h"] += int(pump_limited)
                acc["running_h"] += 1
            else:
                if include_cip_column and h < cip_until_h and hod % 6 == 0:
                    rows.append({"timestamp": t0 + pd.Timedelta(hours=h), "train_id": tid, "feed_temp_c": T,
                                 "feed_pressure_bar": 0.5, "concentrate_pressure_bar": 0.3,
                                 "permeate_pressure_bar": 0.0, "feed_flow_m3h": 0.0, "permeate_flow_m3h": 0.0,
                                 "feed_tds_mgl": np.nan, "permeate_tds_mgl": np.nan, "cip": 1})
            # ---- operator CIP policy (checked daily at 08:00)
            if hod == 8 and not offline:
                if s_days >= 2 and A_base is None:
                    A_base = A_eff
                base = A_base or A_eff
                perceived_n = 100 * (1 - A_eff / base) + r.normal(0, 0.8)
                perceived_d = 100 * ((1 + Fsp) / (1 + fsp_after_clean) - 1) + r.normal(0, 1.0)
                if cip_due_h is None and (perceived_n > thr_n or perceived_d > thr_d or s_days > sched):
                    cip_due_h = h + 24 * int(r.integers(0, 5))
            if cip_due_h is not None and h >= cip_due_h and not offline:
                dur = r.uniform(18, 40)
                cip_until_h = h + dur
                cip_hours.append((h, dur))
                F *= 1 - r.uniform(0.88, 0.97)
                Fsp *= 1 - r.uniform(0.75, 0.92)
                fsp_after_clean = Fsp
                Fsc *= 0.2
                A_irr *= 1 - r.uniform(0.002, 0.008)
                B *= 1.01
                cycle_start_h = cip_until_h
                A_base = None
                cip_due_h = None
                cond_amp = r.uniform(0.015, 0.03)
                thr_n, thr_d, sched = r.uniform(10.5, 15), r.uniform(15, 22), r.uniform(100, 140)
            if hod == 23:
                acc = day_acc.setdefault(d, {"pump_limited_h": 0, "running_h": 0})
                truth.append({
                    "date": (t0 + pd.Timedelta(days=d)).normalize(), "train_id": tid,
                    "true_A_rel": A_eff / A0, "F_rev": F, "F_cond": F_cond, "F_scale": Fsc, "F_spacer": Fsp,
                    "bloom_mult": E_bloom, "upset_mult": E_up, "scaling": scaling, "replacement": replacing,
                    "cip": any(c0 <= h and h - 24 < c0 + cd for c0, cd in cip_hours),
                    "pump_limited_h": acc["pump_limited_h"], "running_h": acc["running_h"],
                    "temp_c": temp_d[d],
                })
    data = pd.DataFrame(rows).sort_values(["train_id", "timestamp"]).reset_index(drop=True)
    data = _inject_defects(data, special, t0, rng)
    if not include_cip_column:
        data = data.drop(columns="cip")
    return data, pd.DataFrame(truth)


def _inject_defects(data: pd.DataFrame, special: dict, t0: pd.Timestamp, rng: np.random.Generator) -> pd.DataFrame:
    data = data.copy()
    num = ["feed_temp_c", "feed_pressure_bar", "concentrate_pressure_bar", "permeate_pressure_bar",
           "feed_flow_m3h", "permeate_flow_m3h", "feed_tds_mgl", "permeate_tds_mgl"]
    running = data["permeate_flow_m3h"] > 1
    # stuck feed-pressure transmitter (50 h)
    tid, d0 = special["stuck"]
    m = (data["train_id"] == tid) & (data["timestamp"] >= t0 + pd.Timedelta(days=d0)) & \
        (data["timestamp"] < t0 + pd.Timedelta(days=d0, hours=50)) & running
    if m.any():
        data.loc[m, "feed_pressure_bar"] = round(float(data.loc[m, "feed_pressure_bar"].iloc[0]), 2)
    # concentrate-pressure transmitter drift: reads up to 0.5 bar low over 60 days, then recalibrated
    tid, d0 = special["drift"]
    ts = data["timestamp"]
    m = (data["train_id"] == tid) & (ts >= t0 + pd.Timedelta(days=d0)) & (ts < t0 + pd.Timedelta(days=d0 + 60)) & running
    frac = ((ts[m] - (t0 + pd.Timedelta(days=d0))).dt.total_seconds() / (60 * 86400)).to_numpy()
    data.loc[m, "concentrate_pressure_bar"] -= 0.5 * frac
    # spikes
    n = len(data)
    idx = rng.choice(n, size=int(0.0005 * n), replace=False)
    for i in idx:
        c = num[rng.integers(0, len(num))]
        if pd.notna(data.at[i, c]) and running.iloc[i]:
            data.at[i, c] *= rng.choice([0.6, 1.4])
    # missing cells
    for c in num:
        mm = rng.random(n) < 0.002
        data.loc[mm & running, c] = np.nan
    data[num] = data[num].round(3)
    return data


def truth_with_indicators(truth: pd.DataFrame) -> pd.DataFrame:
    """Ground-truth fouling indicators relative to each true post-clean baseline (for evaluation)."""
    out = []
    for tid, g in truth.groupby("train_id"):
        g = g.sort_values("date").copy()
        cyc = (g["cip"] | g["replacement"]).astype(int)
        start = cyc.diff().fillna(0) < 0  # first day after a cleaning/replacement block
        start.iloc[0] = True
        g["true_cycle"] = start.cumsum()
        base = g.groupby("true_cycle")["true_A_rel"].transform(lambda s: s.iloc[:3].median())
        g["true_npf_decline_pct"] = 100 * (1 - g["true_A_rel"] / base)
        sp = g.groupby("true_cycle")["F_spacer"].transform(lambda s: s.iloc[:3].median())
        g["true_dp_rise_pct"] = 100 * ((1 + g["F_spacer"]) / (1 + sp) - 1)
        out.append(g)
    return pd.concat(out, ignore_index=True)
