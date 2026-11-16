"""Unit parsing and conversion to the internal unit system.

Internal units: degC, bar (gauge), m3/h, mg/L, uS/cm.
"""
from __future__ import annotations

import re

import numpy as np

# quantity -> {unit token: (scale, offset)}  such that internal = value*scale + offset
_CONVERSIONS: dict[str, dict[str, tuple[float, float]]] = {
    "temperature": {
        "degC": (1.0, 0.0),
        "degF": (5.0 / 9.0, -32.0 * 5.0 / 9.0),
        "K": (1.0, -273.15),
    },
    "pressure": {
        "bar": (1.0, 0.0),
        "psi": (0.0689475729, 0.0),
        "kPa": (0.01, 0.0),
        "MPa": (10.0, 0.0),
    },
    "flow": {
        "m3/h": (1.0, 0.0),
        "m3/d": (1.0 / 24.0, 0.0),
        "L/s": (3.6, 0.0),
        "L/min": (0.06, 0.0),
        "gpm": (0.227124707, 0.0),
        "MGD": (157.725491, 0.0),
    },
    "concentration": {
        "mg/L": (1.0, 0.0),
        "g/L": (1000.0, 0.0),
    },
    "conductivity": {
        "uS/cm": (1.0, 0.0),
        "mS/cm": (1000.0, 0.0),
    },
    "flag": {"-": (1.0, 0.0)},
}

FIELD_QUANTITY = {
    "feed_temp": "temperature",
    "feed_pressure": "pressure",
    "concentrate_pressure": "pressure",
    "diff_pressure": "pressure",
    "permeate_pressure": "pressure",
    "feed_flow": "flow",
    "permeate_flow": "flow",
    "concentrate_flow": "flow",
    "feed_tds": "concentration",
    "permeate_tds": "concentration",
    "feed_conductivity": "conductivity",
    "permeate_conductivity": "conductivity",
    "cip": "flag",
}

# Free-text unit spellings -> canonical token
_ALIASES = {
    "c": "degC", "degc": "degC", "°c": "degC", "celsius": "degC", "degreesc": "degC", "oc": "degC",
    "f": "degF", "degf": "degF", "°f": "degF", "fahrenheit": "degF", "of": "degF",
    "k": "K", "kelvin": "K",
    "bar": "bar", "barg": "bar", "bar(g)": "bar",
    "psi": "psi", "psig": "psi", "psi(g)": "psi",
    "kpa": "kPa", "kpag": "kPa", "mpa": "MPa",
    "m3/h": "m3/h", "m3h": "m3/h", "m³/h": "m3/h", "m3/hr": "m3/h", "m3hr": "m3/h", "cmh": "m3/h",
    "m3/d": "m3/d", "m3d": "m3/d", "m³/d": "m3/d", "m3/day": "m3/d", "cmd": "m3/d",
    "l/s": "L/s", "ls": "L/s", "lps": "L/s",
    "l/min": "L/min", "lmin": "L/min", "lpm": "L/min",
    "gpm": "gpm", "mgd": "MGD",
    "mg/l": "mg/L", "mgl": "mg/L", "ppm": "mg/L",
    "g/l": "g/L", "gl": "g/L", "ppt": "g/L",
    "us/cm": "uS/cm", "uscm": "uS/cm", "µs/cm": "uS/cm", "μs/cm": "uS/cm", "us": "uS/cm",
    "ms/cm": "mS/cm", "mscm": "mS/cm",
}


class UnitError(ValueError):
    pass


def canonical_unit(token: str | None) -> str | None:
    if token is None:
        return None
    t = str(token).strip()
    if t in _CONVERSIONS.get("temperature", {}) or any(t in d for d in _CONVERSIONS.values()):
        return t
    key = t.lower().replace(" ", "").replace("³", "3")
    key = key.replace("μ", "u").replace("µ", "u")
    if key in _ALIASES:
        return _ALIASES[key]
    raise UnitError(f"Unknown unit '{token}'")


def supported_units(field: str) -> list[str]:
    return list(_CONVERSIONS[FIELD_QUANTITY[field]].keys())


def to_internal(values, field: str, unit: str | None):
    """Convert a numeric array/Series for `field` from `unit` to internal units."""
    quantity = FIELD_QUANTITY.get(field)
    if quantity is None or quantity == "flag":
        return values
    unit = canonical_unit(unit) if unit else None
    table = _CONVERSIONS[quantity]
    if unit is None:
        unit = next(iter(table))
    if unit not in table:
        raise UnitError(
            f"Unit '{unit}' is not valid for {field} ({quantity}); supported: {', '.join(table)}"
        )
    scale, offset = table[unit]
    if isinstance(values, (list, tuple)):
        values = np.asarray(values, dtype=float)
    return values * scale + offset


_HEADER_UNIT_RE = re.compile(r"^(?P<base>.*?)[\s_]*[\(\[](?P<unit>[^\)\]]+)[\)\]]\s*$")


def split_header_unit(header: str) -> tuple[str, str | None]:
    """'Feed Pressure (psi)' -> ('Feed Pressure', 'psi'); 'feed_pressure_bar' -> ('feed_pressure', 'bar').

    Only returns a unit if it is recognised; otherwise the whole header is the base.
    """
    h = str(header).strip()
    m = _HEADER_UNIT_RE.match(h)
    if m:
        try:
            return m.group("base").strip(), canonical_unit(m.group("unit"))
        except UnitError:
            return m.group("base").strip(), None
    # trailing token after the last underscore / space, e.g. feed_temp_degf
    parts = re.split(r"[_\s]+", h)
    if len(parts) > 1:
        try:
            unit = canonical_unit(parts[-1])
            return "_".join(parts[:-1]), unit
        except UnitError:
            pass
    return h, None
