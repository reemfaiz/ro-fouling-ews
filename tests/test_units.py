import numpy as np
import pytest

from rofoul.units import UnitError, canonical_unit, split_header_unit, to_internal


def test_pressure_and_temperature_conversions():
    assert to_internal(np.array([14.5038]), "feed_pressure", "psi")[0] == pytest.approx(1.0, rel=1e-4)
    assert to_internal(np.array([100.0]), "feed_pressure", "kPa")[0] == pytest.approx(1.0)
    assert to_internal(np.array([212.0]), "feed_temp", "degF")[0] == pytest.approx(100.0)
    assert to_internal(np.array([298.15]), "feed_temp", "K")[0] == pytest.approx(25.0)


def test_flow_and_concentration_conversions():
    assert to_internal(np.array([24.0]), "permeate_flow", "m3/d")[0] == pytest.approx(1.0)
    assert to_internal(np.array([1.0]), "permeate_flow", "gpm")[0] == pytest.approx(0.2271247)
    assert to_internal(np.array([35.0]), "feed_tds", "g/L")[0] == pytest.approx(35000.0)


@pytest.mark.parametrize("header,base,unit", [
    ("Feed Pressure (psi)", "Feed Pressure", "psi"),
    ("feed_temp_degF", "feed_temp", "degF"),
    ("Feed Temperature (°C)", "Feed Temperature", "degC"),
    ("Permeate Flow [m³/h]", "Permeate Flow", "m3/h"),
    ("feed_cond_uscm", "feed_cond", "uS/cm"),
    ("train_id", "train_id", None),
])
def test_header_unit_parsing(header, base, unit):
    assert split_header_unit(header) == (base, unit)


def test_unknown_unit_and_wrong_quantity():
    with pytest.raises(UnitError):
        canonical_unit("furlongs")
    with pytest.raises(UnitError):
        to_internal(np.array([1.0]), "feed_pressure", "gpm")
