"""The connectome's steering command, and the division of labour behind it."""

import os
from pathlib import Path

import numpy as np
import pytest

from wingloop.brain.readout import (
    AMPLITUDE_TYPE,
    CANDIDATES,
    STEERING_TYPE,
    FlightReadout,
    tuning,
)

CURVE = Path(__file__).parent / "dnbe001_tuning.npz"
DATA_ROOT = Path(
    os.environ.get("FLYLOOP_DATA_ROOT", "/home/user/yijieyin/connectome_data_prep")
)
needs_connectome = pytest.mark.skipif(
    not (DATA_ROOT / "data").is_dir(), reason="needs a connectome_data_prep clone"
)


@pytest.fixture(scope="module")
def curve():
    """The measured tuning, checked in so the body tests need no connectome."""
    return dict(np.load(CURVE))


# ------------------------------------------------------------ the lookup


def test_the_command_interpolates_between_measured_bearings(curve):
    r = FlightReadout(bearings=curve["bearings"], command=curve["command"])
    lo, hi = r.asymmetry(0.0), r.asymmetry(30.0)
    mid = r.asymmetry(15.0)
    assert min(lo, hi) <= mid <= max(lo, hi)


def test_beyond_the_measured_field_the_command_is_held_not_extrapolated():
    """The tuning is not linear out there, and inventing a slope would put the
    strongest commands exactly where the measurements stop."""
    r = FlightReadout(bearings=np.array([-90.0, 0.0, 90.0]), command=np.array([-1.0, 0.0, 2.0]))
    assert r.asymmetry(200.0) == pytest.approx(2.0)
    assert r.asymmetry(-200.0) == pytest.approx(-1.0)


def test_gain_scales_the_command():
    r = FlightReadout(
        bearings=np.array([0.0, 90.0]), command=np.array([0.0, 0.1]), gain=3.0
    )
    assert r.asymmetry(90.0) == pytest.approx(0.3)


# -------------------------------------------------- what the wiring says


def test_the_steering_type_has_the_sign_a_fixating_fly_needs(curve):
    """An object to the right must drive the right side harder. Get this
    backwards and the fly flies smoothly away from whatever it is looking at --
    a failure mode this line of work has already paid for once."""
    r = FlightReadout(bearings=curve["bearings"], command=curve["command"])
    assert r.asymmetry(60.0) > 0.0
    assert r.asymmetry(-60.0) < 0.0


def test_the_amplitude_actuator_is_blind(curve):
    """DNg02 is the largest descending input to the wing steering muscles --
    9.07%, more than double the next -- and it carries no bearing information
    at all. That is the division of labour this module rests on, and it is the
    reason the readout is taken from DNbe001 instead."""
    assert AMPLITUDE_TYPE in curve
    assert float(np.abs(curve[AMPLITUDE_TYPE]).max()) == 0.0


def test_the_steering_type_is_the_best_of_the_candidates(curve):
    """Chosen by measurement, not by name. DNbe001 spans 0.19 across the visual
    field against DNa10's 0.13 and DNge107's 0.04."""
    span = float(curve["command"].max() - curve["command"].min())
    for other in CANDIDATES:
        if other in (STEERING_TYPE, AMPLITUDE_TYPE) or other not in curve:
            continue
        assert span > float(curve[other].max() - curve[other].min()), other


@needs_connectome
@pytest.mark.slow
def test_the_curve_still_matches_the_live_connectome():
    """The checked-in fixture is a measurement, so it can go stale."""
    from flyloop.connectome.data_prep import load_dataset

    c = load_dataset(DATA_ROOT, "malecns", matrix="inprop")
    fresh = FlightReadout.measure(c)
    stored = dict(np.load(CURVE))
    assert np.allclose(fresh.bearings, stored["bearings"])
    assert np.allclose(fresh.command, stored["command"], atol=1e-6)


@needs_connectome
@pytest.mark.slow
def test_tuning_reports_every_type_it_was_asked_for():
    from flyloop.connectome.data_prep import load_dataset

    c = load_dataset(DATA_ROOT, "malecns", matrix="inprop")
    out = tuning(c, types=("DNbe001", "DNg02"), bearings=(-30.0, 0.0, 30.0))
    assert set(out) == {"bearing", "DNbe001", "DNg02"}
    assert len(out["bearing"]) == 3
