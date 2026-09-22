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


# ------------------------------------------------------ the throttle channel

POWER_CURVE = Path(__file__).parent / "power_command.npz"

#: Every family measured for the fixture, so the selectivity claim can be
#: checked against families that do not have it without loading a connectome.
power_families = dict(np.load(POWER_CURVE))


@pytest.fixture(scope="module")
def power():
    from wingloop.brain.readout import PowerReadout

    stored = dict(np.load(POWER_CURVE))
    return PowerReadout(
        command=stored["command"],
        activation=stored["activation"],
        steering=stored["steering"],
    )


def test_the_command_channel_is_a_throttle_not_a_second_steering(power):
    """What makes DNa08 and DNp31 the right neurons to take drive from.

    Driven together they raise the power motor neurons to 0.374 and leave the
    whole 32-neuron steering apparatus at 0.019. Anything that moved both
    would be a different kind of signal, and one family does: see the
    dissociation test below for the control that gives this number its
    meaning.

    The pool this is measured against is the whole of it. Recording four
    steering types instead of all sixteen returned 82x for the same command,
    and two of the families below came out infinitely selective, which is what
    a number looks like when it is measured against the muscles a command
    happens to miss.
    """
    assert 15.0 < power.separation < 30.0, power.separation


def test_a_command_that_is_not_a_throttle_and_one_that_is_only_steering(power):
    """The control that makes the separation above mean something.

    If every descending family came out power-selective, 82x would be a fact
    about the motor pools rather than about this command. It does not:
    DNbe001 moves both pools about equally (1.2x) and DNa02 moves only the
    steering pool. So the three regimes are real and the command sits in one
    of them.
    """
    families = list(power_families["families"])
    pw = power_families["family_power"]
    st = power_families["family_steering"]

    i = families.index("DNbe001")
    assert pw[i] / st[i] < 3.0, "the steering command is no kind of throttle"

    i = families.index("DNa02")
    assert pw[i] == 0.0 and st[i] > 0.0, "and this one reaches steering only"

    for name in ("DNa08", "DNp31"):
        i = families.index(name)
        assert pw[i] / st[i] > 15.0, f"{name} is meant to be the selective one"

    # DNg02 is the largest drive of all and the least selective of the three,
    # which is why the throttle is not built on it: amplitude control is a
    # different job. Stated as an ordering so it cannot quietly invert.
    g, a = families.index("DNg02"), families.index("DNa08")
    assert pw[g] > pw[a] and pw[g] / st[g] < pw[a] / st[a]


def test_drive_rises_with_the_command_and_vanishes_without_it(power):
    assert power.drive(0.0) == pytest.approx(0.0, abs=1e-9)
    levels = [power.drive(c) for c in (0.25, 0.5, 0.75, 1.0)]
    assert all(b > a for a, b in zip(levels[:-1], levels[1:], strict=True))
    # Calibrated so a full command reaches the drive that gives a hovering
    # stroke; see DRIVE_PER_ACTIVATION on why that is a calibration.
    assert 2.5 < power.drive(1.0) < 3.5


def test_the_readout_carries_no_bearing(power):
    """The throttle is a scalar, and that is a choice about the model.

    It was nearly written down as a fact about the wiring -- the power motor
    neurons take 0.00% of their input from visual neurons -- until the same
    measurement on the steering motor neurons returned 0.00% too. Direct
    blindness separates nothing. What the readout has is no bearing input,
    which is a property of this class, so that is all this asserts.
    """
    from wingloop.brain.readout import PowerReadout

    assert "bearing" not in PowerReadout.__dataclass_fields__
    assert power.drive(0.6) == power.drive(0.6)


def test_neither_motor_pool_sees_anything_directly(power):
    """Why the blindness claim was withdrawn, pinned so it cannot come back."""
    assert float(power_families["power_mn_visual"]) == 0.0
    assert float(power_families["steering_mn_visual"]) == 0.0


def test_the_throttle_is_reachable_by_vision_one_synapse_up(power):
    """And the opposite of what was first written: DNp31 is the visual one.

    Half of this command draws 29.5% of its input from visual neurons -- more
    than DNbe001's 20.4%, which is the steering command -- while the other
    half, DNa08, draws 0.7%. So vision can open this throttle. That was not
    designed in and the scalar command does not use it; it is recorded because
    the channel was first written up as one nothing could see.
    """
    families = list(power_families["families"])
    vis = power_families["family_visual"]
    assert vis[families.index("DNp31")] > vis[families.index("DNbe001")] > 0.15
    assert vis[families.index("DNa08")] < 0.02, "the other half of it is not"


@needs_connectome
@pytest.mark.slow
def test_the_power_muscles_get_no_visual_input_at_all(malecns_or_load):
    """The measurement the blindness claim rests on, against the live data."""
    import numpy as _np

    c = malecns_or_load
    types = c.neurons["type"].astype(str)
    classes = c.neurons["super_class"].astype(str)
    from wingloop.brain.readout import POWER_TYPES

    power_rows = _np.flatnonzero(types.isin(POWER_TYPES).to_numpy())
    visual = _np.flatnonzero(classes.str.contains("visual|ol_").to_numpy())
    assert len(power_rows) == 24, len(power_rows)

    matrix = abs(c.W.tocsc())
    total = _np.asarray(matrix[:, power_rows].sum(axis=0)).ravel()
    from_visual = _np.asarray(matrix[visual][:, power_rows].sum(axis=0)).ravel()
    assert float(from_visual.max()) == 0.0
    assert float(total.min()) > 0.0, "they do receive input, just not from vision"


@needs_connectome
@pytest.mark.slow
def test_the_stored_power_curve_still_matches_the_connectome(malecns_or_load):
    from wingloop.brain.readout import PowerReadout

    fresh = PowerReadout.measure(malecns_or_load)
    stored = dict(np.load(POWER_CURVE))
    assert np.allclose(fresh.command, stored["command"])
    assert np.allclose(fresh.activation, stored["activation"], atol=1e-6)


@pytest.fixture(scope="module")
def malecns_or_load():
    from flyloop.connectome.data_prep import load_dataset

    return load_dataset(DATA_ROOT, "malecns", matrix="inprop")
