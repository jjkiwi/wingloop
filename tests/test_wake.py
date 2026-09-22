"""Circulation with memory, and the two bugs that writing it produced."""

import numpy as np
import pytest

from wingloop.aero.wake import RISE_CHORDS, WakeMemory, reference_speed
from wingloop.aero.wing import elliptical_wing


def test_the_first_step_passes_the_force_through_unchanged():
    """Nothing to remember yet, so nothing to lag."""
    wake = WakeMemory(chord=0.816)
    lift, drag, path = wake.update(100.0, 50.0, -20.0, 2516.0, 2e-5)
    assert (lift, drag, path) == pytest.approx((100.0, 50.0, -20.0))


def test_a_steady_input_stays_at_its_input():
    wake = WakeMemory(chord=0.816)
    for _ in range(50):
        out = wake.update(100.0, 50.0, -20.0, 2516.0, 2e-5)
    assert out == pytest.approx((100.0, 50.0, -20.0))


def test_it_converges_when_the_speed_changes():
    """Force goes as the square of speed, so halving the speed quarters it.
    The lag should arrive at the new value, not somewhere between."""
    wake = WakeMemory(chord=0.816)
    wake.update(100.0, 50.0, -20.0, 2516.0, 2e-5)
    for _ in range(400):
        out = wake.update(25.0, 12.5, -5.0, 1258.0, 2e-5)
    assert out[0] == pytest.approx(25.0, rel=0.1)


def test_a_stopped_wing_develops_no_new_circulation():
    """The lag runs in travelled distance, not in time, which is what the
    Wagner effect is a statement about. A wing that is not moving is not
    building circulation however long it waits."""
    wake = WakeMemory(chord=0.816)
    assert wake.coefficient(speed=0.0, dt=2e-5) == 0.0
    assert wake.coefficient(speed=2516.0, dt=2e-5) > 0.0
    # And it builds faster the faster the wing goes.
    assert wake.coefficient(2516.0, 2e-5) > wake.coefficient(1258.0, 2e-5)


def test_the_rise_takes_the_stated_number_of_chords():
    """Half the step response should land near the nominal rise distance."""
    wake = WakeMemory(chord=1.0)
    speed, dt = 100.0, 1e-4  # 0.01 chords per step
    wake.update(0.0, 0.0, 0.0, speed, dt)
    steps = int(round(RISE_CHORDS / (speed * dt / wake.chord)))
    for _ in range(steps):
        out = wake.update(100.0, 0.0, 0.0, speed, dt)
    # One rise-distance of travel through a first-order lag is 1 - 1/e.
    assert out[0] / 100.0 == pytest.approx(1.0 - np.exp(-1.0), rel=0.15)


def test_the_lag_is_on_circulation_so_a_reversal_still_makes_no_force():
    """The bug this class is shaped around.

    Lagging the force directly holds the mid-stroke peak through the reversal,
    where the true force is near zero -- a peak-hold rather than a memory, and
    measured, it inflated mean lift 8.6-fold. What persists is circulation;
    force is circulation times speed, so a stopped wing makes no force however
    much circulation it carries.
    """
    wake = WakeMemory(chord=0.816)
    for _ in range(200):
        wake.update(100.0, 50.0, -20.0, 2516.0, 2e-5)
    # Now stop the wing. The force must collapse with the speed.
    lift, _, _ = wake.update(0.0, 0.0, 0.0, 0.0, 2e-5)
    assert abs(lift) < 1.0, lift


def test_the_reference_speed_is_taken_at_the_radius_of_gyration():
    """The tip overstates it and mid-span understates it; the force integral
    is weighted by the second moment, so that is the radius that matters."""
    wing = elliptical_wing(20)
    radius = float(np.sqrt(wing.moment(2) / wing.area))
    assert 0.5 * wing.length < radius < wing.length
    assert reference_speed(wing, 1000.0) == pytest.approx(radius * 1000.0)
    # Body motion adds to the sweep rather than being ignored.
    assert reference_speed(wing, 1000.0, 500.0) > reference_speed(wing, 1000.0)
