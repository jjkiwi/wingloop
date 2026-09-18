"""The force model, and the one test that catches a unit error."""

from pathlib import Path

import numpy as np
import pytest

from wingloop.aero.blade_element import (
    StrokeState,
    blade_element_forces,
    drag_coefficient,
    hover_check,
    lift_coefficient,
    stroke_average_lift,
)
from wingloop.aero.wing import AIR_DENSITY, Wing, elliptical_wing, wing_from_mesh

#: Mass of the whole NeuroMechFly model, grams, read off the built model.
BODY_MASS = 1.0265e-3
MESH = Path(__file__).parent / "rwing_vertices.npy"


# ------------------------------------------------------------- coefficients


def test_lift_peaks_far_past_where_an_aerofoil_stalls():
    """The leading-edge vortex stays attached at Re ~ 150, so the coefficient
    keeps climbing to about 45 degrees. A model that stalls at 15 is modelling
    an aeroplane and will not hold a fly up."""
    a = np.deg2rad(np.arange(0, 91))
    cl = lift_coefficient(a)
    peak = np.rad2deg(a[int(np.argmax(cl))])
    assert 35 < peak < 55, peak
    assert cl.max() > 1.5


def test_drag_is_least_edge_on_and_greatest_broadside():
    assert drag_coefficient(0.0) < drag_coefficient(np.deg2rad(90))
    assert drag_coefficient(np.deg2rad(90)) > lift_coefficient(np.deg2rad(90))


# -------------------------------------------------------------- geometry


def test_area_and_moments_are_the_integrals_they_claim():
    w = Wing(stations=np.array([0.5, 1.5]), chords=np.array([1.0, 1.0]), length=2.0)
    assert w.dr == pytest.approx(1.0)
    assert w.area == pytest.approx(2.0)
    assert w.mean_chord == pytest.approx(1.0)
    assert w.moment(1) == pytest.approx(0.5 + 1.5)
    assert w.moment(2) == pytest.approx(0.25 + 2.25)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"stations": np.array([0.5]), "chords": np.array([1.0])},
        {"stations": np.array([0.5, 1.5]), "chords": np.array([1.0, -1.0])},
        {"stations": np.array([1.5, 0.5]), "chords": np.array([1.0, 1.0])},
    ],
)
def test_impossible_wings_are_refused(kwargs):
    with pytest.raises(ValueError):
        Wing(**kwargs)


# ----------------------------------------------------------------- forces


def test_force_goes_as_the_square_of_flapping_rate():
    """Dynamic pressure is quadratic in velocity, so twice the stroke rate is
    four times the force. If this is linear the integration is wrong."""
    w = elliptical_wing(20)
    s = StrokeState(alpha=np.deg2rad(45), phi_dot=100.0)
    f1 = blade_element_forces(w, s, include_rotational=False, include_added_mass=False)
    s2 = StrokeState(alpha=np.deg2rad(45), phi_dot=200.0)
    f2 = blade_element_forces(w, s2, include_rotational=False, include_added_mass=False)
    assert f2.lift / f1.lift == pytest.approx(4.0, rel=1e-6)


def test_a_motionless_wing_makes_no_force():
    w = elliptical_wing(20)
    f = blade_element_forces(w, StrokeState(alpha=np.deg2rad(45)))
    assert f.lift == pytest.approx(0.0)
    assert f.drag == pytest.approx(0.0)


def test_the_rotational_term_vanishes_without_rotation_and_appears_with_it():
    """The Kramer force is what makes rotation timing steer the animal, so it
    must be zero mid-stroke and non-zero at reversal, not a constant offset."""
    w = elliptical_wing(20)
    mid = StrokeState(alpha=np.deg2rad(45), phi_dot=1000.0, alpha_dot=0.0)
    rev = StrokeState(alpha=np.deg2rad(45), phi_dot=1000.0, alpha_dot=500.0)
    assert blade_element_forces(w, mid).rotational == pytest.approx(0.0)
    assert blade_element_forces(w, rev).rotational > 0.0


def test_added_mass_follows_acceleration_not_velocity():
    w = elliptical_wing(20)
    steady = StrokeState(phi_dot=1000.0, phi_ddot=0.0)
    accel = StrokeState(phi_dot=1000.0, phi_ddot=1e5)
    assert blade_element_forces(w, steady).added_mass == pytest.approx(0.0)
    assert blade_element_forces(w, accel).added_mass != 0.0


def test_torque_is_the_force_weighted_by_its_moment_arm():
    w = elliptical_wing(20)
    f = blade_element_forces(w, StrokeState(alpha=np.deg2rad(45), phi_dot=1000.0))
    # The effective arm has to sit inside the wing, and outboard of mid-span
    # because force grows with radius.
    arm = f.torque / f.lift
    assert 0.5 * w.length < arm < w.length


# ------------------------------------------------------- the unit-error guard


def test_two_wings_hold_the_animal_up():
    """The only test that catches a dimensional mistake.

    A wrong density, a wrong length unit or a wrong mass unit all return
    perfectly plausible floats and nothing else notices. A real fly hovers, so
    lift within a small factor of body weight is the evidence that the unit
    system is coherent -- it is not evidence that the coefficients are right.

    An earlier density constant here was wrong by a million; this test is what
    it would have failed.
    """
    h = hover_check(elliptical_wing(20), BODY_MASS)
    assert 0.5 < h["ratio"] < 3.0, h


@pytest.mark.skipif(not MESH.is_file(), reason="needs the exported wing mesh")
def test_the_real_wing_mesh_gives_a_drosophila_wing():
    """Measured off NeuroMechFly's own RWing mesh rather than assumed: a real
    *Drosophila* wing is about 2.5 mm long with roughly 2 mm^2 of area."""
    w = wing_from_mesh(np.load(MESH), n=20, span_axis=2, chord_axis=1)
    assert 2.0 < w.length < 3.0
    assert 1.5 < w.area < 2.5
    assert 0.6 < w.mean_chord < 1.0
    assert 0.5 < hover_check(w, BODY_MASS)["ratio"] < 3.0


def test_density_is_air_and_not_something_else():
    """1.225 kg/m^3 expressed in the model's grams and millimetres."""
    assert AIR_DENSITY == pytest.approx(1.225e-6)


def test_stroke_average_scales_with_wing_area():
    small = elliptical_wing(20, mean_chord=0.4)
    big = elliptical_wing(20, mean_chord=0.8)
    assert stroke_average_lift(big) / stroke_average_lift(small) == pytest.approx(
        2.0, rel=1e-6
    )


def test_published_coefficients_are_unchanged():
    """The verified fits, pinned.

    These are the empirical coefficients for *Drosophila* from the
    dynamically-scaled wing work, checked against the literature and recorded
    as verified in docs/LITERATURE.md. Everything this project computes rests
    on them, and a drifted constant would still return plausible forces, so the
    published values are asserted rather than trusted to stay put.

    The source states the phase offsets in degrees; the code carries radians.
    Both forms are checked so a unit slip in either direction fails here.
    """
    from wingloop.aero.blade_element import DRAG_PHASE, LIFT_PHASE

    assert LIFT_PHASE == pytest.approx(np.deg2rad(7.2))
    assert DRAG_PHASE == pytest.approx(np.deg2rad(9.82))

    for a_deg in (0.0, 15.0, 45.0, 90.0):
        a = np.deg2rad(a_deg)
        assert lift_coefficient(a) == pytest.approx(
            0.225 + 1.58 * np.sin(2.13 * a - np.deg2rad(7.2)), rel=1e-6
        )
        assert drag_coefficient(a) == pytest.approx(
            1.92 - 1.55 * np.cos(2.04 * a - np.deg2rad(9.82)), rel=1e-6
        )

    # Where the fit puts its maximum, which is the whole point of it.
    a = np.deg2rad(np.linspace(0, 90, 9001))
    peak = np.rad2deg(a[int(np.argmax(lift_coefficient(a)))])
    assert peak == pytest.approx(45.6, abs=0.1)
