"""A stroke that is produced rather than written down."""

from pathlib import Path

import numpy as np
import pytest

from wingloop.aero.wing import elliptical_wing, wing_from_mesh
from wingloop.body.power import (
    NOMINAL_FREQUENCY,
    PowerOscillator,
    aerodynamic_load,
    measure,
    stiffness_for,
)

MESH = Path(__file__).parent / "rwing_vertices.npy"
DT = 2e-6


@pytest.fixture(scope="module")
def load():
    return aerodynamic_load(wing_from_mesh(np.load(MESH), n=20, span_axis=2, chord_axis=1))


def test_the_stiffness_is_a_calibration_and_says_so():
    """The resonance is put at the observed wingbeat frequency by choosing the
    stiffness, so that number is an input. Stated in one function so it cannot
    be mistaken for a derivation."""
    osc = PowerOscillator()
    assert osc.stiffness == pytest.approx(stiffness_for(NOMINAL_FREQUENCY))
    assert osc.natural_frequency == pytest.approx(NOMINAL_FREQUENCY, rel=1e-9)


def test_it_does_not_oscillate_without_drive(load):
    """No neural activation, no flight. The stretch-activation term is what
    puts energy in, and at zero drive there is none."""
    trace = PowerOscillator(drive=0.0, load=load).run(0.20, DT)
    assert np.degrees(measure(trace)["amplitude"]) < 1.0


def test_drive_sets_amplitude_and_barely_touches_frequency(load):
    """The asynchronous signature, and the whole point of the model.

    *Drosophila* power muscles are stretch-activated rather than driven one
    spike to one contraction: the nervous system sets how hard they pull and
    the thorax's resonance sets how fast. So amplitude should follow the drive
    while frequency stays put. A synchronous model would do the opposite, and
    this is the measurement that tells them apart.
    """
    weak = measure(PowerOscillator(drive=0.5, load=load).run(0.25, DT))
    strong = measure(PowerOscillator(drive=2.0, load=load).run(0.25, DT))

    assert strong["amplitude"] > 3.0 * weak["amplitude"]
    assert strong["frequency"] == pytest.approx(weak["frequency"], rel=0.02)
    # And neither has wandered far from where the stiffness put the resonance.
    for m in (weak, strong):
        assert m["frequency"] == pytest.approx(NOMINAL_FREQUENCY, rel=0.02)


def test_amplitude_rises_monotonically_with_drive(load):
    amplitudes = [
        measure(PowerOscillator(drive=d, load=load).run(0.25, DT))["amplitude"]
        for d in (0.5, 1.0, 1.5, 2.0)
    ]
    assert all(b > a for a, b in zip(amplitudes[:-1], amplitudes[1:], strict=True))


def test_without_the_air_to_work_against_it_runs_away(load):
    """The bug that made this model honest.

    A real wing is damped overwhelmingly by the air it is pushing, and the
    amplitude settles where the muscle's power equals the air's. Left out --
    as the first version left it out -- the only sink is a token structural
    damping, and the oscillation grows to thousands of degrees.
    """
    loaded = measure(PowerOscillator(drive=1.0, load=load).run(0.25, DT))
    unloaded = measure(PowerOscillator(drive=1.0, load=0.0).run(0.25, DT))
    assert np.degrees(loaded["amplitude"]) < 90.0
    assert np.degrees(unloaded["amplitude"]) > 500.0


def test_the_load_comes_from_the_wing_rather_than_a_guess(load):
    """Blade-element drag goes as r squared with an arm of r, so the whole
    wing's damping is set by the third moment of area."""
    small = elliptical_wing(20, mean_chord=0.4)
    big = elliptical_wing(20, mean_chord=0.8)
    assert aerodynamic_load(big) == pytest.approx(2.0 * aerodynamic_load(small), rel=1e-6)
    assert load > 0.0


def test_the_stroke_generator_matches_the_joint_interface(load):
    """Drop-in for harmonic_stroke: same joints, same rates, same knobs."""
    from wingloop.body.flight import WINGS, harmonic_stroke
    from wingloop.body.power import PowerStroke

    osc = PowerOscillator(drive=2.0, load=load)
    osc.angle = np.deg2rad(1.0)
    generator = PowerStroke(osc)
    angles, rates = generator(2e-5, bias=0.1, asymmetry=0.05, phase_asymmetry=0.2)

    reference, _ = harmonic_stroke(0.0, rates=True)
    assert set(angles) == set(reference)
    assert set(rates) == set(reference)
    for wing in WINGS:
        assert np.isfinite(angles[f"joint_{wing}_stroke"])
        assert np.isfinite(rates[f"joint_{wing}_rotation"])
