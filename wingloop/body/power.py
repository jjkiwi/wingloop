"""A stroke that is driven rather than imposed.

Everything up to here handed the wings a stroke and asked what forces it made.
That is the right way to validate a force model and the wrong way to ask what
a fly's muscles are doing, because it assumes the answer: the frequency, the
amplitude and the waveform were all written down rather than produced.

**Drosophila power muscles are asynchronous**, and the word means something
specific. They are not driven one action potential to one contraction. They
are *stretch-activated*: a muscle pulled out produces extra force a short
delay later, and the two antagonistic groups -- the dorsal longitudinal
muscles that drive the downstroke and the dorsoventral muscles that drive the
upstroke -- are mechanically coupled through the thorax so that each stretches
the other. Neural input sets how hard they pull, not when. The frequency comes
from the resonance of the thorax and wings.

So the model here is a resonant second-order system with a delayed
stretch-activation term:

    I phi_ddot + c phi_dot + k phi = -G * phi(t - delay)

``G`` is the neural drive. With a delay near a quarter cycle that right-hand
term is negative damping: it puts energy in, the oscillation grows, and a
force-length saturation stops it growing. Nothing tells it a frequency.

**What is calibrated and what is predicted.** The stiffness ``k`` is chosen to
put the resonance at the observed 218 Hz, so *that* number is an input and not
a result -- a real thorax's stiffness is a property of the animal and this
model has no way to derive it. What the model does predict, and what
distinguishes an asynchronous muscle from a neurally-timed one, is the
**relationship**: amplitude should follow the drive while frequency barely
moves. A synchronous model would give the opposite. That is the claim
``tests/test_power.py`` checks.

**Relayed, unverified.** That *Drosophila* flight muscle is asynchronous and
stretch-activated, and that the antagonistic pair is coupled through thorax
deformation, are stated from memory and recorded as relayed in
``docs/LITERATURE.md``. The delay being a fraction of the cycle is the
mechanism's requirement rather than a measured value.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Wingbeat frequency the stiffness is tuned to put the resonance at, Hz.
NOMINAL_FREQUENCY = 218.0

#: Wing inertia about the hinge in this model's units, from the mesh: a wing
#: masses 2.5e-6 and its radius of gyration is about 1.4 mm.
WING_INERTIA = 1.4e-5

#: Stretch-activation delay as a fraction of one cycle. A quarter cycle puts
#: the extra force in antiphase with velocity, which is where it does work.
DELAY_FRACTION = 0.25

#: Where the muscle force stops growing with strain, radians of stroke. The
#: force-length limit, and what stops the oscillation growing without bound.
SATURATION = np.deg2rad(70.0)


def stiffness_for(frequency: float = NOMINAL_FREQUENCY, inertia: float = WING_INERTIA):
    """Elastic stiffness that puts the resonance at a given frequency.

    This is the calibration, stated in one function so it cannot be mistaken
    for a derivation.
    """
    return inertia * (2.0 * np.pi * frequency) ** 2


@dataclass
class PowerOscillator:
    """Two antagonistic stretch-activated muscle groups on one resonant hinge.

    ``drive`` is what the descending neurons set -- DLMn and DVMn are both in
    MaleCNS -- and it scales the stretch-activation gain. It does not set the
    frequency, and the point of the model is that it cannot.
    """

    inertia: float = WING_INERTIA
    frequency: float = NOMINAL_FREQUENCY
    damping_ratio: float = 0.02
    delay_fraction: float = DELAY_FRACTION
    saturation: float = SATURATION
    drive: float = 1.0
    #: Gain of the stretch-activation term at drive 1, relative to stiffness.
    activation: float = 0.6
    #: Aerodynamic load coefficient, so that the drag moment is
    #: ``-load * phi_dot * |phi_dot|``. **This is the important one.** A real
    #: fly's wings are damped overwhelmingly by the air they are pushing, not
    #: by anything structural, and an oscillator without it has nothing to
    #: spend its muscle power on: the first version of this ran to amplitudes
    #: of 2125 degrees. Build it with :func:`aerodynamic_load` from the wing
    #: rather than guessing.
    load: float = 0.0

    angle: float = 0.0
    rate: float = 0.0
    t: float = 0.0
    _history: list = field(default_factory=list, repr=False)

    @property
    def stiffness(self) -> float:
        return stiffness_for(self.frequency, self.inertia)

    @property
    def natural_frequency(self) -> float:
        """Undriven resonance, Hz. What the stiffness was chosen to set."""
        return float(np.sqrt(self.stiffness / self.inertia) / (2.0 * np.pi))

    def _delayed_angle(self, dt: float) -> float:
        """The stroke angle one activation delay ago."""
        steps = max(1, int(round(self.delay_fraction / (self.frequency * dt))))
        if len(self._history) < steps:
            return self._history[0] if self._history else 0.0
        return self._history[-steps]

    def muscle_moment(self, dt: float) -> float:
        """Stretch activation: force from how the muscle was strained before.

        Saturating, because a muscle's force stops rising with length -- and
        because without it the negative damping has nothing to stop it and the
        amplitude runs away.
        """
        strained = self._delayed_angle(dt)
        return float(
            -self.drive
            * self.activation
            * self.stiffness
            * self.saturation
            * np.tanh(strained / self.saturation)
        )

    def step(self, dt: float) -> float:
        """Advance one timestep and return the new stroke angle."""
        omega = 2.0 * np.pi * self.frequency
        damping = 2.0 * self.damping_ratio * omega * self.inertia
        moment = (
            -self.stiffness * self.angle
            - damping * self.rate
            - self.load * self.rate * abs(self.rate)
            + self.muscle_moment(dt)
        )
        # Semi-implicit Euler: stable for an oscillator at these step sizes in
        # a way that plain forward Euler is not.
        self.rate += (moment / self.inertia) * dt
        self.angle += self.rate * dt
        self._history.append(self.angle)
        # One cycle of history is all the delay can ask for.
        if len(self._history) > int(round(2.0 / (self.frequency * dt))):
            self._history.pop(0)
        self.t += dt
        return self.angle

    def run(self, seconds: float, dt: float, *, kick: float = np.deg2rad(1.0)):
        """Start it from a small displacement and let it find its own cycle."""
        if self.angle == 0.0 and self.rate == 0.0:
            self.angle = kick
        out = {"t": [], "angle": [], "rate": []}
        for _ in range(int(round(seconds / dt))):
            self.step(dt)
            out["t"].append(self.t)
            out["angle"].append(self.angle)
            out["rate"].append(self.rate)
        return {k: np.asarray(v) for k, v in out.items()}


def aerodynamic_load(wing, *, drag_coefficient: float = 1.9, density: float = None):
    """Drag moment per unit of squared stroke rate, for a given wing.

    The blade-element drag on an element at radius ``r`` goes as ``r**2`` and
    its moment arm is ``r``, so the whole wing's damping is set by the third
    moment of area. Two wings, hence the factor.

    This is the load the muscle does work against, and what decides where the
    amplitude settles: the oscillation grows until the power the muscle puts
    in equals the power the air takes out.
    """
    from ..aero.wing import AIR_DENSITY

    rho = AIR_DENSITY if density is None else density
    return float(2.0 * 0.5 * rho * drag_coefficient * wing.moment(3))


def measure(trace: dict, *, settle: float = 0.5) -> dict[str, float]:
    """Amplitude and frequency of the steady cycle, from the trace's tail.

    Frequency is counted from zero crossings rather than fitted, so a waveform
    that is not a sinusoid is still measured rather than assumed.
    """
    t, angle = trace["t"], trace["angle"]
    keep = t > settle * t[-1]
    t, angle = t[keep], angle[keep]
    amplitude = float(0.5 * (angle.max() - angle.min()))
    crossings = np.flatnonzero(np.diff(np.sign(angle)) != 0)
    if len(crossings) < 3:
        return {"amplitude": amplitude, "frequency": float("nan")}
    span = t[crossings[-1]] - t[crossings[0]]
    cycles = (len(crossings) - 1) / 2.0
    return {
        "amplitude": amplitude,
        "frequency": float(cycles / span) if span > 0 else float("nan"),
    }


@dataclass
class PowerStroke:
    """A stroke generator whose sweep is produced, not written down.

    Drop-in for :func:`~wingloop.body.flight.harmonic_stroke`: same joint
    dictionary, same rates, same steering knobs. The difference is that the
    stroke angle and its rate come out of :class:`PowerOscillator` rather than
    out of a sine, so amplitude and frequency are consequences of the drive
    and the load instead of arguments.

    **Rotation stays prescribed, and that is not a shortcut.** In the animal
    the power muscles are asynchronous -- no fixed relationship between spikes
    and contractions -- while the steering muscles that set wing rotation are
    synchronous, firing once per wingbeat with a timing the nervous system
    controls. Generating the sweep and commanding the pitch is the division
    the animal actually has.
    """

    oscillator: PowerOscillator
    alpha: float = np.deg2rad(45.0)

    def __call__(
        self,
        dt: float,
        *,
        bias: float = 0.0,
        asymmetry: float = 0.0,
        phase: float = 0.0,
        phase_asymmetry: float = 0.0,
    ):
        """Advance the oscillator one step and return joint angles and rates."""
        from .flight import SHARPNESS, STROKE_OFFSET, STROKE_SIGN, WINGS

        self.oscillator.step(dt)
        phi, phi_dot = self.oscillator.angle, self.oscillator.rate
        # Rotation is still commanded, and still keyed to where the sweep is:
        # the wing flips at the ends of the stroke, so the phase reference is
        # the stroke's own velocity rather than a clock.
        amplitude = max(abs(phi), 1e-9)
        reference = amplitude * 2.0 * np.pi * self.oscillator.frequency
        s = float(np.clip(phi_dot / reference, -1.0, 1.0))
        gain = {"LWing": 1.0 + asymmetry, "RWing": 1.0 - asymmetry}
        flip = {"LWing": phase + phase_asymmetry, "RWing": phase - phase_asymmetry}

        angles, rates = {}, {}
        for wing in WINGS:
            sign = STROKE_SIGN[wing]
            angles[f"joint_{wing}_stroke"] = (
                STROKE_OFFSET[wing] + sign * (gain[wing] * phi + bias)
            )
            rates[f"joint_{wing}_stroke"] = sign * gain[wing] * phi_dot
            rot = self.alpha * np.tanh(SHARPNESS * (s + flip[wing])) / np.tanh(SHARPNESS)
            angles[f"joint_{wing}_rotation"] = sign * rot
            # The rotation follows the sweep, so its rate follows the sweep's
            # acceleration; a finite difference across the step is honest here
            # and avoids differentiating the oscillator analytically.
            rates[f"joint_{wing}_rotation"] = sign * (
                rot - getattr(self, f"_last_{wing}", rot)
            ) / max(dt, 1e-12)
            setattr(self, f"_last_{wing}", rot)
            angles[f"joint_{wing}_deviation"] = 0.0
            rates[f"joint_{wing}_deviation"] = 0.0
        return angles, rates
