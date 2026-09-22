"""Circulation that takes time to build, and does not vanish at reversal.

The quasi-steady model in :mod:`wingloop.aero.blade_element` answers one
question at every instant: given this velocity and this angle of attack, what
force would a wing in steady flow make? A real wing does not know its own
steady-state answer. After a sudden change it approaches that answer over
**distance travelled**, not over time -- a few chord lengths -- and the
consequence at a stroke reversal is the interesting one: the wing's velocity
passes through zero, so the quasi-steady force does too, while the real wing's
bound circulation persists across the turn and is still there when it starts
back.

That is the Wagner effect, and this is the smallest honest version of it: a
first-order lag on the translational force whose time constant is a fixed
number of chord lengths of travel divided by the current speed. Fast wing,
short lag; slow wing, long lag; stopped wing, no decay at all.

**What this deliberately is not.** Wake capture -- the wing re-encountering the
vortex it shed on the previous half-stroke -- is a different effect and is not
modelled here. Neither is the true Wagner function, which is a sum of
exponentials rather than one. A single lag reproduces the direction and the
timescale of the phenomenon and not its shape, which is enough to answer
whether wake memory is what a sharper stroke is missing, and not enough to
predict forces from.

**Relayed, unverified.** :data:`RISE_CHORDS` -- that circulation reaches most
of its steady value within a couple of chord lengths of travel -- is the one
number here and it comes from memory. ``docs/LITERATURE.md`` records it as
relayed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Chord lengths of travel over which circulation approaches its steady value.
RISE_CHORDS = 2.0

#: Speed below which the wing is treated as stopped for the purpose of
#: dividing by it. It is not a physical threshold, only a guard.
MIN_SPEED = 1.0


@dataclass
class WakeMemory:
    """A lag on one wing's **circulation**, not on its force.

    The distinction is the whole of this class, and getting it wrong gives an
    answer that is obviously wrong, which is lucky. Lag the force directly and
    the filter holds the mid-stroke peak through the reversal, where the true
    force is near zero -- a peak-hold rather than a memory. Measured, that
    inflated the mean lift 8.6-fold, from 13.7 to 117.

    What persists across a reversal is bound circulation. Force is circulation
    times velocity, so when the wing stops the force goes to zero however much
    circulation it is carrying, and picks straight back up when it moves again.
    So the lag is applied to ``force / speed`` and the instantaneous speed is
    multiplied back in.

    The lag also runs in **travelled distance** rather than in time, which is
    how the Wagner effect is stated: a stopped wing develops no new circulation
    at all, and the filter coefficient goes to zero with the distance step
    rather than the time constant going to infinity.
    """

    chord: float
    rise_chords: float = RISE_CHORDS
    min_speed: float = MIN_SPEED
    #: The lagged circulation-like quantities: force divided by speed.
    lift: float = 0.0
    drag: float = 0.0
    path_force: float = 0.0
    _started: bool = field(default=False, repr=False)

    def coefficient(self, speed: float, dt: float) -> float:
        """How much of the steady value is taken up this step, in [0, 1)."""
        travelled = abs(speed) * dt / self.chord
        return float(travelled / (self.rise_chords + travelled))

    def update(self, lift: float, drag: float, path_force: float, speed: float, dt: float):
        """Advance the lag one step and return the lagged forces."""
        u = max(abs(speed), self.min_speed)
        gamma = (lift / u, drag / u, path_force / u)
        if not self._started:
            self.lift, self.drag, self.path_force = gamma
            self._started = True
        else:
            a = self.coefficient(speed, dt)
            self.lift += a * (gamma[0] - self.lift)
            self.drag += a * (gamma[1] - self.drag)
            self.path_force += a * (gamma[2] - self.path_force)
        return self.lift * u, self.drag * u, self.path_force * u

    def reset(self) -> None:
        self.lift = self.drag = self.path_force = 0.0
        self._started = False


def reference_speed(wing, stroke_rate: float, body_velocity: float = 0.0) -> float:
    """Wing speed at the radius of gyration, which is where the force acts.

    Using the tip overstates the lag's speed and using mid-span understates it;
    the second moment of area is what the force integral is weighted by, so its
    radius is the one that matters.
    """
    radius = float(np.sqrt(wing.moment(2) / max(wing.area, 1e-12)))
    return abs(radius * stroke_rate + body_velocity)
