"""The fast fly: a reduced-order model of the flying fly, fitted to the full physics.

The full simulation -- blade-element wings stepped every 20 microseconds,
haltere, throttle and yaw loops -- runs about forty times slower than the
animal. What the navigator and the person watching need from it is much
less: how the fly's speed, heading and height answer its three setpoints.
Those answers were measured on the full physics (the step responses below),
and they are simple enough to be a handful of numbers:

======================  ===========================================  ==========
setpoint                response, measured on the full simulation    here
======================  ===========================================  ==========
pitch 4/8/12/15 deg     119/204/293/364 mm/s, 63% in 0.115-0.12 s     30 + 22.3 p
                        (and 12-15 mm/s of sideways drift)            first order
heading step, hovering  37% at 50 ms, 73% at 100, 95% at 200          2nd order,
                        a 240 deg/s ramp followed 12 deg behind       28 rad/s
heading step, at        51% at 100 ms, 80% at 200                     17 rad/s at
290 mm/s                                                              290 mm/s
height step             21% at 50 ms, 59% at 100, 90% at 200          2nd order,
                                                                      20 rad/s
body pitch              0.86-1.05 of the setpoint, plus about 1       1 + 0.8 p
======================  ===========================================  ==========

Each is a critically damped loop or a first-order lag at the measured rate,
which is what the full simulation's loops are designed to be; the test pins
the model against the recorded traces. What it leaves out: the wingbeat-scale
wobble of the body, the sag in height when the fly pitches over, and the few
degrees of overshoot in a large turn. The wings are drawn beating at 218 Hz
from the stroke generator, for the eye of the person watching -- the model's
flight does not come from them.

With it, and the optic lobe stepped at its own trained 20 ms rather than 5,
a mission runs in about real time instead of forty times slower.
"""

from __future__ import annotations

import math

import numpy as np

from ..body.control import CLIMB_PER_COMMAND, HOVER_COMMAND
from .scene import rotation

#: Forward speed per degree of pitch setpoint, mm/s, and at zero pitch.
SPEED_PER_DEGREE = 22.3
SPEED_AT_ZERO = 30.0
#: Sideways drift, mm/s (to the fly's right): the full model's standing roll offset.
SIDE_DRIFT = -12.0
#: Time constant of the speed's answer to pitch, s.
SPEED_TAU = 0.12
#: Heading loop, rad/s: critically damped, slower as the fly goes faster.
HEADING_BANDWIDTH = 28.0
HEADING_BANDWIDTH_PER_SPEED = 0.038  # rad/s lost per mm/s of speed
#: Height loop, rad/s: the throttle loop's own bandwidth.
HEIGHT_BANDWIDTH = 20.0
#: Body pitch, degrees: an offset and a fraction of the setpoint, with a lag.
PITCH_OFFSET = 1.0
PITCH_FRACTION = 0.8
PITCH_TAU = 0.03
#: Roll, degrees: a standing offset, and a bank into turns per deg/s of yaw.
ROLL_OFFSET = 0.8
ROLL_PER_YAW_RATE = 0.005


class FastFly:
    """The fly's answers to its setpoints, with the interface of :class:`~.watch.Flight`."""

    def __init__(self, height_mm: float = 5.0, timestep: float = 0.001):
        from ..body.control import HaltereController

        self.timestep = timestep
        self._t = 0.0
        self.x = self.y = 0.0
        self.z, self.climb = height_mm, 0.0
        self.forward, self.side = 0.0, 0.0
        self.heading, self.yaw_rate = 0.0, 0.0  # degrees, degrees/s
        self.pitch, self.roll = PITCH_OFFSET, ROLL_OFFSET
        self.target = (0.0, 0.0, height_mm)
        self.knobs: dict = {}
        from .flysim import Brain

        c = HaltereController()
        self._stroke = dict(amplitude=c.amplitude, sharpness=c.sharpness)
        self.brain = Brain.stored()
        self._throttle = 0.0

    @property
    def t(self) -> float:
        return self._t

    def command(self, pitch_deg: float, heading_deg: float, height_mm: float) -> None:
        self.target = (float(pitch_deg), float(heading_deg), float(height_mm))

    def step(self, n: int = 1) -> None:
        dt = self.timestep
        p, h, z = self.target
        for _ in range(n):
            speed = math.hypot(self.forward, self.side)
            w = max(8.0, HEADING_BANDWIDTH - HEADING_BANDWIDTH_PER_SPEED * speed)
            accel = w * w * (h - self.heading) - 2.0 * w * self.yaw_rate
            self.yaw_rate += accel * dt
            self.heading += self.yaw_rate * dt
            self.forward += (
                (SPEED_AT_ZERO + SPEED_PER_DEGREE * p - self.forward) * dt / SPEED_TAU
            )
            self.side += (SIDE_DRIFT - self.side) * dt / SPEED_TAU
            wz = HEIGHT_BANDWIDTH
            lift = wz * wz * (z - self.z) - 2.0 * wz * self.climb
            self.climb += lift * dt
            self.z += self.climb * dt
            self.pitch += (PITCH_OFFSET + PITCH_FRACTION * p - self.pitch) * dt / PITCH_TAU
            self.roll = ROLL_OFFSET + ROLL_PER_YAW_RATE * self.yaw_rate
            c, s = math.cos(math.radians(self.heading)), math.sin(math.radians(self.heading))
            self.x += (c * self.forward - s * self.side) * dt
            self.y += (s * self.forward + c * self.side) * dt
            self._t += dt
        # The flight command the throttle loop would be giving for this climb:
        # what the connectome's flight neurons are read out at.
        self._throttle = HOVER_COMMAND + lift / CLIMB_PER_COMMAND

    def _velocity(self) -> tuple[float, float]:
        c, s = math.cos(math.radians(self.heading)), math.sin(math.radians(self.heading))
        return c * self.forward - s * self.side, s * self.forward + c * self.side

    def state(self) -> dict:
        vx, vy = self._velocity()
        return dict(
            t=self._t,
            x=self.x,
            y=self.y,
            z=self.z,
            vx=vx,
            vy=vy,
            speed=math.hypot(vx, vy),
            climb=self.climb,
            pitch=self.pitch,
            roll=self.roll,
            heading=self.heading,
            neurons=self.brain.read(self._throttle, None),
            fast=True,
        )

    def pose(self) -> tuple[np.ndarray, np.ndarray]:
        R = rotation(
            math.radians(self.roll), math.radians(self.pitch), math.radians(self.heading)
        )
        return np.array([self.x, self.y, self.z]), R

    def rates(self) -> np.ndarray:
        return np.array([0.0, 0.0, math.radians(self.yaw_rate)])

    def wing_angles(self) -> dict:
        """The stroke at this instant, for drawing the wings."""
        from ..body.flight import harmonic_stroke

        return harmonic_stroke(self._t, frequency=218.0, **self._stroke)

    def upright(self) -> bool:
        return True


__all__ = ["FastFly"]
