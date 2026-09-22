"""The feedback that makes flight possible, and the reason the animal needs it.

An open-loop stroke lifts this fly off and tips it over within about thirteen
milliseconds. That is not a defect of the force model -- the vertical force is
right to 0.3% on a vertical rail -- it is that a fly is passively unstable in
pitch. The wing hinge sits about a quarter of a millimetre ahead of the centre
of mass, so the mean aerodynamic force is a nose-down torque, and nothing in a
symmetric stroke opposes it.

Real flies close that loop with their halteres: club-shaped organs beating
antiphase to the wings, whose Coriolis deflection encodes body angular rate.
The signal reaches the wing steering muscles fast enough to matter within a
wingbeat. This is the functional stand-in -- body angular rate read straight
from the simulator rather than through a modelled mechanoreceptor, which is the
same simplification `flyloop` makes for vision and labels the same way.

**The two knobs are the animal's own**, and their authority here is measured,
not assumed:

* a symmetric shift of the mean stroke angle swings both wings fore or aft and
  moves the centre of pressure relative to the centre of mass -- **-0.33 of
  pitch torque per degree**, linear from -10 to +10 degrees, with the trim
  point at about +2 degrees
* an amplitude difference between the sides is roll -- **38.7 per unit of
  asymmetry**, linear through zero

Yaw is not controlled. Nothing in this stroke produces much of it, and adding a
third loop before the first two work would be tuning in the dark.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .flight import harmonic_stroke

#: Mean stroke offset at which the pitch torque about the centre of mass
#: vanishes, radians. Measured by sweeping the bias and reading
#: :meth:`~wingloop.body.flight.FlightBody.wrench_about_com`.
#:
#: The first value here was +2 degrees, read off the moment about the *model
#: origin* -- which on this model is a millimetre below the animal and gives
#: +0.66 where the moment about the centre of mass is -3.49. Wrong magnitude,
#: wrong sign, and the controller trimmed on it pushed the animal over faster
#: with the loop closed than without.
TRIM_BIAS = np.deg2rad(-10.7)

#: Control authority, measured about the centre of mass at the nominal stroke.
#: Pitch torque per radian of stroke bias, and roll torque per unit of
#: left-right amplitude asymmetry. The gains below are derived from these and
#: the body's inertia rather than tuned by hand.
PITCH_PER_BIAS = -18.85
ROLL_PER_ASYMMETRY = 38.2

#: Pitch and roll inertia of the whole animal, from the model's mass matrix.
PITCH_INERTIA = 0.002014
ROLL_INERTIA = 0.001502

#: How far the rotation phase may be shifted, radians. Beyond about this the
#: wing is flipping in the middle of the stroke rather than at its ends, which
#: is no longer a phase shift of the same stroke.
MAX_PHASE = np.deg2rad(45.0)

#: Closed-loop bandwidth, rad/s. Well under the wingbeat's 1370 rad/s, because
#: a loop that tries to act within a stroke is fighting the stroke.
#:
#: **What closing the loop buys, measured.** Open loop the animal is past 84
#: degrees of pitch and losing height by 40 ms. Closed, it holds pitch inside
#: 13 degrees and roll inside 19 for the first 100 ms -- twenty-two wingbeats
#: -- and over 300 ms it climbs 85 mm. That is flight.
#:
#: **It is not yet indefinite.** Past about 100 ms the attitude degrades and by
#: 300 ms it has swung through 60-80 degrees, still airborne and still
#: climbing. Raising or lowering the bandwidth and the filter moves this a
#: little and does not fix it, which points away from tuning and towards the
#: stroke: within one beat the torque about the centre of mass swings between
#: -17 and +20 while its cycle mean is near zero, against a control authority
#: of 8 in pitch and 17 in roll. The animal is kicked harder inside each stroke
#: than the loop can answer between strokes.
#:
#: The suspicion, untested, is the idealised kinematics -- a pure harmonic
#: sweep with a tanh flip and no deviation -- rather than the controller. Real
#: strokes put their rotation at the reversals and trace a figure-of-eight,
#: and both reduce the within-stroke excursion. Measuring that is the next
#: thing to do, and it is a statement about the stroke, not about the loop.
BANDWIDTH = 40.0


def attitude(body) -> tuple[float, float]:
    """Pitch and roll of the body, radians, from its rotation matrix.

    Pitch is how far the body's long axis has tilted out of horizontal and roll
    how far its lateral axis has. Both are zero in the model's rest pose, which
    is what makes them usable as errors without a reference to subtract.
    """
    if body.root_body is None:
        return 0.0, 0.0
    r = body.data.xmat[body.root_body].reshape(3, 3)
    return float(-np.arcsin(np.clip(r[2, 0], -1.0, 1.0))), float(
        np.arcsin(np.clip(r[2, 1], -1.0, 1.0))
    )


def angular_rate(body) -> np.ndarray:
    """Body angular velocity in world axes -- what a haltere reports.

    MuJoCo stores a free joint's angular velocity in the body frame, so it is
    rotated out here. Reading it raw gives a signal that is correct only while
    the animal is upright, which is exactly when the controller is not needed.
    """
    if body.root_body is None or body.root_translation != 3:
        return np.zeros(3)
    r = body.data.xmat[body.root_body].reshape(3, 3)
    return r @ np.asarray(body.data.qvel[body.root_dof + 3 : body.root_dof + 6])


@dataclass
class HaltereController:
    """Proportional-derivative attitude hold on pitch and roll.

    The derivative terms are the haltere part and they carry most of the work:
    rate feedback is what a haltere actually provides, and it is what damps an
    instability, while the proportional terms only decide what counts as level.
    """

    frequency: float = 218.0
    amplitude: float = np.deg2rad(75.0)
    trim_bias: float = TRIM_BIAS
    # Critically damped at BANDWIDTH: a gain is inertia times the desired
    # acceleration divided by the authority that produces it.
    pitch_gain: float = PITCH_INERTIA * BANDWIDTH**2 / abs(PITCH_PER_BIAS)
    pitch_rate_gain: float = PITCH_INERTIA * 2 * BANDWIDTH / abs(PITCH_PER_BIAS)
    roll_gain: float = ROLL_INERTIA * BANDWIDTH**2 / ROLL_PER_ASYMMETRY
    roll_rate_gain: float = ROLL_INERTIA * 2 * BANDWIDTH / ROLL_PER_ASYMMETRY
    #: Bounds on what the loop may ask for, in the units of the two knobs. A
    #: stroke bias beyond this is no longer a bias and an asymmetry beyond it
    #: stops one wing entirely.
    max_bias: float = np.deg2rad(25.0)
    max_asymmetry: float = 0.45
    #: Time constant of the low-pass on the sensed attitude and rate, seconds.
    #:
    #: Not a fudge factor. Within a single stroke the torque about the centre
    #: of mass swings between -17 and +20 while its cycle mean is near zero, so
    #: a loop reading instantaneous state responds mostly to the stroke rather
    #: than to the animal's attitude, and feeds that straight back at stroke
    #: frequency. One wingbeat is 4.6 ms; filtering over about two of them
    #: leaves the signal and drops the beat. The haltere-to-muscle path is
    #: itself low-pass, so this stands in for something real.
    #: Time constant of the low-pass on the sensed attitude and rate.
    #:
    #: **5 ms, and the value matters more than anything else here.** Swept
    #: against loop bandwidth, controlled flight lasts 353 ms at 5 ms of filter
    #: lag and 187 at 20 -- and at every bandwidth tried, more lag is worse.
    #: That is what finally identified the limit: the loop is close to its
    #: phase margin, and everything inside it that adds delay costs flight
    #: time. It explains the integral term making things slightly worse, and
    #: the circulation lag in :mod:`wingloop.aero.wake` making them much worse.
    #:
    #: It cannot go to zero. Within a stroke the torque about the centre of
    #: mass swings between -28 and +27 while its cycle mean is near zero, so an
    #: unfiltered loop responds mostly to the beat; at 3 ms the flight is back
    #: down to 146 ms. The filter is trading stroke noise against phase margin
    #: and 5 ms is where that trade sits, measured rather than assumed.
    tau: float = 0.005
    #: Stroke shape, passed through to :func:`harmonic_stroke`. ``sharpness``
    #: bends the sweep from a sinusoid toward a triangle and ``deviation``
    #: adds the out-of-plane motion that makes a wingtip trace a
    #: figure-of-eight. Both default to off, which is the sinusoid every
    #: earlier result was measured on.
    sharpness: float = 0.0
    deviation: float = 0.0
    deviation_phase: float = 0.0
    #: Integral gains, per second. **Off by default, because they were measured
    #: not to help.**
    #:
    #: The reasoning for adding them was sound and the diagnosis behind it is
    #: still true: proportional-derivative alone leaves a steady-state offset
    #: against a constant disturbance, there is one -- the pitch trim
    #: cross-couples into roll by +0.6 -- and the roll loop does settle about
    #: 10 degrees off level and stay there, without ever saturating (the
    #: commanded asymmetry sits at 0.03 against a limit of 0.45).
    #:
    #: Closing that offset changes nothing. At gain 4 the animal holds attitude
    #: for 179 ms against 187 without, which is slightly worse. So the standing
    #: roll offset is not what ends the flight, and the integral is kept
    #: available and disabled rather than quietly left in.
    pitch_integral_gain: float = 0.0
    roll_integral_gain: float = 0.0
    #: Bound on the accumulated terms, in the units of each knob, so a period
    #: of saturation or a tumble cannot wind them up into a command that
    #: outlives the error that produced it.
    max_integral: float = 0.25

    _state: dict = field(default_factory=dict)
    #: Where a run stopped because the simulation diverged, seconds, or None.
    diverged_at: float | None = None

    def _integrate(self, pitch, roll, dt):
        """Accumulate the attitude error, bounded."""
        self._state["i_pitch"] = float(
            np.clip(
                self._state.get("i_pitch", 0.0) + self.pitch_integral_gain * pitch * dt,
                -self.max_integral,
                self.max_integral,
            )
        )
        self._state["i_roll"] = float(
            np.clip(
                self._state.get("i_roll", 0.0) + self.roll_integral_gain * roll * dt,
                -self.max_integral,
                self.max_integral,
            )
        )
        return self._state["i_pitch"], self._state["i_roll"]

    def _filtered(self, pitch, roll, rate, dt):
        a = dt / (self.tau + dt)
        if not self._state:
            self._state = {"pitch": pitch, "roll": roll, "rate": np.asarray(rate, float)}
        else:
            self._state["pitch"] += a * (pitch - self._state["pitch"])
            self._state["roll"] += a * (roll - self._state["roll"])
            self._state["rate"] += a * (np.asarray(rate, float) - self._state["rate"])
        return self._state["pitch"], self._state["roll"], self._state["rate"]

    def command(self, body, t: float):
        """Stroke angles and rates for this instant, with the loop closed."""
        pitch, roll = attitude(body)
        rate = angular_rate(body)
        pitch, roll, rate = self._filtered(
            pitch, roll, rate, float(body.model.opt.timestep)
        )
        # Pitch authority is negative -- more bias is more nose-down torque --
        # so correcting a positive pitch means *more* bias, not less.
        i_pitch, i_roll = self._integrate(pitch, roll, float(body.model.opt.timestep))
        bias = (
            self.trim_bias
            + self.pitch_gain * pitch
            + self.pitch_rate_gain * rate[1]
            + i_pitch
        )
        asymmetry = -self.roll_gain * roll - self.roll_rate_gain * rate[0] - i_roll
        return harmonic_stroke(
            t,
            amplitude=self.amplitude,
            frequency=self.frequency,
            bias=float(np.clip(bias, -self.max_bias, self.max_bias)),
            asymmetry=float(np.clip(asymmetry, -self.max_asymmetry, self.max_asymmetry)),
            **self.shape,
        )

    @property
    def shape(self) -> dict:
        """Stroke-shape arguments, so every call site stays in step."""
        return {
            "sharpness": self.sharpness,
            "deviation": self.deviation,
            "deviation_phase": self.deviation_phase,
            "rates": True,
        }

    def fly(self, body, seconds: float) -> dict[str, np.ndarray]:
        """Run the loop and report the trajectory."""
        steps = int(round(seconds / body.model.opt.timestep))
        out = {
            k: []
            for k in ("t", "x", "y", "z", "pitch", "roll", "tumble", "bearing", "heading")
        }
        # Heading is accumulated rather than read from atan2 each step. With the
        # loop tuned, a steering command can carry the animal past half a turn
        # inside 100 ms, and a wrapped angle then reports a hard left as a
        # right -- which is exactly how a working steering test starts failing.
        turned = 0.0
        previous = None
        # MuJoCo resets the whole state when it detects a divergence, and the
        # clock goes back to zero with it. Carrying on past that point
        # silently appends a second flight to the first, and any statistic
        # taken over "the first 200 ms" then mixes the two. Stop instead.
        last_time = -1.0
        for _ in range(steps):
            if body.t < last_time:
                self.diverged_at = last_time
                break
            last_time = body.t
            angles, rates = self.command(body, body.t)
            body.set_wings(angles, rates)
            body.apply_aerodynamics()
            body._mj.mj_step(body.model, body.data)
            pitch, roll = attitude(body)
            q = body.data.qpos
            out["t"].append(body.t)
            out["x"].append(float(q[0]))
            out["y"].append(float(q[1]))
            out["z"].append(float(q[2]))
            out["pitch"].append(pitch)
            out["roll"].append(roll)
            out["tumble"].append(float(np.linalg.norm(angular_rate(body))))
            out["bearing"].append(float(getattr(self, "bearing", 0.0)))
            m = body.data.xmat[body.root_body].reshape(3, 3) if body.root_body else None
            raw = float(np.arctan2(m[1, 0], m[0, 0])) if m is not None else 0.0
            if previous is not None:
                step = (raw - previous + np.pi) % (2 * np.pi) - np.pi
                turned += step
            previous = raw
            out["heading"].append(np.degrees(turned))
        return {k: np.asarray(v) for k, v in out.items()}


@dataclass
class SteeringController(HaltereController):
    """Attitude hold, with the connectome deciding which way to go.

    The stabiliser keeps the animal upright; this adds the one thing it is not
    doing, which is choosing a heading. The steering command is a wing amplitude
    asymmetry -- the same knob the roll loop uses -- so the two simply add, and
    the stabiliser fights the roll that steering deliberately creates. That is
    the arrangement in the animal too: the haltere loop does not know the
    difference between a disturbance and an intention.

    ``readout`` is a :class:`~wingloop.brain.readout.FlightReadout`, and
    ``bearing`` is where the object sits relative to the fly's heading,
    positive to the right. Nothing here recomputes the brain: the readout is a
    lookup over a curve measured once, for the reason its module explains.
    """

    readout: object = None
    bearing: float = 0.0
    steer_gain: float = 1.0
    #: World position of the thing being looked at, ``(x, y)``. Given one, the
    #: bearing is recomputed from the animal's own heading every step and
    #: ``bearing`` is ignored -- the loop is closed. Left as None, ``bearing``
    #: is held fixed, which measures the open-loop steering response.
    target: tuple | None = None
    #: Treat the target as a direction rather than a place: the bearing then
    #: depends on heading alone and never changes with position. That is a
    #: distant landmark, and it separates turning from approaching.
    distant: bool = False
    #: Which knob the steering command drives.
    #:
    #: ``"phase"`` shifts when the wings flip relative to the stroke, and it is
    #: the default because it is the one that works: a commanded right turn
    #: yaws right from 10 ms onward and never reverses. ``"amplitude"`` beats
    #: one wing harder, which is the obvious knob and the wrong one -- the
    #: extra drag yaws the animal *away* from the turn for the first 50 ms,
    #: by 46 degrees, before the bank finally takes over. Kept so the
    #: comparison stays runnable.
    #:
    #: Measured about the centre of mass: phase asymmetry is a yaw torque,
    #: +-0.316 at +-30 degrees and antisymmetric, while amplitude asymmetry is
    #: a roll torque of +-7.8 at +-0.2. They are different controls, not two
    #: strengths of the same one.
    steer_mode: str = "phase"
    #: Radians of rotation-phase asymmetry per unit of readout command. The
    #: readout spans about +-0.1, so this puts the full visual field at roughly
    #: +-30 degrees of phase.
    phase_gain: float = 5.2

    def __post_init__(self):
        if self.steer_mode not in ("phase", "amplitude"):
            raise ValueError(
                f"steer_mode must be 'phase' or 'amplitude', not {self.steer_mode!r}"
            )

    def current_bearing(self, body) -> float:
        """Where the object is, in degrees, positive to the fly's right.

        The sign convention has to match the readout's, and it is not the
        obvious one: this fly faces +x and its **left is +y**, because the
        right wing's span points to -y. So a counter-clockwise angle from the
        heading puts the object on the left, and the bearing is its negation.
        Getting this backwards gives an animal that turns smoothly away from
        whatever it is looking at, which reads as a plausible avoidance
        behaviour rather than as a bug.
        """
        if self.target is None or body.root_body is None:
            return self.bearing
        rot = body.data.xmat[body.root_body].reshape(3, 3)
        heading = rot @ np.array([1.0, 0.0, 0.0])
        heading = heading[:2]
        norm = np.linalg.norm(heading)
        if norm < 1e-9:  # nose straight up or down; hold the last command
            return self.bearing
        heading = heading / norm
        if self.distant:
            to_target = np.asarray(self.target, dtype=float)
        else:
            to_target = np.asarray(self.target, dtype=float) - body.data.qpos[:2]
        span = np.linalg.norm(to_target)
        if span < 1e-9:  # standing on it
            return 0.0
        to_target = to_target / span
        cross = heading[0] * to_target[1] - heading[1] * to_target[0]
        self.bearing = float(-np.degrees(np.arctan2(cross, float(heading @ to_target))))
        return self.bearing

    def steering(self, body=None) -> float:
        if self.readout is None:
            return 0.0
        bearing = self.bearing if body is None else self.current_bearing(body)
        return self.steer_gain * self.readout.asymmetry(bearing)

    def command(self, body, t: float):
        angles, rates = super().command(body, t)
        extra = self.steering(body)
        if extra:
            # Re-issue the stroke with the steering asymmetry folded in. The
            # stabiliser's own asymmetry is already inside `angles`, so this is
            # recomputed rather than patched -- editing the joint dictionary
            # would leave the rates describing a different stroke.
            pitch, roll = attitude(body)
            pitch, roll, rate = self._filtered(
                pitch, roll, angular_rate(body), float(body.model.opt.timestep)
            )
            i_pitch = self._state.get("i_pitch", 0.0)
            i_roll = self._state.get("i_roll", 0.0)
            bias = (
                self.trim_bias
                + self.pitch_gain * pitch
                + self.pitch_rate_gain * rate[1]
                + i_pitch
            )
            asym = -self.roll_gain * roll - self.roll_rate_gain * rate[0] - i_roll
            phase_asymmetry = 0.0
            if self.steer_mode == "phase":
                phase_asymmetry = float(
                    np.clip(self.phase_gain * extra, -MAX_PHASE, MAX_PHASE)
                )
            else:
                asym += extra
            angles, rates = harmonic_stroke(
                t,
                amplitude=self.amplitude,
                frequency=self.frequency,
                bias=float(np.clip(bias, -self.max_bias, self.max_bias)),
                asymmetry=float(np.clip(asym, -self.max_asymmetry, self.max_asymmetry)),
                phase_asymmetry=phase_asymmetry,
                **self.shape,
            )
        return angles, rates
