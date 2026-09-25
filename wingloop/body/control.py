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

**Yaw is the third loop, and it turned out to be the one that mattered.** It
was left out because nothing in a symmetric stroke produces much of it --
true, and beside the point: yaw is the light axis, inertia 0.000591 against
0.002014 in pitch, so the little it gets is enough. A normal flight reached
+40 degrees of heading by 100 ms and -86 by 300, at up to 1587 deg/s, while
pitch and roll stayed inside ten.

The knob is the left-right *rotation phase* asymmetry, which is the only one
that yaws the animal at all: see :data:`YAW_PER_PHASE`. Closing it takes the
flight from 320 ms to 1004, holding the heading inside ten degrees the whole
way. Everything above it -- the sensor filter, the loop bandwidth, the whole
phase-margin argument -- was worth tens of milliseconds against that.

What ends the flight is still the heading, which departs twenty to fifty
milliseconds before the attitude does. Three things were found looking for
why, and they moved it from about 1069 ms to about 1250: the wings had never
been told the body was rotating, so there was no yaw damping at all (see
:meth:`~wingloop.body.flight.FlightBody.air_velocity_at`); a rolled animal
that is climbing yaws, which was consuming over half the yaw knob before any
disturbance arrived (see :attr:`HaltereController.roll_integral_gain`); and
the yaw knob drags roll with it, which is cheaper to cancel in advance than to
answer (see :attr:`HaltereController.compensate_yaw_roll`).

It is delayed rather than fixed, and further tuning is not what will fix it:
the differences between settings are now smaller than the differences between
stroke amplitudes, which is the situation that produced the filter-lag spike
:data:`SENSING` had to withdraw.

**A second loop, on a second variable.** :class:`Throttle` closes altitude the
same way, through the connectome's power channel rather than these two knobs:
height error into a flight command, and the command into how hard the muscle
pulls. It is a separate loop because it acts on a separate thing -- which way
the animal points against how hard it flies -- and because the wings serve
both, the two add on the same stroke, exactly as steering and roll do.

It does not extend the flight. Free, with both loops closed, the animal holds
attitude for 276 ms against 338 at a held full command: the limit is still the
attitude loop's phase margin, and closing a loop on height was never going to
move it. What it changes is that the height stops being a side effect of
whichever command was typed.
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

#: Pitch, roll and yaw inertia of the whole animal, from the model's mass
#: matrix. **Yaw is the light axis**, a third of pitch, which is why an
#: uncontrolled yaw runs away faster than either of the others.
PITCH_INERTIA = 0.002014
ROLL_INERTIA = 0.001502
YAW_INERTIA = 0.000591

#: Roll torque the yaw knob drags with it, as the measured curve itself.
#:
#: Swept across the knob's whole range, reading the stroke-averaged roll about
#: the centre of mass and subtracting the undeflected value. Strongly
#: asymmetric -- +1.51 at +45 degrees against +0.33 at -45 -- so it has a
#: large even part that does not cancel between the sides.
#:
#: **Interpolated rather than fitted, because the fits were not good enough to
#: defend.** A quadratic through zero leaves 12.9% of full scale, a cubic
#: 13.2%, and only a quartic gets to 2.9% -- at which point the polynomial is
#: carrying the shape rather than describing it. The first version of this was
#: an unconstrained quadratic, which fitted the middle well and predicted
#: +0.14 of roll at zero deflection, where there is none by construction.
#: These are the measurements; the knob is clipped to this range, so nothing
#: extrapolates.
#:
#: In the units the roll loop works in this is small -- 0.039 of amplitude
#: asymmetry at full deflection, against a limit of 0.45 -- which is the
#: reason it was expected not to matter. See
#: :attr:`HaltereController.compensate_yaw_roll` for what it is actually
#: worth.
ROLL_FROM_PHASE = (
    np.linspace(-np.deg2rad(45.0), np.deg2rad(45.0), 13),
    np.array([
        0.3266, 0.3296, 0.2751, 0.1759, 0.0615, -0.0174, 0.0000,
        0.1501, 0.4167, 0.7452, 1.0728, 1.3426, 1.5065,
    ]),
)

#: Yaw torque per radian of left-right *rotation phase* asymmetry: the two
#: wings flipping at slightly different points in the stroke.
#:
#: Measured by sweeping the knob and averaging the torque about the centre of
#: mass over a stroke. Odd in the knob and linear to 3.6% of full scale from
#: -0.45 to +0.45 rad. Negative because advancing the left wing's flip yaws
#: the animal right.
#:
#: **This is the only knob that makes any yaw at all.** The amplitude
#: asymmetry that rolls the animal makes exactly 0.0000 of it, and the
#: symmetric phase shift makes 0.0004. That is why the steering work found
#: rotation phase to be a yaw control and amplitude a roll control, and it is
#: what this loop is built on.
#:
#: It is not clean: the same knob also makes roll, 0.58 to 1.50 across that
#: range and mostly *even* in the knob, so it does not cancel between sides.
#: The roll loop absorbs it -- against a roll authority of 38.2 per unit of
#: amplitude asymmetry, the worst of it costs 0.04 of a knob that saturates at
#: 0.45 -- but the two loops are coupled through it, in that direction only.
YAW_PER_PHASE = -0.5666

#: How far the rotation phase may be shifted, radians. Beyond about this the
#: wing is flipping in the middle of the stroke rather than at its ends, which
#: is no longer a phase shift of the same stroke.
MAX_PHASE = np.deg2rad(45.0)

#: Closed-loop bandwidth, rad/s. Well under the wingbeat's 1370 rad/s, because
#: a loop that tries to act within a stroke is fighting the stroke.
#:
#: **What closing the loop buys, measured.** Open loop the animal is past 84
#: degrees of pitch and losing height by 40 ms. Closed, and with the sensing
#: below, it holds the whole 300 ms inside 3.2 degrees of pitch and 9.9 of
#: roll -- sixty-five wingbeats -- climbing 154 mm. That is flight.
#:
#: **It is not yet indefinite**, and how far it goes is set by how the loop
#: senses. 40 rad/s was right while the loop read through a first-order
#: filter, which is what every result before :data:`SENSING` was measured on:
#: above 40 the lag in that filter cost more than the gain bought, and flight
#: fell away monotonically. Averaging over a wingbeat instead removes the lag
#: without the stroke ripple, and the ceiling moves with it -- measured across
#: three stroke amplitudes, so a coincidence would show:
#:
#: ======  =================  ==============
#: rad/s   first-order 6 ms   stroke boxcar
#: ======  =================  ==============
#: 30            160 ms           162 ms
#: 40            236 ms           237 ms
#: 50            209 ms           277 ms
#: 55            183 ms           295 ms
#: 60            163 ms           320 ms
#: 70            146 ms           260 ms
#: 80            134 ms           154 ms
#: ======  =================  ==============
#:
#: The two agree exactly where the old default sat and diverge above it. That
#: is the phase margin, spent on loop gain instead of on lag: **236 ms to 320,
#: and the usable bandwidth from 40 to 60.**
BANDWIDTH = 60.0

#: What the sensing change is worth, and the measurement that nearly hid it.
#:
#: **A warning about this project's own numbers first.** Flight time against
#: filter lag is a smooth hump with a *spike* on it. Swept finely, the
#: first-order filter gives 193 ms at 4.0, 214 at 4.5, 249 at 4.8 -- then 353
#: at 5.0 and 379 at 5.2 -- then 256 at 5.5 and 236 at 6.0. The old default
#: was chosen at the top of that spike and the project quoted 353 ms as what
#: the filter was worth.
#:
#: It is not. Change the stroke amplitude by 1%, which has nothing to do with
#: the filter, and the spike moves: at 74.25 degrees the peak is at 4.8 ms,
#: at 75.00 it is at 5.2, at 75.75 it is at 5.5. There is always a spike
#: somewhere near 350-380 ms and where it lands is a coincidence. The honest
#: number for the first-order filter is the trend it sits on, about 240 ms,
#: and every comparison against 353 was a comparison against luck.
#:
#: Which is how the boxcar first looked worse: 237 ms against 353. Against the
#: trend it is level, and above the old bandwidth it is ahead, which is the
#: table on :data:`BANDWIDTH`. Single-run flight times are repeatable here to
#: within a millisecond -- nudging the initial pitch rate changes nothing --
#: so the spike is real. It is just not a property of the filter.
SENSING = ("lowpass", "stroke")


#: The bandwidth the first-order filter tops out at, for comparisons against
#: it. :data:`BANDWIDTH` is the boxcar's, and running the low-pass at that is
#: running it past where it works.
LOWPASS_BANDWIDTH = 40.0


def gains_for(bandwidth: float) -> dict:
    """The four attitude gains for a bandwidth, critically damped.

    A gain is inertia times the desired closed-loop dynamics divided by the
    measured control authority -- the same formula the defaults use, exposed
    so that anything comparing bandwidths states the one it means.
    """
    return {
        "pitch_gain": PITCH_INERTIA * bandwidth**2 / abs(PITCH_PER_BIAS),
        "pitch_rate_gain": PITCH_INERTIA * 2 * bandwidth / abs(PITCH_PER_BIAS),
        "roll_gain": ROLL_INERTIA * bandwidth**2 / ROLL_PER_ASYMMETRY,
        "roll_rate_gain": ROLL_INERTIA * 2 * bandwidth / ROLL_PER_ASYMMETRY,
        "yaw_gain": YAW_INERTIA * bandwidth**2 / abs(YAW_PER_PHASE),
        "yaw_rate_gain": YAW_INERTIA * 2 * bandwidth / abs(YAW_PER_PHASE),
    }


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


#: Vertical acceleration per unit flight command, mm/s^2. Measured on the rail
#: by sweeping the command and fitting the slope of the vertical velocity once
#: the muscle has settled: the response is linear to the eye across 0.5 to 1.0.
#:
#: This is the altitude loop's control authority, and the gains below are
#: derived from it and the wingbeat rather than tuned, the same way the
#: attitude gains come from :data:`PITCH_PER_BIAS` and the body's inertia.
CLIMB_PER_COMMAND = 21038.0

#: Flight command at which lift equals weight, so the animal neither climbs nor
#: sinks. The zero crossing of the same sweep.
#:
#: Not the same number as the break-even on a 200 ms rail run, which is nearer
#: 0.75: that one starts from rest and spends the first wingbeats falling while
#: the muscle spins up, so it has height to make back. This is the steady
#: state, and it is what the loop trims around.
HOVER_COMMAND = 0.695

#: Altitude loop bandwidth, rad/s, and its damping.
#:
#: Half the attitude loop's 40: an animal cannot usefully chase a height faster
#: than it can hold the attitude it climbs on, and this loop acts through the
#: same wings. Critically damped, because an altitude loop that overshoots
#: downwards has a floor to hit.
ALTITUDE_BANDWIDTH = 20.0
ALTITUDE_DAMPING = 1.0


def heading(body) -> float:
    """Which way the animal is pointing, radians, wrapped to (-pi, pi].

    Read from the simulator like :func:`attitude`. In the animal yaw is sensed
    by the same halteres -- their Coriolis deflection encodes all three axes --
    so unlike altitude this needs no second sense to be plausible.
    """
    if body.root_body is None:
        return 0.0
    m = body.data.xmat[body.root_body].reshape(3, 3)
    return float(np.arctan2(m[1, 0], m[0, 0]))


def altitude(body) -> tuple[float, float]:
    """Height and climb rate, in millimetres and mm/s.

    Read from the simulator, as :func:`attitude` is, rather than through a
    modelled sense organ. In the animal height comes from vision -- ventral
    optic flow -- and nothing here stands in for that; what this asserts is
    only that the signal exists, not how it is obtained.
    """
    if body.root_translation == 3:
        i = body.root_dof
        return float(body.data.qpos[i + 2]), float(body.data.qvel[i + 2])
    if body.root_translation == 1:
        i = body.root_dof
        return float(body.data.qpos[i]), float(body.data.qvel[i])
    return 0.0, 0.0


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
    #: Yaw, through the rotation-phase asymmetry. Set both to zero for the
    #: uncontrolled yaw every result before this was measured with.
    #:
    #: **The bandwidth is shared with pitch and roll, and a sweep says it
    #: should be.** Yaw carries a third of pitch's inertia, so at 60 rad/s the
    #: gain is high enough that 12 degrees of heading error saturates the
    #: knob, and the loop spends the whole flight bang-bang -- visibly so,
    #: 100% saturated for stretches while the heading swings plus or minus
    #: thirteen. That looked like an obvious thing to fix by giving yaw its
    #: own, lower bandwidth. Measured over three stroke amplitudes it is not:
    #:
    #: ======  ==============  ========
    #: rad/s   saturates at    median
    #: ======  ==============  ========
    #: 20        107.9 deg      1206 ms
    #: 25         69.0 deg      1154 ms
    #: 30         47.9 deg      1090 ms
    #: 40         27.0 deg      1042 ms
    #: 60         12.0 deg      1218 ms
    #: ======  ==============  ========
    #:
    #: Non-monotonic, and the shared 60 is already at the top of it. At low
    #: gain the loop stops saturating and starts drifting instead -- the
    #: heading wanders to 142 degrees at bandwidth 20 against 106 at 60 -- and
    #: flies no longer for it. Saturation is real and is not what limits this.
    yaw_gain: float = YAW_INERTIA * BANDWIDTH**2 / abs(YAW_PER_PHASE)
    yaw_rate_gain: float = YAW_INERTIA * 2 * BANDWIDTH / abs(YAW_PER_PHASE)
    #: Feed the yaw knob's roll cross-coupling forward into the roll knob,
    #: from :data:`ROLL_FROM_PHASE`, instead of leaving the roll loop to
    #: discover it as a disturbance.
    #:
    #: **On, and it was expected not to matter.** At full deflection the
    #: coupling is 0.039 of amplitude asymmetry against a limit of 0.45, which
    #: looked far too small to be worth cancelling -- the roll loop has ten
    #: times the authority it needs.
    #:
    #: It helps, at every stroke amplitude tried, and by an amount this cannot
    #: pin down. Off gives 1218/1221/1097 ms; on gives 1473/1230/1250, a
    #: median of 1218 to 1250. An earlier version of the compensation, using a
    #: fit that was wrong at small deflections, gave 1491/1377/1129 -- median
    #: 1377. **Both are positive at all three amplitudes and they disagree by
    #: more than the effect**, so what is claimed here is the sign and not the
    #: size. The accurate curve is kept because it is the one that is right,
    #: not because it measured better.
    #:
    #: The mechanism it is betting on is timing rather than magnitude: the yaw
    #: knob saturates for long stretches, so the roll it drags arrives as a
    #: step the roll loop can only answer once its own filter has seen it, and
    #: a known disturbance fed forward skips that delay for free.
    compensate_yaw_roll: bool = True
    #: Heading to hold, radians, accumulated rather than wrapped so that a
    #: turn past half a circle is a large error and not a small one of the
    #: wrong sign.
    target_heading: float = 0.0
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
    #: **Only read when ``sensing`` is ``"lowpass"``**, which is no longer the
    #: default. It is kept because every result before the boxcar was measured
    #: through it, and those should stay reproducible.
    #:
    #: The direction this identified is right and still is: the loop sits near
    #: its phase margin, and everything inside it that adds delay costs flight
    #: time. It is why the integral term made things slightly worse and the
    #: circulation lag in :mod:`wingloop.aero.wake` made them much worse.
    #:
    #: The filter cannot go to zero -- within a stroke the torque about the
    #: centre of mass swings about a near-zero mean, so an unfiltered loop
    #: chases the beat. It trades stroke noise against phase margin, and where
    #: that trade sits **belongs to the stroke, not to the loop**: 5-6 ms on
    #: the sinusoid, 3.5 ms on the sharper sweep this animal now flies, which
    #: carries its torque differently and wants less filtering. At 1 ms the
    #: flight is 116 ms and at 20 it is 21.
    #:
    #: 5 ms was also once quoted as worth 353 ms of flight against 187 at 20,
    #: and that was a spike which moved when the stroke amplitude moved. See
    #: :data:`SENSING`. The 3.5 ms peak does not move with amplitude, which is
    #: the difference between the two.
    #:
    #: None of this is the default path. A boxcar over one wingbeat beats
    #: every low-pass setting tried -- 1162 ms at bandwidth 40 against the best
    #: low-pass's 932 -- which is why it is what the loop actually reads.
    tau: float = 0.005
    #: How the sensed attitude and rate are cleaned up before the loop reads
    #: them. ``"lowpass"`` is the first-order filter :attr:`tau` sets, which
    #: every result before this was measured on. ``"stroke"`` is a running
    #: mean over exactly one wingbeat.
    #:
    #: **The reason to prefer the second is not tuning, it is where the zeros
    #: are.** A boxcar of exactly one period has an exact null at the stroke
    #: frequency *and at every harmonic of it*, which is precisely the
    #: disturbance the filter is there to reject -- and its group delay is half
    #: a period, 2.29 ms at 218 Hz, against the 5 ms of the first-order lag it
    #: replaces. Better rejection of the one thing that needs rejecting, for
    #: less than half the phase cost. What that is worth is measured in
    #: :data:`SENSING`, not asserted here.
    sensing: str = "stroke"

    #: Stroke shape, passed through to :func:`harmonic_stroke`. ``sharpness``
    #: bends the sweep from a sinusoid toward a triangle and ``deviation``
    #: adds the out-of-plane motion that makes a wingtip trace a
    #: figure-of-eight.
    #:
    #: **0.9, and it took three corrections to get here.** A real stroke sweeps
    #: at a flatter speed with the turnaround compressed into the reversals,
    #: and that cuts the within-stroke torque swing by 20%. This project
    #: predicted that would buy flight, measured a ninefold *loss*, and wrote
    #: the hypothesis up as refuted. The loss shrank every time the loop's
    #: sensing improved -- nine at 20 ms of filter lag, 1.5 at 5 ms, 1.25 under
    #: the stroke boxcar -- and reversed outright once the wings were told the
    #: body rotates:
    #:
    #: ======  =======  =======  ======
    #: amp     sine     sharp    ratio
    #: ======  =======  =======  ======
    #: 74.25    1473     2467     1.68
    #: 75.00    1230     2123     1.73
    #: 75.75    1250     1879     1.50
    #: ======  =======  =======  ======
    #:
    #: What it costs is lift: 12.19 against 13.66, from 1.36 of body weight
    #: down to 1.21. Still more than enough to fly, and the amplitude that
    #: would restore it exactly is 79.41 degrees -- not taken, because the
    #: table above was measured at 75 and raising the amplitude is a different
    #: change with its own measurements owing.
    #:
    #: The pitch trim moves with the shape, and barely: the torque about the
    #: centre of mass vanishes at -10.95 degrees of bias here against -10.67
    #: for the sinusoid, so :data:`TRIM_BIAS` is 0.25 degrees out and stays.
    sharpness: float = 0.9
    deviation: float = 0.0
    deviation_phase: float = 0.0
    #: Integral gains, per second. **The roll one is on now, and the story of
    #: why is worth more than the number.**
    #:
    #: Proportional-derivative alone leaves a steady-state offset against a
    #: constant disturbance, and there is one: the pitch trim cross-couples
    #: +0.6 into roll, so the roll loop settles several degrees off level and
    #: stays there without ever saturating.
    #:
    #: That was measured once and dismissed -- at gain 4 the animal held
    #: attitude for 179 ms against 187 without, slightly worse -- and written
    #: up as *the standing roll offset is not what ends the flight*. True at
    #: the time. A tumble ended the flight at 187 ms and a few degrees of roll
    #: had nothing to do with it.
    #:
    #: It became false without being re-measured. Once the yaw loop pushed
    #: flight past a second, the standing roll started to matter through a
    #: route that did not exist before: **roll and climb together make yaw**.
    #: Rolled 5 degrees while still is 0.0008 of yaw torque and climbing while
    #: level is 0.0000, but rolled and climbing is 0.2362 -- sideslip -- and
    #: that standing torque was consuming over half the yaw knob. Nulling the
    #: roll takes the mean roll from 7.94 degrees to 0.17 and the standing
    #: phase deflection from -22.5 degrees to +4.3.
    #:
    #: The gain is 1.5 because it is the one that helps at *every* stroke
    #: amplitude tried. Medians over three amplitudes: 1069 ms at gain 0,
    #: 1154 at 1.0, **1218 at 1.5**, 1360 at 2.0, 1146 at 3.0. Gain 2.0 has
    #: the better median and its worst amplitude is 1044, barely above
    #: baseline; 1.5 improves all three and has the best worst case. The
    #: spread between amplitudes is comparable to the effect, so this is worth
    #: about 15% and not more -- nothing like the tripling that closing the
    #: yaw loop gave.
    #:
    #: The pitch integral stays off: added on top it makes things worse, 1204
    #: ms against 1382 on the same run.
    pitch_integral_gain: float = 0.0
    roll_integral_gain: float = 1.5
    #: Bound on the accumulated terms, in the units of each knob, so a period
    #: of saturation or a tumble cannot wind them up into a command that
    #: outlives the error that produced it.
    max_integral: float = 0.25

    _state: dict = field(default_factory=dict)
    _turn: dict = field(default_factory=dict)
    #: Where a run stopped because the simulation diverged, seconds, or None.
    diverged_at: float | None = None
    #: An optional stroke generator. Given one -- a
    #: :class:`~wingloop.body.power.PowerStroke` -- the sweep is produced by a
    #: muscle model rather than written down, and ``amplitude`` and
    #: ``frequency`` stop being inputs to it. The loop does not care which:
    #: it works the same two knobs either way.
    stroke: object = None
    #: An optional :class:`Throttle`. Given one, how hard the animal flies is
    #: no longer an argument either: the altitude loop sets the oscillator's
    #: drive every step from the height error. Attitude and altitude are then
    #: two loops on the same wings, which is the arrangement the animal has.
    throttle: object = None

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

    def _filtered(self, pitch, roll, yaw, rate, dt):
        if self.sensing == "stroke":
            return self._stroke_averaged(pitch, roll, yaw, rate, dt)
        a = dt / (self.tau + dt)
        if not self._state:
            self._state = {
                "pitch": pitch,
                "roll": roll,
                "yaw": yaw,
                "rate": np.asarray(rate, float),
            }
        else:
            self._state["pitch"] += a * (pitch - self._state["pitch"])
            self._state["roll"] += a * (roll - self._state["roll"])
            self._state["yaw"] += a * (yaw - self._state["yaw"])
            self._state["rate"] += a * (np.asarray(rate, float) - self._state["rate"])
        return (
            self._state["pitch"],
            self._state["roll"],
            self._state["yaw"],
            self._state["rate"],
        )

    def _turned(self, body) -> float:
        """Heading, accumulated across the wrap.

        Kept apart from the filter state because it has to be unwrapped
        *before* anything averages it: a mean taken across the +pi/-pi seam is
        not a heading, and with the loop tuned this animal can be carried past
        half a turn inside 100 ms.
        """
        raw = heading(body)
        if "raw" in self._turn:
            step = (raw - self._turn["raw"] + np.pi) % (2 * np.pi) - np.pi
            self._turn["total"] += step
        else:
            self._turn["total"] = raw
        self._turn["raw"] = raw
        return self._turn["total"]

    def _stroke_averaged(self, pitch, roll, yaw, rate, dt):
        """A running mean over exactly one wingbeat, held in a ring buffer.

        Kept as a running mean rather than one sample per beat on purpose: the
        nulls are the same either way, and holding the command for a whole
        cycle would add another half period of delay for nothing.
        """
        n = max(1, int(round(1.0 / (self.frequency * dt))))
        if not self._state:
            self._state = {
                "ring": np.zeros((n, 6)),
                "sum": np.zeros(6),
                "i": 0,
                "filled": 0,
            }
        st = self._state
        sample = np.array([pitch, roll, yaw, *np.asarray(rate, float)[:3]])
        st["sum"] += sample - st["ring"][st["i"]]
        st["ring"][st["i"]] = sample
        st["i"] = (st["i"] + 1) % n
        st["filled"] = min(st["filled"] + 1, n)
        mean = st["sum"] / st["filled"]
        return float(mean[0]), float(mean[1]), float(mean[2]), mean[3:6]

    def knobs(self, body) -> dict:
        """The three stroke knobs the stabiliser wants this instant.

        Called **once per step**, because it advances the sensor filter and
        the heading unwrap. An earlier version had the steering controller
        recompute this after calling it, which ran the filter twice a step and
        quietly halved the time constant that everything else here argues
        about.
        """
        dt = float(body.model.opt.timestep)
        if self.throttle is not None:
            self.throttle.update(body, dt)
        pitch, roll = attitude(body)
        pitch, roll, yaw, rate = self._filtered(
            pitch, roll, self._turned(body), angular_rate(body), dt
        )
        # Pitch authority is negative -- more bias is more nose-down torque --
        # so correcting a positive pitch means *more* bias, not less. Yaw is
        # the same sign story through the phase knob.
        i_pitch, i_roll = self._integrate(pitch, roll, dt)
        bias = (
            self.trim_bias
            + self.pitch_gain * pitch
            + self.pitch_rate_gain * rate[1]
            + i_pitch
        )
        asymmetry = -self.roll_gain * roll - self.roll_rate_gain * rate[0] - i_roll
        phase_asymmetry = self.yaw_gain * (
            yaw - self.target_heading
        ) + self.yaw_rate_gain * rate[2]
        if self.compensate_yaw_roll:
            # Cancel the roll the yaw knob is about to make, from the measured
            # curve, before the roll loop has to see it as a disturbance.
            p = float(np.clip(phase_asymmetry, -MAX_PHASE, MAX_PHASE))
            asymmetry -= (
                float(np.interp(p, *ROLL_FROM_PHASE)) / ROLL_PER_ASYMMETRY
            )
        return {
            "bias": float(np.clip(bias, -self.max_bias, self.max_bias)),
            "asymmetry": float(
                np.clip(asymmetry, -self.max_asymmetry, self.max_asymmetry)
            ),
            "phase_asymmetry": float(np.clip(phase_asymmetry, -MAX_PHASE, MAX_PHASE)),
        }

    def command(self, body, t: float):
        """Stroke angles and rates for this instant, with the loop closed."""
        return self._stroke(t, body, **self.knobs(body))

    def _stroke(self, t, body, **knobs):
        """The stroke, from a generator when there is one and a sine otherwise."""
        if self.stroke is not None:
            return self.stroke(float(body.model.opt.timestep), **knobs)
        return harmonic_stroke(
            t,
            amplitude=self.amplitude,
            frequency=self.frequency,
            **knobs,
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
            for k in (
                "t", "x", "y", "z", "pitch", "roll", "tumble", "bearing",
                "heading", "throttle",
            )
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
            # A free base carries x, y, z; a vertical rail carries only z, and
            # reading three numbers off a one-number qpos is an IndexError
            # rather than a wrong answer -- which is the better failure, but
            # still one worth not having.
            if body.root_translation == 3:
                x, y, z = (float(v) for v in body.data.qpos[body.root_dof : body.root_dof + 3])
            elif body.root_translation == 1:
                x, y, z = 0.0, 0.0, float(body.data.qpos[body.root_dof])
            else:
                x = y = z = 0.0
            out["t"].append(body.t)
            out["x"].append(x)
            out["y"].append(y)
            out["z"].append(z)
            out["pitch"].append(pitch)
            out["roll"].append(roll)
            out["tumble"].append(float(np.linalg.norm(angular_rate(body))))
            out["bearing"].append(float(getattr(self, "bearing", 0.0)))
            out["throttle"].append(
                float(self.throttle.last) if self.throttle is not None else float("nan")
            )
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
    #: Commanded turn rate per unit of steering command, rad/s.
    #:
    #: **This is what the steering command means once yaw is stabilised.** With
    #: the yaw loop closed, adding the command to the phase knob no longer
    #: turns the animal: the stabiliser is holding a heading and simply undoes
    #: it, measured at -14.6 degrees by 90 ms and back to -1.3 by 250. So the
    #: command moves the *setpoint* instead -- bearing error into turn rate,
    #: which is the fixation law -- and the yaw loop flies it.
    #:
    #: Set to 0.0 to get the old behaviour, where the command is added to the
    #: knob directly. That is what every steering result before the yaw loop
    #: was measured with, and it only turns because nothing was holding the
    #: heading.
    turn_rate_gain: float = 20.0
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
        """The stabiliser's knobs with the steering command added to one.

        Which one is the whole point of ``steer_mode``: a steering intention
        and a yaw disturbance arrive at the same knob and the loop cannot tell
        them apart, exactly as the roll loop cannot tell a roll disturbance
        from a commanded turn. That is the animal's arrangement too.
        """
        extra = self.steering(body)
        # The setpoint moves before the knobs are computed, so the yaw loop
        # sees this step's intention rather than the previous one's.
        if extra and self.turn_rate_gain and self.yaw_gain:
            self.target_heading -= (
                self.turn_rate_gain * extra * float(body.model.opt.timestep)
            )
            # And it is not allowed to run away from the animal. Without this
            # the setpoint ramps at the commanded rate whatever the body does,
            # the yaw error saturates the phase knob, and the roll the knob
            # cross-couples takes the flight down -- measured, a sustained
            # command at gain 55 ended it at 138 ms instead of flying. Holding
            # the setpoint a bounded lead ahead makes the turn rate whatever
            # the loop can actually deliver, which is the point of having one.
            lead = MAX_PHASE / self.yaw_gain
            here = self._turn.get("total", 0.0)
            self.target_heading = float(
                np.clip(self.target_heading, here - lead, here + lead)
            )
        knobs = self.knobs(body)
        if extra and not (self.turn_rate_gain and self.yaw_gain):
            if self.steer_mode == "phase":
                knobs["phase_asymmetry"] = float(
                    np.clip(
                        knobs["phase_asymmetry"] + self.phase_gain * extra,
                        -MAX_PHASE,
                        MAX_PHASE,
                    )
                )
            else:
                knobs["asymmetry"] = float(
                    np.clip(
                        knobs["asymmetry"] + extra,
                        -self.max_asymmetry,
                        self.max_asymmetry,
                    )
                )
        return self._stroke(t, body, **knobs)


@dataclass
class Throttle:
    """The altitude loop: how high the animal is, back onto how hard it flies.

    Everything above this closed a loop around *attitude* -- which way the
    animal points -- and left how hard it flaps as a number someone chose.
    That is an animal that can stay upright while it sinks. This closes the
    other loop, through the channel the connectome supplies: height error into
    a flight command, the command into power motor neuron activity through
    :class:`~wingloop.brain.readout.PowerReadout`, the activity into oscillator
    drive, and the drive into stroke amplitude.

    The law is the same shape as the attitude one and its gains come from the
    same place -- a measured authority, not a knob::

        command = hover - (w^2 (z - z*) + 2 zeta w zdot) / CLIMB_PER_COMMAND

    so ``bandwidth`` and ``damping`` are the only choices, and both are
    argued for where they are defined.

    **The filter turned out to matter much less than the attitude loop's**,
    which is not what was written here first. Holding zero on the rail for
    600 ms, against the time constant:

    ======  ============  ===============
    tau     command swing altitude error
    ======  ============  ===============
    0 ms       0.0128        0.126 mm
    1 ms       0.0044        0.028 mm
    5 ms       0.0011        0.054 mm
    20 ms      0.0007        0.064 mm
    50 ms      0.0550        0.910 mm
    ======  ============  ===============

    Unfiltered the loop still holds height to a tenth of a millimetre and puts
    1.3% of stroke ripple into the command: worth removing, not what the loop
    stands or falls on. The end that bites is the other one, and it is the
    familiar one -- at 50 ms the lag has eaten the phase margin and both the
    ripple and the error jump by an order of magnitude. Why heave tolerates
    what pitch does not is **not established here**: the obvious guess, that
    the body's mass integrates the stroke ripple away before the loop sees it,
    is wrong -- measured against its own mean, heave velocity ripples *more*
    than pitch rate does (3.5 against 1.7). The number is kept at the attitude
    loop's 5 ms because nothing measured argues for moving it.
    """

    #: The connectome channel this loop acts through.
    readout: object
    #: The muscle it sets the drive of.
    oscillator: object
    #: Height to hold, millimetres, in whatever frame the body reports.
    target: float = 0.0
    bandwidth: float = ALTITUDE_BANDWIDTH
    damping: float = ALTITUDE_DAMPING
    tau: float = 0.005
    hover: float = HOVER_COMMAND
    authority: float = CLIMB_PER_COMMAND
    #: Hold this command and ignore the height instead of closing the loop.
    #: The open-loop control the closed-loop results are measured against.
    held: float | None = None
    #: The command most recently issued, for the trace to record.
    last: float = float("nan")

    _state: dict = field(default_factory=dict)

    def _filtered(self, z, w, dt):
        a = dt / (self.tau + dt)
        if not self._state:
            self._state = {"z": z, "w": w}
        else:
            self._state["z"] += a * (z - self._state["z"])
            self._state["w"] += a * (w - self._state["w"])
        return self._state["z"], self._state["w"]

    def command(self, z: float, w: float, dt: float) -> float:
        """Flight command for this height and climb rate, in [0, 1]."""
        if self.held is not None:
            return float(np.clip(self.held, 0.0, 1.0))
        z, w = self._filtered(z, w, dt)
        wanted = -(
            self.bandwidth**2 * (z - self.target)
            + 2.0 * self.damping * self.bandwidth * w
        )
        return float(np.clip(self.hover + wanted / self.authority, 0.0, 1.0))

    def update(self, body, dt: float) -> float:
        """Read the body, set the muscle's drive, and report the command.

        Called once per step and only once: :meth:`command` advances the
        filter, so a second call for the sake of recording the number would
        halve the time constant that the docstring above argues for. The
        command is kept in :attr:`last` for whoever wants to look at it.
        """
        z, w = altitude(body)
        self.last = self.command(z, w, dt)
        self.oscillator.drive = self.readout.drive(self.last)
        return self.last
