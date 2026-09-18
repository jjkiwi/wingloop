"""Quasi-steady flapping-wing aerodynamics, the reason this is a new project.

MuJoCo has no aerodynamics worth the name. It can be given a fluid density and
will then apply ellipsoid drag, which is the wrong model twice over: a flapping
wing spends its stroke at angles of attack where an aeroplane would have
stalled, and it generates most of its lift from effects that do not exist in a
steady flow at all. A fly held up by ellipsoid drag would need a wing it does
not have.

So the forces are computed here and written into the simulator as applied
forces. Three terms, which is the standard quasi-steady decomposition:

**Translational.** Lift and drag from the wing's velocity through the air, with
coefficients that are functions of angle of attack measured on real
*Drosophila* wings rather than derived from thin-aerofoil theory. The wing does
not stall because the leading-edge vortex stays attached at the Reynolds number
it flies at (~150), which is why the lift coefficient keeps climbing past 40
degrees where an aerofoil's would collapse.

**Rotational.** When the wing rotates about its spanwise axis while translating
-- which it does at every stroke reversal -- it sheds circulation and gains a
force proportional to the product of the two rates. This is the Kramer effect,
and it is what makes the timing of wing rotation control the force, hence what
makes a fly steer by rotating one wing earlier than the other.

**Added mass.** Accelerating a wing accelerates the air around it. At this
scale the entrained air is not negligible compared with the wing's own mass,
and it matters most exactly where the rotational term does: at reversal.

**The coefficients are relayed, not verified.** The functional forms and
constants below are the standard empirical fits for *Drosophila*; they are
recorded in ``docs/LITERATURE.md`` as relayed until somebody checks them
against the papers. :func:`hover_check` is the guard that matters in the
meantime: a model whose numbers are wrong will not hold the animal up, and that
is a test rather than a citation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .wing import AIR_DENSITY, Wing

#: Phase offsets of the published fits. The source states them in degrees, so
#: they are converted here rather than written out as rounded radians -- an
#: earlier version carried 0.1256 and 0.1714, which is a 0.4% error in the
#: coefficient at low incidence and exactly the kind of drift
#: :func:`test_published_coefficients_are_unchanged` exists to catch.
LIFT_PHASE = np.deg2rad(7.2)
DRAG_PHASE = np.deg2rad(9.82)


def lift_coefficient(alpha: np.ndarray | float) -> np.ndarray:
    """Translational lift coefficient against angle of attack, radians.

    Peaks near 45 degrees rather than stalling near 15, which is the whole
    difference between a wing at Re ~ 150 with an attached leading-edge vortex
    and an aerofoil.
    """
    a = np.asarray(alpha, dtype=float)
    return 0.225 + 1.58 * np.sin(2.13 * a - LIFT_PHASE)


def drag_coefficient(alpha: np.ndarray | float) -> np.ndarray:
    """Translational drag coefficient against angle of attack, radians.

    Minimum near zero incidence and largest edge-on-to-flow at 90 degrees,
    where it exceeds the lift coefficient -- a flapping wing pays heavily in
    drag, which is why the power muscles are the biggest in the animal.
    """
    a = np.asarray(alpha, dtype=float)
    return 1.92 - 1.55 * np.cos(2.04 * a - DRAG_PHASE)


@dataclass
class StrokeState:
    """The wing's instantaneous kinematics, in the stroke plane.

    ``phi`` sweeps the wing fore and aft, ``alpha`` is its angle of attack, and
    the dotted quantities are their rates. This is the minimal state a
    quasi-steady model needs, and it is exactly what a wing-hinge joint in
    MuJoCo can report.
    """

    phi: float = 0.0  # stroke position, rad
    alpha: float = 0.0  # angle of attack, rad
    phi_dot: float = 0.0  # rad/s
    alpha_dot: float = 0.0  # rad/s
    phi_ddot: float = 0.0  # rad/s^2


@dataclass
class Forces:
    """What one wing produced over one instant."""

    lift: float  # normal to the wing's path, positive up
    drag: float  # magnitude of the along-path force, always positive
    rotational: float
    added_mass: float
    torque: float  # about the hinge, from the translational terms
    #: The along-path force as a *signed* component, positive along the
    #: direction a positive stroke rate sweeps the wing. Separate from ``drag``
    #: because once the body moves, different blade elements can travel in
    #: opposite directions within one stroke -- the inner wing going backwards
    #: while the animal flies forwards -- and a single magnitude cannot say
    #: which way the sum points.
    path_force: float = 0.0

    @property
    def total_normal(self) -> float:
        """Every term that pushes normal to the stroke plane."""
        return self.lift + self.rotational + self.added_mass


def blade_element_forces(
    wing: Wing,
    state: StrokeState,
    *,
    density: float = AIR_DENSITY,
    body_velocity: float = 0.0,
    rotational_coefficient: float = 1.55,
    include_rotational: bool = True,
    include_added_mass: bool = True,
) -> Forces:
    """Integrate the quasi-steady forces over the span, one element at a time.

    Each element sees a velocity ``r * phi_dot`` from the wing's own sweep, so
    the outer elements do nearly all the work -- force goes as the second
    moment of area, which is why a wing's shape matters through its moments and
    not its outline.

    ``rotational_coefficient`` is the Kramer coefficient, which depends on where
    along the chord the rotation axis sits; 1.55 corresponds to rotation about
    the quarter-chord.

    ``body_velocity`` is the animal's own speed through the air resolved along
    the wing's sweep direction, and it is what separates flight from hovering.
    Added to every element, it makes one half-stroke faster than the other --
    the asymmetry that produces net thrust, and the reason a hovering model
    cannot be asked what happens when the fly moves. It can exceed the sweep
    speed at the inner elements, which then travel backwards relative to the
    air, so the along-path force is summed per element with its own sign rather
    than taken from the stroke direction.
    """
    r = wing.stations
    c = wing.chords
    dr = wing.dr

    # Velocity of each element through the air: its own sweep, plus whatever
    # the animal is doing.
    u = r * state.phi_dot + body_velocity
    q = 0.5 * density * u**2  # dynamic pressure per unit area

    cl = float(lift_coefficient(state.alpha))
    cd = float(drag_coefficient(state.alpha))
    lift = float(np.sum(q * cl * c) * dr)
    drag = float(np.sum(q * cd * c) * dr)
    # Opposing each element's own motion, so the sum can cancel when parts of
    # the wing travel in opposite directions.
    path_force = float(-np.sum(q * cd * c * np.sign(u)) * dr)
    # Torque about the hinge: the same force weighted by its moment arm.
    torque = float(np.sum(q * cl * c * r) * dr)

    rot = 0.0
    if include_rotational:
        # Kramer: proportional to the product of translational and rotational
        # rate, so it vanishes mid-stroke and peaks at reversal.
        rot = float(
            rotational_coefficient
            * density
            * state.alpha_dot
            * np.sum(np.abs(u) * c**2) * dr
        )

    am = 0.0
    if include_added_mass:
        # The entrained air is a cylinder of diameter equal to the chord.
        am = float(
            0.25 * np.pi * density * state.phi_ddot * np.sum(c**2 * r) * dr
        )

    return Forces(
        lift=lift,
        drag=drag,
        rotational=rot,
        added_mass=am,
        torque=torque,
        path_force=path_force,
    )


def stroke_average_lift(
    wing: Wing,
    *,
    amplitude: float = np.deg2rad(75.0),
    frequency: float = 218.0,
    alpha_mid: float = np.deg2rad(45.0),
    density: float = AIR_DENSITY,
    samples: int = 512,
) -> float:
    """Mean vertical force over one complete stroke cycle.

    Harmonic sweep at a fixed mid-stroke angle of attack: the textbook
    idealisation, not measured kinematics. It exists so :func:`hover_check` has
    something to compare against body weight, and so a change to the force model
    can be seen as a number rather than as a fly that falls over.

    218 Hz is the wingbeat frequency of *Drosophila melanogaster*; amplitude
    near 75 degrees each way is a normal hovering stroke.
    """
    t = np.linspace(0.0, 1.0 / frequency, samples, endpoint=False)
    w = 2.0 * np.pi * frequency
    phi_dot = amplitude * w * np.cos(w * t)
    total = 0.0
    for pd_ in phi_dot:
        f = blade_element_forces(
            wing,
            StrokeState(alpha=alpha_mid, phi_dot=float(pd_)),
            density=density,
            include_rotational=False,
            include_added_mass=False,
        )
        # Both half-strokes push the same way when the wing flips its angle of
        # attack at reversal, which is what a hovering fly does.
        total += abs(f.lift)
    return float(total / samples)


def hover_check(
    wing: Wing, body_mass: float, *, density: float = AIR_DENSITY, gravity: float = 9810.0
) -> dict[str, float]:
    """Can two of these wings hold up a body of this mass?

    The only honest test of an aerodynamic model that was assembled from relayed
    coefficients and a unit system nobody has verified end to end. A real fly
    hovers, so a model of one that produces a tenth of its weight, or ten times
    it, is wrong somewhere -- and this says by how much rather than leaving it to
    be discovered when the simulated animal sinks through the floor.

    Returns the lift of two wings, the weight, and their ratio. A ratio near 1
    means the model is at least dimensionally sane; it does not mean the
    coefficients are right.
    """
    lift = 2.0 * stroke_average_lift(wing, density=density)
    weight = body_mass * gravity
    return {
        "lift": lift,
        "weight": weight,
        "ratio": float(lift / weight) if weight else float("inf"),
    }
