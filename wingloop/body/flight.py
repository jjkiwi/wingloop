"""A fly with wings that move and air to move them against.

This is where the force model stops being a function and starts being physics.
MuJoCo integrates the body; every step, the wings' own kinematics are read out
of the joints, handed to the blade-element model, and the resulting forces are
written back with :func:`mujoco.mj_applyFT`.

**Why the forces are applied rather than simulated.** MuJoCo's fluid model is
ellipsoid drag in a uniform flow. A flapping wing at Re ~ 150 gets most of its
lift from an attached leading-edge vortex, a rotational circulation term and
entrained air -- none of which that model contains. Setting a fluid density
here would add a force that is wrong in magnitude and wrong in phase, so the
simulation stays in vacuum and the aerodynamics arrive as applied forces.

**Where they are applied.** At the centre of pressure, which sits at
``moment(3) / moment(2)`` along the span -- about 0.7 of wing length, because
force grows as radius squared. Applying at the body's centre of mass instead
would lose the roll torque that makes asymmetric flapping steer the animal,
which is the entire point of putting this in a body.

**The frame.** Forces are computed in the wing's own frame and rotated into
the world by the wing body's orientation. The span runs along local **y** (see
``hinge.SPAN_LOCAL`` for how that was measured and what assuming z cost), the
stroke sweeps about local z, so an element at radius ``r`` moves along
``z_hat x span`` and the force normal to its path comes out along local z --
which at rest is world up. The direction is derived from the span vector by
cross product rather than written down, because the two wings are mirrored and
a hardcoded sign is right for one of them.

Reading the wing's velocity from MuJoCo instead would be more general and much
slower; the cost of this choice is that the model sees only the wing's own
sweep and not the body's motion through the air, so it describes hovering or a
tether.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..aero.blade_element import StrokeState, blade_element_forces
from ..aero.wing import AIR_DENSITY, Wing
from .hinge import HINGE_DOFS, SPAN_LOCAL, WINGS

_HINT = (
    "The flight body needs MuJoCo:\n    pip install mujoco\n"
    "Headless machines can still run the physics; set MUJOCO_GL=disable."
)


@dataclass
class WingTelemetry:
    """What one wing did on one step, for anyone plotting or debugging."""

    stroke: float = 0.0
    rotation: float = 0.0
    stroke_rate: float = 0.0
    rotation_rate: float = 0.0
    normal: float = 0.0
    drag: float = 0.0
    world_force: np.ndarray = field(default_factory=lambda: np.zeros(3))


class FlightBody:
    """NeuroMechFly with a hinge, in air, driven by wing angles.

    ``command`` is a mapping from joint name to target angle, which is what a
    stroke generator produces and what a descending readout will eventually
    set. Nothing here knows about neurons.
    """

    def __init__(
        self,
        model_path,
        wing: Wing,
        *,
        timestep: float = 2e-5,
        density: float = AIR_DENSITY,
        gravity: bool = True,
    ):
        try:
            import mujoco
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(_HINT) from exc
        self._mj = mujoco

        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.model.opt.timestep = timestep
        # Vacuum on purpose; see the module docstring.
        self.model.opt.density = 0.0
        self.model.opt.viscosity = 0.0
        if not gravity:
            self.model.opt.gravity[:] = 0.0
        self.data = mujoco.MjData(self.model)

        self.wing = wing
        self.density = density
        # Force acts at the radius that the r^2-weighted distribution centres
        # on, not at mid-span and not at the tip.
        self.cop = wing.moment(3) / wing.moment(2)
        self.span = {w: np.asarray(v, dtype=float) for w, v in SPAN_LOCAL.items()}
        # Which way an element travels when the stroke joint turns positively.
        self.sweep_dir = {
            w: np.cross([0.0, 0.0, 1.0], s) for w, s in self.span.items()
        }

        self.body_id = {w: self._id(mujoco.mjtObj.mjOBJ_BODY, w) for w in WINGS}
        self.joint_id = {
            w: {d: self._id(mujoco.mjtObj.mjOBJ_JOINT, f"joint_{w}_{d}") for d in HINGE_DOFS}
            for w in WINGS
        }
        self.actuator_id = {
            f"joint_{w}_{d}": self._id(
                mujoco.mjtObj.mjOBJ_ACTUATOR, f"actuator_position_joint_{w}_{d}"
            )
            for w in WINGS
            for d in HINGE_DOFS
        }
        self.telemetry: dict[str, WingTelemetry] = {w: WingTelemetry() for w in WINGS}
        mujoco.mj_forward(self.model, self.data)

    def _id(self, kind, name: str) -> int:
        i = self._mj.mj_name2id(self.model, kind, name)
        if i < 0:
            raise KeyError(f"the model has no {kind} named {name!r}; was it hinged?")
        return i

    # ------------------------------------------------------------- kinematics

    def angles(self, wing: str) -> dict[str, float]:
        return {
            d: float(self.data.qpos[self.model.jnt_qposadr[j]])
            for d, j in self.joint_id[wing].items()
        }

    def rates(self, wing: str) -> dict[str, float]:
        return {
            d: float(self.data.qvel[self.model.jnt_dofadr[j]])
            for d, j in self.joint_id[wing].items()
        }

    # ------------------------------------------------------------------ forces

    def wing_forces(self, wing: str) -> np.ndarray:
        """The world-frame force on one wing, and the telemetry behind it.

        The sign convention matters and is easy to get backwards: drag opposes
        the wing's motion, so it flips with the stroke direction, while the
        normal force does not -- a fly that flips its angle of attack at
        reversal gets lift on both half-strokes, and a model where the normal
        force flipped too would hover by accident on the average of nothing.
        """
        a, v = self.angles(wing), self.rates(wing)
        # The coefficients are fits over angle of attack from zero to ninety.
        # Feed them a negative angle -- which the upstroke always does, since
        # the wing flips -- and CL comes back negative: at -45 degrees it
        # returns -1.31 instead of +1.31, so the upstroke pushes the animal
        # into the ground. Incidence is symmetric about zero; the sign lives in
        # which face the flow meets, not in the coefficient.
        state = StrokeState(
            phi=a["stroke"],
            alpha=abs(a["rotation"]),
            phi_dot=v["stroke"],
            alpha_dot=v["rotation"],
        )
        f = blade_element_forces(self.wing, state, density=self.density)

        # Resolved in the stroke plane, not in the wing's own frame: lift is by
        # definition perpendicular to the wing's path and drag opposes it, and
        # for a horizontal stroke plane that makes lift vertical throughout.
        # Rotating a wing-frame normal into the world instead ties the force
        # direction to the pitch angle twice over -- once in the coefficient and
        # once in the frame -- and the two cancel over a cycle.
        rot = self.data.xmat[self.body_id[wing]].reshape(3, 3)
        path = rot @ self.sweep_dir[wing]
        path[2] = 0.0
        norm = np.linalg.norm(path)
        path = path / norm if norm > 1e-12 else np.zeros(3)
        world = np.array([0.0, 0.0, f.total_normal], dtype=float)
        world = world - np.sign(state.phi_dot) * f.drag * path

        t = self.telemetry[wing]
        t.stroke, t.rotation = state.phi, state.alpha
        t.stroke_rate, t.rotation_rate = state.phi_dot, state.alpha_dot
        t.normal, t.drag = f.total_normal, f.drag
        t.world_force = world
        return world

    def apply_aerodynamics(self) -> None:
        """Write both wings' forces into the simulation for this step.

        Through ``qfrc_applied`` rather than ``xfrc_applied``: ``mj_applyFT``
        converts a Cartesian force applied at a point into generalised forces,
        which is what carries the moment arm from the centre of pressure into
        the hinge and the body. Writing a force into ``xfrc_applied`` instead
        would apply it at the wing's centre of mass and silently discard that
        arm. Both accumulate, so the buffer is cleared each step.
        """
        self.data.qfrc_applied[:] = 0.0
        for wing in WINGS:
            force = self.wing_forces(wing)
            bid = self.body_id[wing]
            rot = self.data.xmat[bid].reshape(3, 3)
            # Centre of pressure, out along the wing's own span axis.
            point = self.data.xpos[bid] + rot @ (self.cop * self.span[wing])
            self._mj.mj_applyFT(
                self.model,
                self.data,
                force,
                np.zeros(3),
                point,
                bid,
                self.data.qfrc_applied,
            )

    # -------------------------------------------------------------- stepping

    def command(self, targets: dict[str, float]) -> None:
        """Set position targets for wing joints, by joint name."""
        for name, value in targets.items():
            self.data.ctrl[self.actuator_id[name]] = float(value)

    def prescribe(
        self, angles: dict[str, float], rates: dict[str, float] | None = None
    ) -> None:
        """Drive the wing joints kinematically instead of through the servo.

        The aerodynamic load on a wing is not small: at a full 218 Hz stroke the
        drag on one wing peaks near four times the animal's body weight, and a
        position servo stiff enough to win that fight is stiff enough to need a
        timestep nobody wants. Worse, when the servo loses, the joint's rate
        disagrees in sign with the command, the drag term flips with it, and the
        simulation diverges -- which is what the first version did, at DOF 3,
        within half a millisecond.

        So the stroke is imposed and the forces are read off it. That is the
        standard arrangement for validating a blade-element model, and it is
        honest about what is being tested: whether these kinematics produce
        these forces in this body, not whether a muscle could drive them. The
        muscle model is a later problem, and it is the one that needs the
        power numbers this arrangement produces.
        """
        for name, value in angles.items():
            jid = self._joint_by_name(name)
            self.data.qpos[self.model.jnt_qposadr[jid]] = float(value)
            if rates is not None and name in rates:
                self.data.qvel[self.model.jnt_dofadr[jid]] = float(rates[name])
        self._mj.mj_forward(self.model, self.data)

    def _joint_by_name(self, name: str) -> int:
        wing, dof = name.replace("joint_", "").split("_")
        return self.joint_id[wing][dof]

    def step(self, targets: dict[str, float] | None = None) -> None:
        if targets:
            self.command(targets)
        self.apply_aerodynamics()
        self._mj.mj_step(self.model, self.data)

    @property
    def t(self) -> float:
        return float(self.data.time)

    def total_applied_force(self) -> np.ndarray:
        """Summed world force from both wings, for checking against weight."""
        return np.sum([self.telemetry[w].world_force for w in WINGS], axis=0)


#: How abruptly the idealised stroke flips the wing at reversal. Larger is
#: closer to a square wave and to what a fly does; too large and the rotational
#: force term, which scales with the rotation rate, dominates the stroke.
SHARPNESS = 3.0


def harmonic_stroke(
    t: float,
    *,
    amplitude: float = np.deg2rad(75.0),
    frequency: float = 218.0,
    alpha: float = np.deg2rad(45.0),
    rates: bool = False,
):
    """A textbook stroke: harmonic sweep, angle of attack flipped at reversal.

    Not measured kinematics and not what a fly does in detail -- real strokes
    are closer to a sawtooth in position with rotation concentrated at the
    ends. It exists so the body can be driven with something before a stroke
    generator exists, and so the force model can be exercised in a body at all.

    The two wings are mirrored: the same stroke angle on both sides means the
    same sweep in body coordinates, because the wing frames are mirrored in the
    model.
    """
    w = 2.0 * np.pi * frequency
    phi = amplitude * np.sin(w * t)
    # Pitch the wing one way on the downstroke and the other on the upstroke.
    # Smoothed rather than a sign flip: a real wing takes a finite time to
    # rotate, and a step here is not merely unrealistic but unsimulable -- the
    # rotational force term is proportional to alpha_dot, so an instantaneous
    # flip is an infinite force. The first version of this used np.sign and
    # MuJoCo answered with "Nan, Inf or huge value in QACC".
    rot = alpha * np.tanh(SHARPNESS * np.cos(w * t)) / np.tanh(SHARPNESS)
    out = {}
    for wing in WINGS:
        out[f"joint_{wing}_stroke"] = phi
        out[f"joint_{wing}_rotation"] = rot
        out[f"joint_{wing}_deviation"] = 0.0
    if not rates:
        return out

    # Analytic derivatives, so prescribed kinematics carry exact rates rather
    # than a finite difference that lags by half a step -- the rotational force
    # term is proportional to one of them.
    phi_dot = amplitude * w * np.cos(w * t)
    c = np.cos(w * t)
    rot_dot = (
        alpha * SHARPNESS * (1.0 - np.tanh(SHARPNESS * c) ** 2) * (-w * np.sin(w * t))
    ) / np.tanh(SHARPNESS)
    drates = {}
    for wing in WINGS:
        drates[f"joint_{wing}_stroke"] = phi_dot
        drates[f"joint_{wing}_rotation"] = rot_dot
        drates[f"joint_{wing}_deviation"] = 0.0
    return out, drates
