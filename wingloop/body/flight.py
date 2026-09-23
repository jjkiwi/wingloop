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

from ..aero.blade_element import Forces, StrokeState, blade_element_forces
from ..aero.wake import WakeMemory, reference_speed
from ..aero.wing import AIR_DENSITY, Wing
from .hinge import HINGE_DOFS, SPAN_LOCAL, STROKE_OFFSET, STROKE_SIGN, WINGS

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
        massless_wings: bool = True,
        wake_memory: bool = False,
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
        # Where along the span the force acts, for the rotational part of the
        # oncoming air. Same radius the wake model uses, and for the same
        # reason: the force integral is weighted by the second moment of area.
        self._gyration_radius = float(
            np.sqrt(self.wing.moment(2) / max(self.wing.area, 1e-12))
        )
        # Which way an element travels when the stroke joint turns positively.
        self.sweep_dir = {
            w: np.cross([0.0, 0.0, 1.0], s) for w, s in self.span.items()
        }

        self.body_id = {w: self._id(mujoco.mjtObj.mjOBJ_BODY, w) for w in WINGS}
        # Wing joints are optional: the pose is composed analytically, so the
        # simulated model does not need them and is better off without them.
        self.joint_id = {}
        for w in WINGS:
            found = {}
            for d in HINGE_DOFS:
                j = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, f"joint_{w}_{d}"
                )
                if j >= 0:
                    found[d] = j
            if found:
                self.joint_id[w] = found
        #: The stroke currently commanded, which is what the forces are read
        #: from. Set by :meth:`set_wings`.
        self.wing_angles: dict[str, float] = {}
        self.wing_rates: dict[str, float] = {}
        # Where the root's linear velocity lives, if the model has a free joint.
        # A free joint carries three translational DOFs first; a vertical
        # slide rig carries one, and reading three from it would pick up wing
        # angles as if they were body velocity.
        free = [j for j in range(self.model.njnt) if self.model.jnt_type[j] == 0]
        slide = [
            j
            for j in range(self.model.njnt)
            if self.model.jnt_type[j] == 2 and "slide_" in (self._name(j) or "")
        ]
        tether = [
            j
            for j in range(self.model.njnt)
            if self.model.jnt_type[j] == 3 and "hinge_" in (self._name(j) or "")
        ]
        if free:
            root_joint = free[0]
            self.root_dof, self.root_translation = int(self.model.jnt_dofadr[free[0]]), 3
        elif slide:
            root_joint = slide[0]
            self.root_dof, self.root_translation = int(self.model.jnt_dofadr[slide[0]]), 1
        elif tether:
            # A pitch tether: the animal rotates and does not translate, so it
            # has a root body for the wrench to act on and no position to read.
            root_joint = tether[0]
            self.root_dof, self.root_translation = (
                int(self.model.jnt_dofadr[tether[0]]),
                0,
            )
        else:
            root_joint = None
            self.root_dof, self.root_translation = None, 0
        # The body the aerodynamic wrench is applied to. Without a root joint
        # the animal is welded to the world and nothing can move it anyway.
        self.root_body = (
            int(self.model.jnt_bodyid[root_joint]) if root_joint is not None else None
        )
        self._capture_mounts()
        if massless_wings and self.joint_id:
            self._make_wings_massless()
        # Off by default: every earlier result in this project was measured
        # without it, and turning it on silently would invalidate them all.
        self.wake = (
            {w: WakeMemory(chord=wing.mean_chord) for w in WINGS} if wake_memory else None
        )
        self.telemetry: dict[str, WingTelemetry] = {w: WingTelemetry() for w in WINGS}
        mujoco.mj_forward(self.model, self.data)

    def _capture_mounts(self) -> None:
        """The wings' rest orientation relative to the body, kept once.

        With the pose computed rather than simulated, this is the only thing
        the physics model still has to tell us about the wings.
        """
        self._mj.mj_forward(self.model, self.data)
        root = self.data.xmat[self.root_body].reshape(3, 3) if self.root_body else np.eye(3)
        self.mount = {
            w: root.T @ self.data.xmat[self.body_id[w]].reshape(3, 3) for w in WINGS
        }
        self.hinge_local = {
            w: root.T @ (self.data.xpos[self.body_id[w]] - self.data.xpos[self.root_body or 0])
            for w in WINGS
        }

    @staticmethod
    def _rot(axis: int, angle: float) -> np.ndarray:
        c, s_ = np.cos(angle), np.sin(angle)
        if axis == 0:
            return np.array([[1, 0, 0], [0, c, -s_], [0, s_, c]])
        if axis == 1:
            return np.array([[c, 0, s_], [0, 1, 0], [-s_, 0, c]])
        return np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])

    def wing_pose(self, wing: str, angles: dict[str, float]):
        """World rotation and hinge position of one wing, computed not simulated.

        Prescribed kinematics and dynamic wing joints cannot both hold. A wing
        swept at 1792 rad/s carries Coriolis and centrifugal terms that the mass
        matrix couples straight into the body, so the body responds to
        accelerations that the next ``prescribe`` erases -- measured on the
        vertical rail as -13804 mm/s^2 where the force balance said +4280.
        Lightening the wings to break the coupling only makes their own joints
        singular; every scaling tried diverged within a step.

        So the wings leave the physics entirely. Their pose is composed here
        from the commanded angles about the axes ``hinge.HINGE_AXES`` names --
        stroke about z, deviation about x, rotation about y -- and the only
        thing the simulator still carries is a rigid body with a root joint.
        :func:`test_the_analytic_pose_matches_the_simulated_one` checks this
        composition against MuJoCo's own kinematics on a hinged model, so the
        shortcut is verified rather than assumed.
        """
        root_rot = (
            self.data.xmat[self.root_body].reshape(3, 3)
            if self.root_body is not None
            else np.eye(3)
        )
        local = (
            self.mount[wing]
            @ self._rot(2, angles.get(f"joint_{wing}_stroke", 0.0))
            @ self._rot(0, angles.get(f"joint_{wing}_deviation", 0.0))
            @ self._rot(1, angles.get(f"joint_{wing}_rotation", 0.0))
        )
        rot = root_rot @ local
        root_pos = (
            self.data.xpos[self.root_body]
            if self.root_body is not None
            else np.zeros(3)
        )
        return rot, root_pos + root_rot @ self.hinge_local[wing]

    def _make_wings_massless(self) -> None:
        """Take the wings out of the mass matrix, because their motion is given.

        Prescribed kinematics and dynamic wings cannot both be true. A wing
        swept at 1792 rad/s carries enormous Coriolis and centrifugal terms,
        and the mass matrix couples them straight into the body's degrees of
        freedom -- so the body responds to accelerations that the next
        ``prescribe`` erases. Measured on the vertical rail: applied force
        13.664, gravity 9.272, and a vertical acceleration of -13804 mm/s^2
        where the force balance says +4280.

        Zeroing the wings' mass makes the coupling vanish and leaves them as
        what this model treats them as: massless surfaces whose position is
        given and whose aerodynamic force is applied to the body. The physical
        cost is the wings' own inertial reaction, which is the standard
        approximation in flapping-flight models and a mild one here -- both
        wings together are 0.24% of body mass.

        It is a real approximation all the same. A model that needs the
        inertial power of the stroke, rather than the aerodynamic force it
        produces, must not use this.
        """
        for wing in WINGS:
            bid = self.body_id[wing]
            self.model.body_mass[bid] = 0.0
            self.model.body_inertia[bid] = 0.0
        # A joint with no inertia behind it is singular; armature gives the
        # solver something to invert without giving the body anything to feel.
        for wing in WINGS:
            for jid in self.joint_id[wing].values():
                self.model.dof_armature[self.model.jnt_dofadr[jid]] = 1e-9

    def _name(self, jid: int):
        return self._mj.mj_id2name(self.model, self._mj.mjtObj.mjOBJ_JOINT, jid)

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
        a = {
            d: self.wing_angles.get(f"joint_{wing}_{d}", 0.0) for d in ("stroke", "rotation")
        }
        v = {
            d: self.wing_rates.get(f"joint_{wing}_{d}", 0.0) for d in ("stroke", "rotation")
        }
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
        # The animal's own speed through the air, resolved along the direction
        # this wing sweeps. Zero on a tether; the whole difference between
        # hovering and flight once the body is free.
        rot, hinge = self.wing_pose(wing, self.wing_angles)
        # The stroke plane belongs to the animal, not to the world. Its normal
        # is the body's own vertical, so lift tilts when the body tilts --
        # which is most of how attitude couples back into the forces, and
        # without it a pitched fly still gets its full weight straight up and
        # the controller is steering something that cannot be steered.
        normal = self.body_axis()
        path = rot @ self.sweep_dir[wing]
        path = path - np.dot(path, normal) * normal
        norm = np.linalg.norm(path)
        path = path / norm if norm > 1e-12 else np.zeros(3)
        along = float(np.dot(self.air_velocity_at(wing, rot, hinge), path))

        f = blade_element_forces(
            self.wing, state, density=self.density, body_velocity=along
        )
        if self.wake is not None:
            # Only the translational terms lag. The rotational force and the
            # added mass are genuinely instantaneous -- they are responses to
            # what the wing is doing right now, not to circulation it has had
            # time to build.
            speed = reference_speed(self.wing, state.phi_dot, along)
            # Not `path`: that name already holds the unit vector the force is
            # applied along, a few lines up. Shadowing it multiplied a vector
            # by a scalar force and broadcast the result across all three axes,
            # which showed up as 364 of lift where the wing was making 27.
            lagged_lift, lagged_drag, lagged_path = self.wake[wing].update(
                f.lift, f.drag, f.path_force, speed, float(self.model.opt.timestep)
            )
            f = Forces(
                lift=lagged_lift,
                drag=lagged_drag,
                rotational=f.rotational,
                added_mass=f.added_mass,
                torque=f.torque,
                path_force=lagged_path,
            )

        # Resolved in the stroke plane, not in the wing's own frame: lift is by
        # definition perpendicular to the wing's path and drag opposes it, and
        # for a horizontal stroke plane that makes lift vertical throughout.
        # Rotating a wing-frame normal into the world instead ties the force
        # direction to the pitch angle twice over -- once in the coefficient and
        # once in the frame -- and the two cancel over a cycle.
        # path_force already carries its sign, summed per element, so the
        # stroke direction must not be applied a second time.
        world = f.total_normal * normal + f.path_force * path

        t = self.telemetry[wing]
        t.stroke, t.rotation = state.phi, state.alpha
        t.stroke_rate, t.rotation_rate = state.phi_dot, state.alpha_dot
        t.normal, t.drag = f.total_normal, f.drag
        t.world_force = world
        return world

    def apply_aerodynamics(self) -> None:
        """Write both wings' forces into the simulation as a wrench on the body.

        **Not through the wing bodies.** The obvious implementation resolves
        each wing's force at its centre of pressure with ``mj_applyFT``, which
        distributes it into every generalised coordinate it touches -- the wing
        joints included. With kinematics prescribed that is wrong twice over:
        the wing's inertia about its hinge is 1.4e-5, so a torque of order 20
        gives it 1e6 rad/s^2, and the next ``prescribe`` overwrites the result
        anyway, leaving the body holding the reaction to an acceleration that
        never happened. Measured on the vertical rail, the animal sank at three
        times gravity while the static force balance said it should climb.

        So the wings are treated as force generators attached to the body: the
        two forces are summed with their moments about the body's centre of
        mass and applied as a single wrench. This neglects the wings' own
        inertial reaction on the body, which is the standard approximation in
        flapping-flight models and a mild one here -- both wings together are
        0.24% of body mass.
        """
        self.data.qfrc_applied[:] = 0.0
        self.data.xfrc_applied[:] = 0.0
        # A model welded to the world has nothing to apply a wrench to, but it
        # is still a valid preparation -- the tether is where the force model
        # gets measured -- so the forces and telemetry are computed either way
        # and only the write is skipped.
        com = (
            self.data.xipos[self.root_body]
            if self.root_body is not None
            else self.data.subtree_com[0]
        )
        total = np.zeros(6)
        for wing in WINGS:
            force = self.wing_forces(wing)
            rot, hinge = self.wing_pose(wing, self.wing_angles)
            point = hinge + rot @ (self.cop * self.span[wing])
            total[:3] += force
            total[3:] += np.cross(point - com, force)
        if self.root_body is not None:
            self.data.xfrc_applied[self.root_body] = total

    def body_axis(self) -> np.ndarray:
        """The animal's own vertical: the normal of its stroke plane.

        World +z while it is level, and it parts company with world +z exactly
        when attitude starts to matter.
        """
        if self.root_body is None:
            return np.array([0.0, 0.0, 1.0])
        return self.data.xmat[self.root_body].reshape(3, 3) @ np.array([0.0, 0.0, 1.0])

    def wrench_about_com(self) -> np.ndarray:
        """The applied wrench referred to the animal's centre of mass.

        ``xfrc_applied`` acts at the root body's own inertial point, and on
        this model that point is the world origin: ``FlyBody`` is a massless
        wrapper whose ``xipos`` is (0, 0, 0) while the animal's mass actually
        sits at (-0.304, 0.007, 1.067). The wrench applied there is correct
        physics -- a force and a moment about any one point describe the same
        thing -- but the moment it reports is about the origin, and reading
        that as a body torque is a mistake with a lever arm of a millimetre in
        it.

        It is not a small mistake. At the nominal stroke the moment about the
        origin is +0.66 in pitch while the moment about the centre of mass is
        **-3.49**: different magnitude and opposite sign. Trimmed on the first
        number, the controller pushed the wrong way and the animal pitched over
        faster with the loop closed than without it.
        """
        if self.root_body is None:
            return np.zeros(6)
        applied = np.asarray(self.data.xfrc_applied[self.root_body], dtype=float)
        offset = self.data.xipos[self.root_body] - self.data.subtree_com[self.root_body]
        return np.concatenate([applied[:3], applied[3:] + np.cross(offset, applied[:3])])

    def air_velocity_at(self, wing: str, rot, hinge) -> np.ndarray:
        """How fast this wing is moving through the air because the *body* is.

        Not the same as the body's velocity, and the difference is the whole
        of a fly's passive yaw damping. A wing sits out at the radius of
        gyration; when the body rotates at ``omega`` that point moves at
        ``omega x r`` on top of the body's translation, so a spinning animal
        has one wing advancing into the air and the other retreating. The
        forces no longer balance and the difference opposes the spin. This is
        **flapping counter-torque**, and it is the dominant damping on insect
        yaw.

        Without it this model had **exactly zero** yaw damping: an imposed
        spin of 2000 deg/s produced 0.0000 of yaw torque, measured. The animal
        flew for a second and then entered a flat spin that nothing resisted
        and the loop, reading a filtered heading, could not catch. Yaw is also
        the axis with no restoring term of any kind -- pitch and roll at least
        have gravity and the stroke plane -- so it was the only one where the
        omission was fatal.

        The point taken is the radius of gyration rather than the tip or the
        hinge, for the reason :func:`~wingloop.aero.wake.reference_speed`
        gives: the force integral is weighted by the second moment of area, so
        that is the radius the force acts at.
        """
        v = self.body_velocity()
        if self.root_body is None or self.root_translation != 3:
            return v
        omega = self.data.xmat[self.root_body].reshape(3, 3) @ np.asarray(
            self.data.qvel[self.root_dof + 3 : self.root_dof + 6]
        )
        centre = np.asarray(hinge) + rot @ (self._gyration_radius * self.span[wing])
        arm = centre - self.centre_of_mass()
        return v + np.cross(omega, arm)

    def centre_of_mass(self) -> np.ndarray:
        """World position of the whole animal's centre of mass."""
        return np.asarray(self.data.subtree_com[self.root_body or 0], dtype=float)

    def body_velocity(self) -> np.ndarray:
        """World translational velocity of the root body, or zeros on a tether.

        A model without a free joint reports nothing to read here, and that is
        the normal case for the shipped NeuroMechFly -- so this returns zeros
        rather than failing, and the force model reduces to the hovering one.
        """
        if self.root_dof is None:
            return np.zeros(3)
        v = np.zeros(3)
        n = self.root_translation
        if n == 3:
            v[:] = self.data.qvel[self.root_dof : self.root_dof + 3]
        else:  # a vertical rail: the only motion is along z
            v[2] = self.data.qvel[self.root_dof]
        return v

    def set_wings(
        self, angles: dict[str, float], rates: dict[str, float] | None = None
    ) -> None:
        """Command the stroke. The wings are kinematic; nothing integrates them."""
        self.wing_angles = dict(angles)
        self.wing_rates = dict(rates or {})
        if self.joint_id:  # keep a hinged model's joints in sync, for viewing
            self.prescribe(angles, rates)

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
            if jid is None:
                continue
            self.data.qpos[self.model.jnt_qposadr[jid]] = float(value)
            if rates is not None and name in rates:
                self.data.qvel[self.model.jnt_dofadr[jid]] = float(rates[name])
        self._mj.mj_forward(self.model, self.data)

    def _joint_by_name(self, name: str):
        wing, dof = name.replace("joint_", "").split("_")
        return self.joint_id.get(wing, {}).get(dof)

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
    bias: float = 0.0,
    asymmetry: float = 0.0,
    phase: float = 0.0,
    phase_asymmetry: float = 0.0,
    sharpness: float = 0.0,
    deviation: float = 0.0,
    deviation_phase: float = 0.0,
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
    # Stroke position. `sharpness` bends a sinusoid toward a triangle wave:
    # 0 is the sinusoid, and as it approaches 1 the wing spends more of the
    # cycle at near-constant velocity with the turn-around compressed into the
    # ends. A real stroke is closer to the triangle, and the reason matters
    # here -- a sinusoid's velocity, and so its force, is peaked in the middle
    # of every half-stroke, which is where the within-stroke torque swings
    # this model cannot control come from.
    if sharpness > 0.0:
        k = float(np.clip(sharpness, 0.0, 0.999))
        phi = amplitude * np.arcsin(k * np.sin(w * t)) / np.arcsin(k)
    else:
        phi = amplitude * np.sin(w * t)
    # Out-of-plane deviation at twice the wingbeat frequency, which is what
    # makes a real wingtip trace a figure-of-eight rather than an arc.
    dev = deviation * np.cos(2.0 * w * t + deviation_phase)
    # The two control knobs a fly actually has, and the ones this model needs:
    # a symmetric shift of the mean stroke angle swings both wings fore or aft,
    # moving the centre of pressure relative to the centre of mass, which is
    # pitch; an amplitude difference between the sides is roll.
    gain = {"LWing": 1.0 + asymmetry, "RWing": 1.0 - asymmetry}
    # When the wing flips, relative to where it is in the stroke. Advancing the
    # rotation puts the wing at a high angle of attack while it is still moving
    # fast, which is extra lift; delaying it costs lift. Unlike amplitude, this
    # works through the rotational force term rather than through drag, so a
    # left-right difference should roll the animal without first yawing it the
    # wrong way. That is the claim this parameter exists to test.
    flip = {
        "LWing": phase + phase_asymmetry,
        "RWing": phase - phase_asymmetry,
    }
    # Pitch the wing one way on the downstroke and the other on the upstroke.
    # Smoothed rather than a sign flip: a real wing takes a finite time to
    # rotate, and a step here is not merely unrealistic but unsimulable -- the
    # rotational force term is proportional to alpha_dot, so an instantaneous
    # flip is an infinite force. The first version of this used np.sign and
    # MuJoCo answered with "Nan, Inf or huge value in QACC".
    def rotation(p: float) -> float:
        return alpha * np.tanh(SHARPNESS * np.cos(w * t + p)) / np.tanh(SHARPNESS)
    out = {}
    for wing in WINGS:
        # Offset into the flight posture first: the model's rest pose has both
        # wings folded back over the abdomen, and flapping from there sweeps
        # them sideways.
        out[f"joint_{wing}_stroke"] = (
            STROKE_OFFSET[wing] + STROKE_SIGN[wing] * (gain[wing] * phi + bias)
        )
        out[f"joint_{wing}_rotation"] = STROKE_SIGN[wing] * rotation(flip[wing])
        out[f"joint_{wing}_deviation"] = STROKE_SIGN[wing] * dev
    if not rates:
        return out

    # Analytic derivatives, so prescribed kinematics carry exact rates rather
    # than a finite difference that lags by half a step -- the rotational force
    # term is proportional to one of them.
    if sharpness > 0.0:
        k = float(np.clip(sharpness, 0.0, 0.999))
        c = k * np.cos(w * t) * w
        phi_dot = amplitude * c / (np.arcsin(k) * np.sqrt(1.0 - (k * np.sin(w * t)) ** 2))
    else:
        phi_dot = amplitude * w * np.cos(w * t)
    dev_dot = -2.0 * w * deviation * np.sin(2.0 * w * t + deviation_phase)

    def rotation_rate(p: float) -> float:
        c = np.cos(w * t + p)
        return (
            alpha
            * SHARPNESS
            * (1.0 - np.tanh(SHARPNESS * c) ** 2)
            * (-w * np.sin(w * t + p))
        ) / np.tanh(SHARPNESS)
    drates = {}
    for wing in WINGS:
        drates[f"joint_{wing}_stroke"] = STROKE_SIGN[wing] * gain[wing] * phi_dot
        drates[f"joint_{wing}_rotation"] = STROKE_SIGN[wing] * rotation_rate(flip[wing])
        drates[f"joint_{wing}_deviation"] = STROKE_SIGN[wing] * dev_dot
    return out, drates
