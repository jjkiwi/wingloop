"""Giving NeuroMechFly a wing hinge, because the shipped model has none.

The wings in `neuromechfly_seqik_kinorder_ypr.xml` are `<geom>` elements welded
straight onto the thorax: no joint, no actuator, `contype="0"` so they do not
even collide. They are there to be looked at. This rewrites the MJCF so each
wing hangs off three hinge joints, which is the minimum that can express a
flapping stroke.

**The three angles are the standard decomposition**, and they are not
interchangeable:

``stroke`` (phi)
    Sweeps the wing fore and aft. This is the large one -- about 150 degrees
    peak to peak in hovering -- and its rate is what sets the wing's velocity
    through the air, so nearly all of the force scales with it.
``deviation`` (theta)
    Takes the wing out of the stroke plane, giving the wingtip path its shape
    (the figure-of-eight or shallow oval a real fly traces). Small, and it
    changes force mostly by changing the other two.
``rotation`` (alpha)
    Pitches the wing about its own spanwise axis. This is the angle of attack,
    and because the rotational force term depends on ``alpha_dot`` times the
    stroke rate, *when* this happens within the stroke matters as much as how
    far it goes. It is the axis a fly steers with.

The axes are read off the mesh rather than assumed: NeuroMechFly's wing mesh
spans 2.40 along its local z and 1.09 along local y, so the span is **local z**
and rotation has to be about that. Getting this wrong gives a wing that pitches
about its chord, which still flaps and produces nothing.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Joint suffixes, outermost last: stroke, then deviation, then rotation.
#: Order is the nesting order in the kinematic chain, not a naming convention.
HINGE_DOFS = ("stroke", "deviation", "rotation")

#: Rotation axes in the wing body's own frame.
#:
#: **Span is local y, not local z**, and that cost a debugging session worth
#: recording. The wing *mesh* is longest along its own axis 2, so the obvious
#: reading is that the span is local z -- but the wing body carries a 90 degree
#: rotation about z relative to the thorax, and the mesh sits inside that. Read
#: off the built model instead of the mesh: the right wing's hinge is at world
#: (-0.05, -0.37, 1.48) and its farthest vertex at (-2.34, -0.39, 1.42), so the
#: span runs along world -x, which is local -y. Local z points straight up.
#:
#: With the axes as first written, the "rotation" joint pitched the wing about
#: a vertical axis and the stroke swept it in a vertical plane. It still
#: flapped. It produced -0.001 of body weight.
HINGE_AXES = {
    "stroke": "0 0 1",      # sweep, in the horizontal stroke plane
    "deviation": "1 0 0",   # out of that plane
    "rotation": "0 1 0",    # angle of attack, about the span
}

#: The span direction in each wing body's own frame, measured as above. The
#: wings are mirror images, so the sign differs and the sweep drives them in
#: opposite local directions.
SPAN_LOCAL = {"LWing": (0.0, 1.0, 0.0), "RWing": (0.0, -1.0, 0.0)}

#: Stroke angle at which each wing sticks out sideways, ready to fly.
#:
#: **The model's rest pose is a resting fly**, wings folded back along the
#: abdomen: both spans point down world -x. Flapping from there sweeps the
#: wingtips side to side instead of fore and aft, and the giveaway is that both
#: wings' drag then points the same way laterally instead of cancelling -- a
#: measured net roll torque of 3.7 on a stroke that is symmetric by
#: construction.
#:
#: Rotating the stroke joint a quarter turn swings the span out to the side,
#: which is the flight posture: span along +-y, sweep along +-x, drag along the
#: body axis where it belongs. The signs are opposite because the wings fold to
#: the same side but must extend to different ones.
STROKE_OFFSET = {"LWing": -np.pi / 2, "RWing": np.pi / 2}

#: Which way a positive stroke command moves each wing, so that one command
#: sweeps both wings forwards together rather than scissoring them.
STROKE_SIGN = {"LWing": -1.0, "RWing": 1.0}

#: Ranges in radians. Stroke is the wide one; rotation spans enough to flip the
#: wing at both reversals, which is how a fly gets lift on the upstroke too.
HINGE_RANGES = {
    "stroke": (-3.2, 3.2),
    "deviation": (-0.6, 0.6),
    "rotation": (-2.2, 2.2),
}

WINGS = ("LWing", "RWing")


@dataclass
class HingedModel:
    """Where the rewritten model went, and what to address in it."""

    path: Path
    joints: dict[str, list[str]]  # wing -> joint names, in chain order
    actuators: dict[str, list[str]]

    @property
    def all_joints(self) -> list[str]:
        return [j for w in WINGS for j in self.joints[w]]


def add_wing_hinges(
    source: str | Path,
    destination: str | Path,
    *,
    damping: float = 1e-7,
    gear: float = 1.0,
    kp: float = 30.0,
) -> HingedModel:
    """Rewrite an MJCF so both wings hang off a three-degree-of-freedom hinge.

    The wing body is left where it is and the joints are inserted into it, so
    the wing's rest pose, mesh and mass are untouched -- this adds freedom, it
    does not move anything.

    ``kp`` is sized from what the stroke demands rather than guessed. A wing's
    inertia about the hinge is about ``2.5e-6 * 2.4**2 = 1.4e-5``; a 75 degree
    harmonic sweep at 218 Hz peaks at ``(2*pi*218)**2 * 1.31 = 2.5e6`` rad/s^2,
    so peak torque is around 35, and a position servo closing that against an
    error of roughly the amplitude needs ``kp ~ 27``. The first value here was
    1e-4, five orders of magnitude short, and the wing tracked a third of its
    commanded stroke -- which, with force going as the square of stroke rate,
    is most of two orders of magnitude of missing lift.

    ``damping`` stays small because the parts are: a value sized for a leg
    holds a wing still.
    """
    source, destination = Path(source), Path(destination)
    tree = ET.parse(source)
    root = tree.getroot()

    bodies = {b.get("name"): b for b in root.iter("body")}
    missing = [w for w in WINGS if w not in bodies]
    if missing:
        raise KeyError(f"{source.name} has no {missing} body to hinge")

    joints: dict[str, list[str]] = {}
    for wing in WINGS:
        body = bodies[wing]
        existing = [j.get("name") for j in body.findall("joint")]
        if existing:
            raise ValueError(
                f"{wing} already has joints {existing}; this model has been "
                "hinged already, or is not the one this was written for"
            )
        names = []
        for dof in HINGE_DOFS:
            name = f"joint_{wing}_{dof}"
            lo, hi = HINGE_RANGES[dof]
            # Inserted at the front so the joints precede the geom, matching
            # how the leg bodies in this file are written.
            ET.SubElement(
                body,
                "joint",
                {
                    "name": name,
                    "type": "hinge",
                    "pos": "0 0 0",
                    "axis": HINGE_AXES[dof],
                    "range": f"{lo} {hi}",
                    "limited": "true",
                    "damping": str(damping),
                    "stiffness": "0.0",
                    "frictionloss": "0.0",
                },
            )
            names.append(name)
        joints[wing] = names

    actuator = root.find("actuator")
    if actuator is None:
        actuator = ET.SubElement(root, "actuator")
    actuators: dict[str, list[str]] = {}
    for wing in WINGS:
        made = []
        for name in joints[wing]:
            act = f"actuator_position_{name}"
            ET.SubElement(
                actuator,
                "position",
                {"name": act, "joint": name, "kp": str(kp), "gear": str(gear)},
            )
            made.append(act)
        actuators[wing] = made

    _absolutise_meshdir(root, source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tree.write(destination, encoding="unicode")
    return HingedModel(path=destination, joints=joints, actuators=actuators)


def _absolutise_meshdir(root: ET.Element, source: Path) -> None:
    """Pin mesh lookup to the source model's own directory.

    The shipped MJCF names its meshes as ``../mesh/Thorax.stl``, relative to
    wherever the file sits. Write the rewritten model anywhere else and MuJoCo
    looks for the meshes beside *that* file and fails to open the first one.
    Setting an absolute ``meshdir`` makes the output relocatable, which it has
    to be -- the alternative is writing into the installed package.
    """
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    if compiler.get("meshdir"):
        return
    # Every mesh in this file shares one prefix; resolve it once and strip it
    # from the individual entries so meshdir alone decides where they come from.
    files = [m.get("file", "") for m in root.iter("mesh") if m.get("file")]
    if not files:
        return
    prefix = Path(files[0]).parent
    if any(Path(f).parent != prefix for f in files):
        raise ValueError(
            "the model's meshes come from several directories; meshdir cannot "
            "stand in for all of them"
        )
    compiler.set("meshdir", str((source.parent / prefix).resolve()))
    for m in root.iter("mesh"):
        if m.get("file"):
            m.set("file", Path(m.get("file")).name)


def add_free_base(
    source: str | Path,
    destination: str | Path,
    *,
    root: str = "FlyBody",
    dofs: str = "free",
):
    """Unweld the animal from the world so it can actually go somewhere.

    The shipped model has no free joint anywhere: `nv` counts only the leg and
    head joints, and the fly's root body is fixed. That is the right choice for
    a walking simulation, where the ground carries the animal and contact does
    the rest -- and it means every flight result measured on the unmodified
    model is a force on a tether, not a trajectory.

    ``dofs="free"`` gives all six degrees of freedom back. Note what that
    exposes: a fly is passively unstable in pitch, so a free body driven by an
    open-loop stroke lifts off and tips over within about ten milliseconds.
    That is not a bug in the aerodynamics -- the vertical force is right, and
    measured to be right at the generalised-force level -- it is the reason the
    animal has halteres, and it is what a controller is for.

    ``dofs="z"`` gives a single vertical slide instead. That is a real
    preparation, not a dodge: a fly on a vertical rail can be asked whether it
    generates enough force to climb without also being asked to balance, and
    the two questions have different answers here. Use it to test the
    aerodynamics; use ``"free"`` to test control.
    """
    source, destination = Path(source), Path(destination)
    tree = ET.parse(source)
    xml_root = tree.getroot()
    bodies = {b.get("name"): b for b in xml_root.iter("body")}
    if root not in bodies:
        raise KeyError(f"{source.name} has no {root!r} body to free")
    body = bodies[root]
    if body.find("freejoint") is not None or any(
        j.get("type") == "free" for j in body.findall("joint")
    ):
        raise ValueError(f"{root} is already free; this model has been freed already")
    if dofs == "free":
        body.insert(0, ET.Element("freejoint", {"name": f"free_{root}"}))
    elif dofs == "z":
        body.insert(
            0,
            ET.Element(
                "joint",
                {
                    "name": f"slide_{root}_z",
                    "type": "slide",
                    "axis": "0 0 1",
                    "pos": "0 0 0",
                    "damping": "0.0",
                    "limited": "false",
                },
            ),
        )
    else:
        raise ValueError(f"dofs must be 'free' or 'z', not {dofs!r}")
    _absolutise_meshdir(xml_root, source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tree.write(destination, encoding="unicode")
    return Path(destination)


def tuck_legs(source: str | Path, destination: str | Path, *, keep: str = "Wing"):
    """Make everything but the wings rigid, because a flying fly tucks its legs.

    The shipped model's leg joints carry no actuators -- FlyGym adds those for
    walking. Free the body without dealing with them and 60-odd unpowered
    joints flail under gravity: the first free-flight run here diverged at
    15 ms with "Inertia matrix is too close to singular at DOF 87", which is a
    leg, and MuJoCo silently reset the simulation three times.

    Removing the joints welds each segment to its parent, which is both the
    numerically sound thing and the biologically right one for flight. The legs
    keep their mass and their inertia; they simply stop being a pendulum.

    Returns the path and how many joints were removed.
    """
    source, destination = Path(source), Path(destination)
    tree = ET.parse(source)
    xml_root = tree.getroot()
    removed = 0
    for body in xml_root.iter("body"):
        for joint in list(body.findall("joint")):
            if keep not in (joint.get("name") or ""):
                body.remove(joint)
                removed += 1
    # Actuators whose joint just disappeared would fail to compile.
    actuator = xml_root.find("actuator")
    if actuator is not None:
        names = {j.get("name") for j in xml_root.iter("joint")}
        for act in list(actuator):
            if act.get("joint") and act.get("joint") not in names:
                actuator.remove(act)
    # Without this the compiler fuses every jointless body into its parent,
    # and the wings -- which have just lost their joints -- stop existing as
    # addressable bodies. Their pose is computed rather than simulated, but
    # something still has to report where they are mounted.
    compiler = xml_root.find("compiler")
    if compiler is not None:
        compiler.set("fusestatic", "false")
    _absolutise_meshdir(xml_root, source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tree.write(destination, encoding="unicode")
    return Path(destination), removed


def neuromechfly_model() -> Path:
    """Path to the MJCF that ships with FlyGym, if it is installed.

    Kept here rather than hardcoded into a test so the one place that knows
    where the model lives is the module that rewrites it.
    """
    try:
        import flygym_gymnasium
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "the source model comes from FlyGym:\n    pip install flygym-gymnasium"
        ) from exc
    path = (
        Path(flygym_gymnasium.__file__).parent
        / "data"
        / "mjcf"
        / "neuromechfly_seqik_kinorder_ypr.xml"
    )
    if not path.is_file():  # pragma: no cover - depends on the FlyGym version
        raise FileNotFoundError(f"FlyGym is installed but {path} is not there")
    return path
