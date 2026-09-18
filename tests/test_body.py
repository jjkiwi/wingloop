"""The hinge, and whether the force model still works once it is in a body."""

from pathlib import Path

import numpy as np
import pytest

from wingloop.aero.blade_element import (
    hover_check,
    lift_coefficient,
    stroke_average_lift,
)
from wingloop.aero.wing import wing_from_mesh

mujoco = pytest.importorskip("mujoco", reason="the flight body needs MuJoCo")

from wingloop.body.flight import FlightBody, harmonic_stroke  # noqa: E402
from wingloop.body.hinge import (  # noqa: E402
    HINGE_DOFS,
    SPAN_LOCAL,
    WINGS,
    add_free_base,
    add_wing_hinges,
    neuromechfly_model,
    tuck_legs,
)

MESH = Path(__file__).parent / "rwing_vertices.npy"
FREQUENCY = 218.0

try:
    SOURCE = neuromechfly_model()
except Exception:  # pragma: no cover - FlyGym absent
    SOURCE = None

needs_model = pytest.mark.skipif(SOURCE is None, reason="needs FlyGym's MJCF")
pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def hinged(tmp_path_factory):
    out = tmp_path_factory.mktemp("mjcf") / "hinged.xml"
    return add_wing_hinges(SOURCE, out)


@pytest.fixture(scope="module")
def wing():
    return wing_from_mesh(np.load(MESH), n=20, span_axis=2, chord_axis=1)


@pytest.fixture(scope="module")
def body(hinged, wing):
    return FlightBody(hinged.path, wing)


# ------------------------------------------------------------------- the hinge


@needs_model
def test_the_shipped_model_really_has_no_wing_joints():
    """The premise of this whole module. If FlyGym ever ships a hinged wing,
    this fails and the rewrite should be reconsidered rather than stacked on."""
    m = mujoco.MjModel.from_xml_path(str(SOURCE))
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt)]
    assert not [n for n in names if n and ("Wing" in n or "Haltere" in n)]


@needs_model
def test_hinging_adds_three_joints_and_an_actuator_each(hinged):
    assert set(hinged.joints) == set(WINGS)
    for w in WINGS:
        assert hinged.joints[w] == [f"joint_{w}_{d}" for d in HINGE_DOFS]
        assert len(hinged.actuators[w]) == 3


@needs_model
def test_the_rewritten_model_loads_and_is_relocatable(hinged):
    """Meshes are named relative to the source file, so a rewrite written
    anywhere else fails to open the first one unless meshdir is absolutised."""
    m = mujoco.MjModel.from_xml_path(str(hinged.path))
    assert m.nu == 6
    assert hinged.path.parent != SOURCE.parent


@needs_model
def test_hinging_twice_is_refused(hinged, tmp_path):
    with pytest.raises(ValueError, match="already has joints"):
        add_wing_hinges(hinged.path, tmp_path / "again.xml")


@needs_model
def test_the_span_is_local_y_not_the_mesh_long_axis(body):
    """Measured, because assuming it cost a session: the wing body carries a 90
    degree rotation, so the mesh's longest axis is not the body's span axis."""
    for w in WINGS:
        hinge = body.data.xpos[body.body_id[w]]
        rot = body.data.xmat[body.body_id[w]].reshape(3, 3)
        tip = hinge + rot @ (body.wing.length * np.asarray(SPAN_LOCAL[w]))
        # The wing lies along the body axis, roughly horizontal, not vertical.
        assert abs(tip[2] - hinge[2]) < 0.3 * body.wing.length
        assert abs(tip[0] - hinge[0]) > 0.8 * body.wing.length


# ------------------------------------------------------- forces inside a body


@needs_model
def test_a_stroke_in_the_body_lifts_about_what_the_analytic_model_says(body, wing):
    """Two independent paths to the same number.

    ``hover_check`` integrates the stroke analytically with no simulator in
    sight. This drives the wings through MuJoCo with prescribed kinematics and
    sums what is applied to the body. They agree, which is evidence that the
    frames, the centre of pressure and the sign conventions in the body layer
    do not quietly undo the force model.
    """
    analytic = hover_check(wing, float(body.model.body_mass.sum()))["ratio"]

    weight = float(body.model.body_mass.sum()) * 9810.0
    vertical = []
    for t in np.linspace(0, 2.0 / FREQUENCY, 800, endpoint=False):
        angles, rates = harmonic_stroke(t, frequency=FREQUENCY, rates=True)
        body.set_wings(angles, rates)
        body.apply_aerodynamics()
        vertical.append(body.total_applied_force()[2])
    embodied = float(np.mean(vertical)) / weight

    assert embodied == pytest.approx(analytic, rel=0.1)
    assert 0.5 < embodied < 3.0


@needs_model
def test_the_two_wings_cancel_sideways(body):
    """Mirrored wings on a symmetric stroke must leave no net side force. A
    residual here means one wing's drag has the wrong sign, which would look
    like a fly that steers while flying straight."""
    horizontal = []
    for t in np.linspace(0, 2.0 / FREQUENCY, 800, endpoint=False):
        angles, rates = harmonic_stroke(t, frequency=FREQUENCY, rates=True)
        body.set_wings(angles, rates)
        body.apply_aerodynamics()
        horizontal.append(body.total_applied_force()[:2])
    mean = np.abs(np.mean(horizontal, axis=0))
    weight = float(body.model.body_mass.sum()) * 9810.0
    assert np.all(mean < 0.01 * weight), mean


@needs_model
def test_the_upstroke_does_not_push_the_fly_down(body):
    """The bug this is here for: the coefficient fits run from zero to ninety
    degrees, and the upstroke always presents a negative pitch angle. Feeding
    that in directly returns a negative lift coefficient, and the wing pushes
    the animal into the ground for half of every cycle."""
    assert lift_coefficient(np.deg2rad(-45.0)) < 0, "the fit is signed, as assumed"

    per_step = []
    for t in np.linspace(0, 1.0 / FREQUENCY, 400, endpoint=False):
        angles, rates = harmonic_stroke(t, frequency=FREQUENCY, rates=True)
        body.set_wings(angles, rates)
        body.apply_aerodynamics()
        per_step.append((angles["joint_RWing_rotation"], body.total_applied_force()[2]))

    upstroke = [v for a, v in per_step if a < -0.1]
    assert upstroke, "the test stroke never pitched the wing negative"
    assert np.mean(upstroke) > 0, "the upstroke is producing downward force"


# ------------------------------------------------------------- free of the rig


@pytest.fixture(scope="module")
def rigid(tmp_path_factory):
    """A fly with no joints at all: legs welded, wings welded, pose computed."""
    d = tmp_path_factory.mktemp("rigid")
    out, removed = tuck_legs(SOURCE, d / "rigid.xml", keep="__none__")
    return out, removed


@needs_model
def test_welding_everything_keeps_the_wings_addressable(rigid):
    """`fusestatic` folds jointless bodies into their parent, and the wings --
    having just lost their joints -- stop existing. Something still has to
    report where they are mounted."""
    out, removed = rigid
    assert removed > 80
    m = mujoco.MjModel.from_xml_path(str(out))
    assert m.njnt == 0
    for w in WINGS:
        assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, w) >= 0


@needs_model
def test_the_analytic_pose_matches_the_simulated_one(body):
    """The licence for taking the wings out of the physics.

    Their pose is composed from the commanded angles instead of integrated, so
    the composition -- mount, then stroke about z, deviation about x, rotation
    about y -- has to reproduce what MuJoCo's own kinematics produce on a model
    that does have the joints. It does, to machine precision.
    """
    worst = 0.0
    for t in (0.0, 0.001, 0.002, 0.0031):
        angles, rates = harmonic_stroke(t, rates=True)
        body.prescribe(angles, rates)
        for w in WINGS:
            simulated = body.data.xmat[body.body_id[w]].reshape(3, 3)
            analytic, position = body.wing_pose(w, angles)
            worst = max(
                worst,
                float(np.abs(simulated - analytic).max()),
                float(np.abs(body.data.xpos[body.body_id[w]] - position).max()),
            )
    assert worst < 1e-12, worst


@needs_model
def test_on_a_vertical_rail_it_climbs_at_the_predicted_rate(rigid, wing, tmp_path):
    """The whole point: force production, in a body, that actually moves it.

    A rail is a real preparation rather than a dodge -- it asks whether the
    animal makes enough force to climb without also asking it to balance, and
    those have different answers here.
    """
    rail = add_free_base(rigid[0], tmp_path / "rail.xml", dofs="z")
    body = FlightBody(rail, wing, timestep=2e-5)
    assert body.model.nv == 1, "the rail should leave exactly one degree of freedom"

    mass = float(body.model.body_mass.sum())
    samples = []
    for _ in range(int(0.06 / body.model.opt.timestep)):
        angles, rates = harmonic_stroke(body.t, frequency=FREQUENCY, rates=True)
        body.set_wings(angles, rates)
        body.apply_aerodynamics()
        body._mj.mj_step(body.model, body.data)
        samples.append((body.data.time, float(body.data.qvel[0])))

    assert body.data.qpos[0] > 0, "it should be higher than it started"
    (t0, v0), (t1, v1) = samples[len(samples) // 3], samples[-1]
    observed = (v1 - v0) / (t1 - t0)
    lift = 2.0 * stroke_average_lift(wing)
    predicted = (lift - mass * 9810.0) / mass
    assert observed == pytest.approx(predicted, rel=0.05)


@needs_model
def test_free_in_six_degrees_it_lifts_off_and_tips_over(rigid, wing, tmp_path):
    """Not a failure of the model: a fly is passively unstable in pitch.

    The wing hinge sits about a quarter of a millimetre ahead of the centre of
    mass, so vertical force there is a nose-down torque, and nothing in an
    open-loop stroke opposes it. The animal solves this with halteres and
    steering muscles; this project solves it with a controller, and until then
    the honest description of free flight here is "lifts off, then tumbles".
    """
    free = add_free_base(rigid[0], tmp_path / "free.xml", dofs="free")
    body = FlightBody(free, wing, timestep=2e-5)
    assert body.model.nv == 6

    for _ in range(int(0.04 / body.model.opt.timestep)):
        angles, rates = harmonic_stroke(body.t, frequency=FREQUENCY, rates=True)
        body.set_wings(angles, rates)
        body.apply_aerodynamics()
        body._mj.mj_step(body.model, body.data)

    assert body.data.qpos[2] > 0, "it should still have gained height"
    tumble = float(np.linalg.norm(body.data.qvel[3:6]))
    assert tumble > 100.0, "open-loop flapping is not supposed to be stable"


@needs_model
def test_an_asymmetric_stroke_rolls_the_animal(rigid, wing, tmp_path):
    """Why the force is applied at the centre of pressure and not at the hinge.

    On a symmetric stroke the two wings' lateral offsets mirror and the roll
    torques cancel, so applying at the hinge instead gives the same answer and
    nothing notices -- a sabotage check confirmed exactly that. It stops being
    true the moment the wings do different things, which is the only way a fly
    turns. Beating one wing harder than the other has to produce a roll, and
    its size depends on the moment arm being right.
    """
    free = add_free_base(rigid[0], tmp_path / "roll.xml", dofs="free")
    body = FlightBody(free, wing, timestep=2e-5)

    def roll_torque(left_gain: float) -> float:
        total = 0.0
        n = 400
        for t in np.linspace(0, 1.0 / FREQUENCY, n, endpoint=False):
            angles, rates = harmonic_stroke(t, frequency=FREQUENCY, rates=True)
            for key in list(angles):
                if key.startswith("joint_LWing") and key.endswith("stroke"):
                    angles[key] *= left_gain
                    rates[key] *= left_gain
            body.set_wings(angles, rates)
            body.apply_aerodynamics()
            total += float(body.data.xfrc_applied[body.root_body][3])
        return total / n

    balanced = roll_torque(1.0)
    lopsided = roll_torque(0.6)
    assert abs(balanced) < 0.05, f"a symmetric stroke should not roll: {balanced}"
    # Measured: 7.4 with the force at the centre of pressure, 1.6 with it at
    # the hinge. The threshold sits between them on purpose -- a loose one
    # passes either way, which a sabotage check caught it doing.
    assert abs(lopsided) > 4.0, f"the moment arm is too short: {lopsided}"
