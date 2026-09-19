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

from wingloop.body.control import (  # noqa: E402
    PITCH_PER_BIAS,
    TRIM_BIAS,
    HaltereController,
)
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

    from wingloop.body.control import attitude

    pitch, _ = attitude(body)
    assert abs(np.degrees(pitch)) > 45.0, "it is supposed to tip over"
    # Once the stroke plane tilts with the body, a pitched fly stops getting
    # its weight straight up, so it does not even hold height.
    assert body.data.qpos[2] < 1.0
    assert float(np.linalg.norm(body.data.qvel[3:6])) > 100.0


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


# ------------------------------------------------------------------- control


@needs_model
def test_the_reported_torque_is_about_the_animal_not_the_origin(rigid, wing, tmp_path):
    """The bug that made the first controller push the wrong way.

    ``xfrc_applied`` acts at the root body's inertial point, and on this model
    that point is the world origin -- ``FlyBody`` is a massless wrapper -- while
    the animal's mass sits a millimetre away. The wrench applied there is
    correct physics, but the moment it reports is about the origin, and reading
    that as a body torque gets both the magnitude and the sign wrong.
    """
    free = add_free_base(rigid[0], tmp_path / "com.xml", dofs="free")
    body = FlightBody(free, wing, timestep=2e-5)
    assert np.allclose(body.data.xipos[body.root_body], 0.0, atol=1e-9)
    assert np.linalg.norm(body.data.subtree_com[body.root_body]) > 0.5

    raw, about_com = np.zeros(6), np.zeros(6)
    n = 400
    for t in np.linspace(0, 1.0 / FREQUENCY, n, endpoint=False):
        angles, rates = harmonic_stroke(t, frequency=FREQUENCY, rates=True)
        body.set_wings(angles, rates)
        body.apply_aerodynamics()
        raw += body.data.xfrc_applied[body.root_body]
        about_com += body.wrench_about_com()
    raw, about_com = raw / n, about_com / n

    assert np.allclose(raw[:3], about_com[:3]), "the force must not change"
    # Opposite signs in pitch: +0.66 about the origin, -3.49 about the animal.
    assert raw[4] > 0 and about_com[4] < 0
    assert abs(about_com[4]) > 3.0


@needs_model
def test_the_trim_bias_nulls_the_pitch_torque(rigid, wing, tmp_path):
    """What TRIM_BIAS is, measured rather than asserted from theory."""
    free = add_free_base(rigid[0], tmp_path / "trim.xml", dofs="free")
    body = FlightBody(free, wing, timestep=2e-5)

    def pitch_torque(bias: float) -> float:
        total, n = 0.0, 400
        for t in np.linspace(0, 1.0 / FREQUENCY, n, endpoint=False):
            angles, rates = harmonic_stroke(
                t, frequency=FREQUENCY, bias=bias, rates=True
            )
            body.set_wings(angles, rates)
            body.apply_aerodynamics()
            total += float(body.wrench_about_com()[4])
        return total / n

    assert abs(pitch_torque(TRIM_BIAS)) < 0.2, "trim should leave no pitch torque"
    assert pitch_torque(0.0) < -3.0, "untrimmed, it pitches nose-down hard"
    # Authority, and its sign: more bias is more nose-down.
    slope = (pitch_torque(np.deg2rad(5)) - pitch_torque(np.deg2rad(-5))) / np.deg2rad(10)
    assert slope == pytest.approx(PITCH_PER_BIAS, rel=0.15)


@needs_model
def test_the_stroke_plane_follows_the_body(rigid, wing, tmp_path):
    """Lift is normal to the animal's stroke plane, not to the world.

    Kept vertical regardless of attitude, a pitched fly still gets its whole
    weight straight up and the controller is steering something that cannot be
    steered.
    """
    free = add_free_base(rigid[0], tmp_path / "plane.xml", dofs="free")
    body = FlightBody(free, wing, timestep=2e-5)
    assert np.allclose(body.body_axis(), [0, 0, 1], atol=1e-6)

    # Roll the body a quarter turn about its long axis and look again.
    body.data.qpos[3:7] = [np.cos(np.pi / 8), np.sin(np.pi / 8), 0.0, 0.0]
    body._mj.mj_forward(body.model, body.data)
    tilted = body.body_axis()
    assert tilted[2] == pytest.approx(np.cos(np.pi / 4), abs=1e-3)
    assert abs(tilted[1]) > 0.5


@needs_model
def test_closing_the_loop_holds_attitude_at_first_and_then_does_not(
    rigid, wing, tmp_path
):
    """The honest state of the controller, pinned so it cannot drift unnoticed.

    Four wingbeats of genuine attitude hold, then the within-stroke torque
    wins. This asserts both halves: that the loop does something real early,
    and that it does not yet do the thing it is for. When the stroke kinematics
    improve, the second assertion is the one that should start failing.
    """
    free = add_free_base(rigid[0], tmp_path / "loop.xml", dofs="free")
    body = FlightBody(free, wing, timestep=2e-5)
    trace = HaltereController().fly(body, 0.30)

    # Twenty-two wingbeats of attitude hold, against an open-loop fly that is
    # past 45 degrees within nine.
    early = trace["t"] < 0.100
    assert np.degrees(np.abs(trace["pitch"][early])).max() < 20.0
    assert np.degrees(np.abs(trace["roll"][early])).max() < 25.0

    assert trace["t"][-1] > 0.29, "it should stay in the air"
    assert trace["z"][-1] > 50.0, "and climb while it does"

    # And the part that is not finished. When the stroke kinematics improve
    # this should start failing, which is the point of asserting it.
    assert np.degrees(np.abs(trace["pitch"])).max() > 30.0, (
        "attitude hold now lasts the whole run -- update this claim"
    )


# --------------------------------------------------- the connectome steering


@needs_model
def test_the_connectome_steers_the_body_toward_the_object(rigid, wing, tmp_path):
    """The whole point of the project, in a body that flies.

    Vision reaches DNbe001, its right-minus-left difference shifts when the
    wings flip, and the shift yaws the animal. It turns toward the side the
    object is on -- fixation, which is what the walking version does too.

    The heading is the thing to read, not the sideways displacement: rotation
    phase is a yaw control. Amplitude steering banks instead, and is checked
    that way in ``test_a_right_turn_command_yaws_left_first``.

    The control is what makes it a measurement: the same machinery with a flat
    curve carries no bearing information and must steer not at all.
    """
    from wingloop.body.control import HaltereController, SteeringController
    from wingloop.brain.readout import FlightReadout

    stored = dict(np.load(Path(__file__).parent / "dnbe001_tuning.npz"))
    real = FlightReadout(bearings=stored["bearings"], command=stored["command"])
    flat = FlightReadout(
        bearings=stored["bearings"], command=np.zeros_like(stored["command"])
    )

    def heading(readout, bearing) -> float:
        free = add_free_base(rigid[0], tmp_path / f"h{bearing:+.0f}.xml", dofs="free")
        body = FlightBody(free, wing, timestep=2e-5)
        SteeringController(readout=readout, bearing=bearing).fly(body, 0.10)
        m = body.data.xmat[body.root_body].reshape(3, 3)
        return float(np.degrees(np.arctan2(m[1, 0], m[0, 0])))

    # The fly faces +x and its left is +y, so turning left is increasing yaw.
    left_object = heading(real, -45.0)
    right_object = heading(real, 45.0)
    assert left_object > right_object + 90.0, (left_object, right_object)

    # Without bearing information every bearing gives the same heading, and it
    # is the one the stabiliser reaches on its own.
    blind_left = heading(flat, -45.0)
    blind_right = heading(flat, 45.0)
    assert blind_left == pytest.approx(blind_right, abs=1e-6)

    free = add_free_base(rigid[0], tmp_path / "bare.xml", dofs="free")
    bare = FlightBody(free, wing, timestep=2e-5)
    HaltereController().fly(bare, 0.10)
    m = bare.data.xmat[bare.root_body].reshape(3, 3)
    assert blind_left == pytest.approx(
        float(np.degrees(np.arctan2(m[1, 0], m[0, 0]))), abs=1e-6
    )

    # And the steering straddles that baseline rather than sitting to one side.
    assert left_object > blind_left > right_object


@needs_model
def test_steering_response_follows_the_bearing_monotonically(rigid, wing, tmp_path):
    """Not just two points: the command is graded, so the turn should be."""
    from wingloop.body.control import SteeringController
    from wingloop.brain.readout import FlightReadout

    stored = dict(np.load(Path(__file__).parent / "dnbe001_tuning.npz"))
    readout = FlightReadout(bearings=stored["bearings"], command=stored["command"])

    headings = []
    for bearing in (-60.0, -30.0, 30.0):
        free = add_free_base(rigid[0], tmp_path / f"m{bearing:+.0f}.xml", dofs="free")
        body = FlightBody(free, wing, timestep=2e-5)
        SteeringController(readout=readout, bearing=bearing).fly(body, 0.06)
        m = body.data.xmat[body.root_body].reshape(3, 3)
        headings.append(float(np.degrees(np.arctan2(m[1, 0], m[0, 0]))))

    assert headings[0] > headings[1] > headings[2], headings
