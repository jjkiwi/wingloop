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
def test_closing_the_loop_holds_attitude_for_the_whole_run(rigid, wing, tmp_path):
    """What the loop does now that the filter lag was measured, not guessed.

    The earlier version of this test asserted the opposite -- that attitude
    hold lasted about 20 ms and then failed -- with a note saying the
    assertion should start failing if the controller ever got better. It did.
    Dropping the sensor filter from 20 ms to 5 took controlled flight from
    187 ms to 353 and brought the whole 300 ms run inside 20 degrees.
    """
    free = add_free_base(rigid[0], tmp_path / "loop.xml", dofs="free")
    body = FlightBody(free, wing, timestep=2e-5)
    controller = HaltereController()
    trace = controller.fly(body, 0.30)

    assert controller.diverged_at is None, "it should not blow up inside 300 ms"
    assert np.degrees(np.abs(trace["pitch"])).max() < 20.0
    assert np.degrees(np.abs(trace["roll"])).max() < 25.0
    assert trace["z"][-1] > 100.0, "and climb the whole way"


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
        """Cumulative turn, not the wrapped angle.

        With the loop properly tuned a steering command carries the animal
        past half a turn inside 100 ms, and a wrapped atan2 then reports a
        hard left as a right -- which is how a working steering test starts
        failing for a reason that has nothing to do with steering.
        """
        free = add_free_base(rigid[0], tmp_path / f"h{bearing:+.0f}.xml", dofs="free")
        body = FlightBody(free, wing, timestep=2e-5)
        return float(
            SteeringController(readout=readout, bearing=bearing).fly(body, 0.10)[
                "heading"
            ][-1]
        )

    # The fly faces +x and its left is +y, so turning left is a positive turn.
    left_object = heading(real, -45.0)
    right_object = heading(real, 45.0)
    assert left_object > 100.0, left_object
    assert right_object < 0.0, right_object

    # Without bearing information every bearing gives the same heading, and it
    # is the one the stabiliser reaches on its own.
    blind_left = heading(flat, -45.0)
    blind_right = heading(flat, 45.0)
    assert blind_left == pytest.approx(blind_right, abs=1e-6)

    free = add_free_base(rigid[0], tmp_path / "bare.xml", dofs="free")
    bare = FlightBody(free, wing, timestep=2e-5)
    bare_turn = HaltereController().fly(bare, 0.10)["heading"][-1]
    assert blind_left == pytest.approx(float(bare_turn), abs=1e-6)

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
        headings.append(
            float(
                SteeringController(readout=readout, bearing=bearing).fly(body, 0.06)[
                    "heading"
                ][-1]
            )
        )

    assert headings[0] > headings[1] > headings[2], headings


# ------------------------------------------------------------ stroke shape


def test_the_stroke_shape_knobs_have_exact_rates():
    """Everything downstream reads velocities, and the rotational force term
    is proportional to one of them, so a finite-difference approximation here
    would show up as a force error rather than as a kinematics error."""
    from wingloop.body.flight import harmonic_stroke

    for kwargs in (
        {},
        {"sharpness": 0.8},
        {"deviation": np.deg2rad(15)},
        {"sharpness": 0.8, "deviation": np.deg2rad(15), "deviation_phase": 0.7},
    ):
        for t in np.linspace(0, 1 / 218.0, 17):
            h = 1e-8
            before, _ = harmonic_stroke(t - h, frequency=218.0, rates=True, **kwargs)
            after, _ = harmonic_stroke(t + h, frequency=218.0, rates=True, **kwargs)
            _, rates = harmonic_stroke(t, frequency=218.0, rates=True, **kwargs)
            for joint, value in rates.items():
                finite = (after[joint] - before[joint]) / (2 * h)
                assert abs(finite - value) / max(abs(value), 1.0) < 1e-6, (
                    joint,
                    kwargs,
                )


def test_sharpness_bends_the_sweep_toward_a_triangle():
    """At 0 it is a sinusoid; as it rises the wing spends more of the cycle at
    near-constant speed with the turn-around squeezed into the ends."""
    from wingloop.body.flight import harmonic_stroke

    def speeds(sharpness):
        out = []
        for t in np.linspace(0, 1 / 218.0, 200, endpoint=False):
            _, rates = harmonic_stroke(
                t, frequency=218.0, sharpness=sharpness, rates=True
            )
            out.append(abs(rates["joint_RWing_stroke"]))
        return np.asarray(out)

    sinusoid, triangular = speeds(0.0), speeds(0.9)
    # A triangle wave holds its speed; a sinusoid's is peaked in mid-stroke.
    assert triangular.std() / triangular.mean() < sinusoid.std() / sinusoid.mean()


def test_deviation_moves_the_wing_out_of_the_stroke_plane_twice_a_beat():
    """What makes a real wingtip trace a figure-of-eight rather than an arc."""
    from wingloop.body.flight import harmonic_stroke

    flat = [
        harmonic_stroke(t, frequency=218.0)["joint_RWing_deviation"]
        for t in np.linspace(0, 1 / 218.0, 64, endpoint=False)
    ]
    assert max(abs(v) for v in flat) == pytest.approx(0.0)

    wavy = np.asarray(
        [
            harmonic_stroke(t, frequency=218.0, deviation=np.deg2rad(15))[
                "joint_RWing_deviation"
            ]
            for t in np.linspace(0, 1 / 218.0, 256, endpoint=False)
        ]
    )
    assert np.abs(wavy).max() == pytest.approx(np.deg2rad(15), rel=1e-3)
    # Two full cycles of deviation per wingbeat: four sign changes.
    crossings = int((np.diff(np.sign(wavy)) != 0).sum())
    assert crossings == 4, crossings


@needs_model
def test_realistic_kinematics_cut_the_torque_swing_and_still_fly_worse(
    rigid, wing, tmp_path
):
    """A hypothesis of this project's own, refuted with the controls attached.

    The reasoning was that the within-stroke torque swing -- which runs from
    -28 to +27 about a near-zero mean, against a control authority of 8 -- is
    what ends the flight, and that a more realistic stroke would shrink it.
    The first half is right: a sharper sweep cuts the swing by 20%. The second
    half is wrong in the opposite direction. Attitude hold goes from 187 ms to
    20.

    Two confounds were ruled out rather than argued away. The trim was
    re-measured for each stroke shape and moves by less than half a degree.
    The lost lift was restored by raising the amplitude to 79.4 degrees, which
    recovers the force exactly and the attitude hold not at all -- 19 ms.

    So the swing is not what limits the flight, and what does is still open.
    """
    from wingloop.body.control import HaltereController
    from wingloop.body.flight import harmonic_stroke

    free = add_free_base(rigid[0], tmp_path / "shape.xml", dofs="free")
    body = FlightBody(free, wing, timestep=2e-5)

    def swing(**kwargs) -> float:
        torques = []
        for t in np.linspace(0, 1 / 218.0, 300, endpoint=False):
            angles, rates = harmonic_stroke(
                t, frequency=218.0, bias=np.deg2rad(-10.7), rates=True, **kwargs
            )
            body.set_wings(angles, rates)
            body.apply_aerodynamics()
            torques.append(body.wrench_about_com()[4])
        return float(np.ptp(torques))

    assert swing(sharpness=0.9) < 0.85 * swing()

    def holds_until(**kwargs) -> float:
        b = FlightBody(free, wing, timestep=2e-5)
        trace = HaltereController(**kwargs).fly(b, 0.25)
        bad = np.degrees(np.maximum(np.abs(trace["pitch"]), np.abs(trace["roll"])))
        over = np.flatnonzero(bad > 30.0)
        return float(trace["t"][over[0]] * 1000) if len(over) else 250.0

    plain = holds_until()
    sharp = holds_until(sharpness=0.9)
    assert plain > 200.0, plain
    # Still a penalty, and much smaller than it first looked. Measured with
    # the 20 ms sensor filter the sharp stroke held for 20 ms against 187 --
    # a factor of nine. Most of that was the filter: at 5 ms it is 171
    # against 250, a factor of 1.5. The first measurement was right about the
    # sign and wrong about the size by six times.
    assert sharp < 0.85 * plain, (plain, sharp)


@needs_model
def test_wake_memory_changes_the_forces_barely_and_the_flight_a_lot(
    rigid, wing, tmp_path
):
    """The wake suspicion, refuted -- and the thing it led to instead.

    The suspicion was that a quasi-steady model, which applies the steady-state
    force at every instant, is missing what a sharper stroke needs: circulation
    that takes a couple of chord lengths to build and carries across a
    reversal. It is missing that, and it does not matter. Over a cycle the mean
    lift is identical to two decimals and the torque swing moves 4%, in the
    wrong direction.

    What it does do is end the flight, and that is the informative part: the
    lag sits inside the attitude feedback loop. Adding delay there costs phase
    margin, and the flight collapses from 300 ms to under 50 -- while the
    aerodynamics it changed are the same aerodynamics. That is what identified
    the real limit, and dropping the sensor filter from 20 ms to 5 then nearly
    doubled the controlled flight.
    """
    from wingloop.body.control import HaltereController
    from wingloop.body.flight import harmonic_stroke

    free = add_free_base(rigid[0], tmp_path / "wake.xml", dofs="free")

    def cycle_mean_lift(wake: bool) -> tuple[float, float]:
        body = FlightBody(free, wing, timestep=2e-5, wake_memory=wake)
        vertical, torque = [], []
        for t in np.linspace(0, 3 / 218.0, 900, endpoint=False):
            angles, rates = harmonic_stroke(
                t, frequency=218.0, bias=np.deg2rad(-10.7), rates=True
            )
            body.set_wings(angles, rates)
            body.apply_aerodynamics()
            if t > 1 / 218.0:  # let the lag settle
                w = body.wrench_about_com()
                vertical.append(w[2])
                torque.append(w[4])
        return float(np.mean(vertical)), float(np.ptp(torque))

    plain_lift, plain_swing = cycle_mean_lift(False)
    lagged_lift, lagged_swing = cycle_mean_lift(True)

    # The aerodynamics barely notice.
    assert lagged_lift == pytest.approx(plain_lift, rel=0.02)
    assert lagged_swing == pytest.approx(plain_swing, rel=0.10)
    # And it is not the reduction the suspicion predicted.
    assert lagged_swing > plain_swing

    def holds_until(wake: bool) -> float:
        body = FlightBody(free, wing, timestep=2e-5, wake_memory=wake)
        trace = HaltereController().fly(body, 0.30)
        bad = np.degrees(np.maximum(np.abs(trace["pitch"]), np.abs(trace["roll"])))
        over = np.flatnonzero(bad > 30.0)
        return float(trace["t"][over[0]] * 1000) if len(over) else 300.0

    # The flight, however, notices a great deal: same forces, added delay.
    # Measured 300 ms against 104 with the filter at its tuned 5 ms, and
    # 187 against 27 back when the filter itself was costing most of the
    # margin -- the ratio survives the retuning, which is the point.
    plain_flight = holds_until(False)
    assert plain_flight > 250.0
    assert holds_until(True) < 0.5 * plain_flight


@needs_model
def test_less_sensor_lag_buys_more_flight(rigid, wing, tmp_path):
    """The answer the wake detour produced, and the reason it is the answer.

    Swept against loop bandwidth, controlled flight lasts 353 ms at 5 ms of
    filter lag and 187 at 20, and at every bandwidth tried more lag is worse.
    It cannot go to zero either -- within a stroke the torque swings between
    -28 and +27 about a near-zero mean, so an unfiltered loop chases the beat,
    and at 3 ms the flight is back to 146 ms. The filter trades stroke noise
    against phase margin, and the default sits where that trade was measured.
    """
    from wingloop.body.control import HaltereController

    free = add_free_base(rigid[0], tmp_path / "lag.xml", dofs="free")

    def holds_until(tau: float) -> float:
        body = FlightBody(free, wing, timestep=2e-5)
        trace = HaltereController(tau=tau).fly(body, 0.40)
        bad = np.degrees(np.maximum(np.abs(trace["pitch"]), np.abs(trace["roll"])))
        over = np.flatnonzero(bad > 30.0)
        return float(trace["t"][over[0]] * 1000) if len(over) else 400.0

    assert holds_until(0.005) > 1.5 * holds_until(0.020)
    assert holds_until(0.005) > holds_until(0.003)


@needs_model
def test_a_muscle_driven_stroke_flies(rigid, wing, tmp_path):
    """The last thing that was still being written down rather than produced.

    The sweep now comes out of a stretch-activated oscillator working against
    the air, so its amplitude and frequency are consequences of the neural
    drive and the load. The animal flies on it: attitude held past 200 ms and
    180 mm of altitude, against 148 for the prescribed sine, because the
    muscle settles on a larger stroke than the sine was told to make.

    Rotation is still commanded, which is the division the animal has: power
    muscles asynchronous, steering muscles synchronous.
    """
    from wingloop.body.control import HaltereController
    from wingloop.body.power import PowerOscillator, PowerStroke, aerodynamic_load

    free = add_free_base(rigid[0], tmp_path / "muscle.xml", dofs="free")
    oscillator = PowerOscillator(drive=3.0, load=aerodynamic_load(wing))
    oscillator.angle = np.deg2rad(1.0)

    body = FlightBody(free, wing, timestep=2e-5)
    controller = HaltereController(stroke=PowerStroke(oscillator))
    trace = controller.fly(body, 0.30)

    bad = np.degrees(np.maximum(np.abs(trace["pitch"]), np.abs(trace["roll"])))
    over = np.flatnonzero(bad > 30.0)
    held = float(trace["t"][over[0]] * 1000) if len(over) else 300.0
    assert held > 150.0, held
    assert trace["z"][-1] > 100.0, trace["z"][-1]

    # And the oscillator really did run: it is not sitting where it started.
    assert np.degrees(abs(oscillator.angle)) > 5.0
