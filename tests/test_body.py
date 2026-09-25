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
    attitude,
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

#: Switches the yaw loop off.
#:
#: Every sensing and filter comparison below uses it, and deliberately. Those
#: were measured before there was a yaw loop, in a regime where flight ended
#: at 150-350 ms and the metric could tell settings apart. With yaw held the
#: same settings all fly for a second or more and the comparison saturates --
#: which is itself the finding, recorded in
#: ``test_holding_yaw_is_worth_more_than_any_of_the_sensor_tuning``.
YAW_OFF = {"yaw_gain": 0.0, "yaw_rate_gain": 0.0}

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
    assertion should start failing if the controller ever got better. It did,
    repeatedly, and the numbers here have been rewritten each time.

    The climb is the part to read carefully. It was 148 mm over this run on
    the sinusoid and is **65 mm** on the sharper stroke that is now the
    default, because that stroke makes 12.19 of cycle-mean lift against 13.66
    -- 1.21 of body weight against 1.36. It buys attitude with lift, and this
    is where that shows. The amplitude that would buy the lift back is 79.41
    degrees; taking it is a separate change with its own measurements owing.
    """
    free = add_free_base(rigid[0], tmp_path / "loop.xml", dofs="free")
    body = FlightBody(free, wing, timestep=2e-5)
    controller = HaltereController()
    trace = controller.fly(body, 0.30)

    assert controller.diverged_at is None, "it should not blow up inside 300 ms"
    assert np.degrees(np.abs(trace["pitch"])).max() < 20.0
    assert np.degrees(np.abs(trace["roll"])).max() < 25.0
    assert trace["z"][-1] > 50.0, "and climb the whole way"


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
            SteeringController(readout=readout, bearing=bearing).fly(body, 0.45)[
                "heading"
            ][-1]
        )

    # The fly faces +x and its left is +y, so turning left is a positive turn.
    #
    # This asked for more than 100 degrees in 100 ms until yaw was stabilised,
    # which it reached because nothing was holding the heading and the phase
    # command simply span the animal. Two things have shortened it since. The
    # yaw loop moved the steering command onto the heading *setpoint*, and
    # flapping counter-torque damps a commanded turn as much as a
    # disturbance, so the window has to be longer to see anything: the left
    # and right objects are 12 degrees apart at 150 ms, 24 at 200 and 29 at
    # 300. The claim was always the paired difference; the absolute number
    # was the spin.
    left_object = heading(real, -45.0)
    right_object = heading(real, 45.0)
    assert left_object > right_object + 5.0, (left_object, right_object)

    # Without bearing information every bearing gives the same heading, and it
    # is the one the stabiliser reaches on its own.
    blind_left = heading(flat, -45.0)
    blind_right = heading(flat, 45.0)
    assert blind_left == pytest.approx(blind_right, abs=1e-6)

    free = add_free_base(rigid[0], tmp_path / "bare.xml", dofs="free")
    bare = FlightBody(free, wing, timestep=2e-5)
    bare_turn = HaltereController().fly(bare, 0.45)["heading"][-1]
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
                SteeringController(readout=readout, bearing=bearing).fly(body, 0.20)[
                    "heading"
                ][-1]
            )
        )

    # 200 ms, where it was 60. Counter-torque damps the turn, so at 60 ms the
    # three bearings are still inside the transient and come out in the wrong
    # order (+1.8, +9.2, +3.2); by 200 ms they are +20.6, +0.3, -16.9.
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
def test_realistic_kinematics_cut_the_torque_swing_and_fly_better(
    rigid, wing, tmp_path
):
    """A refutation of this project's own refutation.

    The original reasoning was that the within-stroke torque swing -- -28 to
    +27 about a near-zero mean, against a control authority of 8 -- is what
    ends the flight, and that a more realistic stroke would shrink it. A
    sharper sweep does cut the swing by 20%, and every measurement of what
    that was worth came out negative: attitude hold fell from 187 ms to 20,
    and this test was written to record the hypothesis being wrong.

    **The penalty shrank every time the loop's sensing improved** -- a factor
    of nine at 20 ms of filter lag, 1.5 at 5 ms, 1.25 under the stroke boxcar
    -- and always for the same reason, which was never the stroke. With the
    last of it gone, the sign has gone with it:

    ======  =======  =======  ======
    amp     plain    sharp    ratio
    ======  =======  =======  ======
    74.25    1473     2467     1.68
    75.00    1230     2123     1.73
    75.75    1250     1879     1.50
    ======  =======  =======  ======

    What was missing was flapping counter-torque. An animal with no passive
    yaw damping cannot bank the reduced swing, because what ends its flight is
    not the swing. Once the wings are told the body rotates, the original
    hypothesis is simply right, and it was right all along.

    ``sharpness`` is now 0.9 by default, so the *sinusoid* is the special case
    here and this test asks for it explicitly.
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

    assert swing(sharpness=0.9) < 0.85 * swing(sharpness=0.0)

    def holds_until(**kwargs) -> float:
        b = FlightBody(free, wing, timestep=2e-5)
        trace = HaltereController(**kwargs).fly(b, 3.0)
        bad = np.degrees(np.maximum(np.abs(trace["pitch"]), np.abs(trace["roll"])))
        back = np.flatnonzero(np.diff(trace["t"]) < 0)
        end = int(back[0]) + 1 if len(back) else len(trace["t"])
        bad, t = bad[:end], trace["t"][:end]
        over = np.flatnonzero(bad > 30.0)
        return float(t[over[0]] * 1000) if len(over) else float(t[-1] * 1000)

    plain = holds_until(sharpness=0.0)
    sharp = holds_until()
    assert plain > 900.0, plain
    assert sharp > 1.3 * plain, (plain, sharp)


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
        trace = HaltereController().fly(body, 1.0)
        bad = np.degrees(np.maximum(np.abs(trace["pitch"]), np.abs(trace["roll"])))
        over = np.flatnonzero(bad > 30.0)
        return float(trace["t"][over[0]] * 1000) if len(over) else 1000.0

    # The flight, however, notices a great deal: same forces, added delay.
    # Measured 300 ms against 104 with the filter at its tuned 5 ms, and
    # 187 against 27 back when the filter itself was costing most of the
    # margin -- the ratio survives the retuning, which is the point.
    plain_flight = holds_until(False)
    assert plain_flight > 900.0
    # 618 ms against the full 1000, a factor of 0.62 where this once measured
    # 0.35. Flapping counter-torque gives the animal something to fall back on
    # that the circulation lag cannot take away, so the lag costs less than it
    # did -- and still costs plainly more than the forces it changes.
    assert holds_until(True) < 0.8 * plain_flight


@needs_model
def test_less_sensor_lag_buys_more_flight(rigid, wing, tmp_path):
    """The direction is right: inside the loop, delay costs flight.

    Measured on the first-order filter this was found with. It cannot go to
    zero either -- within a stroke the torque swings between -28 and +27 about
    a near-zero mean, so an unfiltered loop chases the beat, and at 3 ms the
    flight is back to 146 ms.

    The optimum is interior, and **where it sits depends on the stroke**. On
    the sinusoid it was a broad hump around 5-6 ms. On the stroke this animal
    now flies it is a pronounced peak at 3.5 ms, with 116 ms of flight at 1 ms
    of lag and 21 at 20. So this asserts the shape -- worse on both sides --
    and not a number, because the number belongs to the stroke.

    See ``test_the_filter_optimum_moves_with_the_stroke_and_not_with_luck``
    for why the peak is believed this time.
    """
    from wingloop.body.control import (
        LOWPASS_BANDWIDTH,
        HaltereController,
        gains_for,
    )

    free = add_free_base(rigid[0], tmp_path / "lag.xml", dofs="free")

    def holds_until(tau: float) -> float:
        body = FlightBody(free, wing, timestep=2e-5)
        trace = HaltereController(
            sensing="lowpass", tau=tau, **{**gains_for(LOWPASS_BANDWIDTH), **YAW_OFF}
        ).fly(body, 1.50)
        bad = np.degrees(np.maximum(np.abs(trace["pitch"]), np.abs(trace["roll"])))
        over = np.flatnonzero(bad > 30.0)
        return float(trace["t"][over[0]] * 1000) if len(over) else 1500.0

    assert holds_until(0.0035) > holds_until(0.020)
    assert holds_until(0.0035) > holds_until(0.001)


@needs_model
def test_the_filter_optimum_moves_with_the_stroke_and_not_with_luck(
    rigid, wing, tmp_path
):
    """Third time this curve has been measured, and the first believable peak.

    The history is the point. The filter optimum was first quoted as 353 ms at
    5 ms of lag; that turned out to be a spike that **moved when the stroke
    amplitude changed by 1%**, so where it landed was a coincidence. Adding
    the missing yaw damping flattened it into a broad hump at 5-6 ms. Turning
    ``sharpness`` on moved it again, to a pronounced peak at 3.5 ms:

    ======  =====  =====  =====  =====  =====
    tau ms   2.0    2.5    3.0    3.5    4.0
    ======  =====  =====  =====  =====  =====
    74.25    338    539    932   1092    437
    75.00    300    505    847    934    459
    75.75    282    426    761    942    476
    ======  =====  =====  =====  =====  =====

    **This peak does not move with amplitude**, which is the test the first
    one failed: all three put it at 3.5 ms. It moves with ``sharpness``, to
    3.0 at 0.95, and that is a stroke parameter, so depending on it is what a
    real optimum should do. A sharper sweep carries its torque differently and
    wants less filtering; the lag then costs more, and 20 ms of it is worth
    21 ms of flight.

    So this asserts the discriminator directly: three amplitudes, one peak.
    """
    from wingloop.body.control import (
        LOWPASS_BANDWIDTH,
        HaltereController,
        gains_for,
    )

    free = add_free_base(rigid[0], tmp_path / "spike.xml", dofs="free")

    def holds_until(tau: float, amplitude: float) -> float:
        body = FlightBody(free, wing, timestep=2e-5)
        trace = HaltereController(
            sensing="lowpass",
            tau=tau,
            amplitude=np.deg2rad(amplitude),
            **{**gains_for(LOWPASS_BANDWIDTH), **YAW_OFF},
        ).fly(body, 2.0)
        bad = np.degrees(np.maximum(np.abs(trace["pitch"]), np.abs(trace["roll"])))
        over = np.flatnonzero(bad > 30.0)
        return float(trace["t"][over[0]] * 1000) if len(over) else 2000.0

    taus = (0.0025, 0.003, 0.0035, 0.004)
    peaks = []
    for amplitude in (74.25, 75.0, 75.75):
        curve = [holds_until(tau, amplitude) for tau in taus]
        # It climbs to the peak rather than jumping to it: a spike stands out
        # from both neighbours, this one has a side.
        assert curve[0] < curve[1] < curve[2], curve
        peaks.append(taus[int(np.argmax(curve))])
    assert len(set(peaks)) == 1, (
        f"the peak moved with stroke amplitude ({peaks}), which is what the "
        "first version of this curve did and the reason its optimum was not "
        "believed. If this fires, the 3.5 ms figure needs withdrawing."
    )


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


@needs_model
def test_the_connectome_command_sets_how_hard_it_flies(rigid, wing, tmp_path):
    """The whole stack, end to end: a command in the brain, altitude in a body.

    A flight command drives DNa08 and DNp31, the power motor neurons follow,
    their activation becomes oscillator drive, the oscillator sets the stroke,
    and the stroke decides whether the animal goes up or down. Nothing between
    the command and the altitude is written down except the one calibration
    that turns activation into drive.

    On a vertical rail, which asks for force without asking for balance, the
    measured result over 200 ms is a throttle with a hover point on it:

    ======= ========= =========
    command amplitude 200 ms
    ======= ========= =========
    0.25      24.6 deg  -169 mm
    0.50      48.7 deg   -93 mm
    0.75      63.5 deg   +1.8 mm
    1.00      71.2 deg  +100 mm
    ======= ========= =========

    The fly hovers near three quarters of the command it has. That is a
    consequence of the connectome curve and the calibration, not something
    aimed at, and it is the reason the calibration was left where it is.
    """
    from wingloop.body.power import PowerOscillator, PowerStroke, aerodynamic_load
    from wingloop.brain.readout import PowerReadout

    stored = dict(np.load(Path(__file__).parent / "power_command.npz"))
    readout = PowerReadout(
        command=stored["command"],
        activation=stored["activation"],
        steering=stored["steering"],
    )
    rail = add_free_base(rigid[0], tmp_path / "throttle.xml", dofs="z")
    load = aerodynamic_load(wing)

    def climb(command: float) -> tuple[float, float]:
        oscillator = PowerOscillator(drive=readout.drive(command), load=load)
        oscillator.angle = np.deg2rad(1.0)
        body = FlightBody(rail, wing, timestep=2e-5)
        generator = PowerStroke(oscillator)
        peak = 0.0
        for _ in range(int(0.20 / body.model.opt.timestep)):
            angles, rates = generator(float(body.model.opt.timestep))
            body.set_wings(angles, rates)
            body.apply_aerodynamics()
            body._mj.mj_step(body.model, body.data)
            peak = max(peak, abs(oscillator.angle))
        return float(body.data.qpos[0]), float(np.degrees(peak))

    heights, amplitudes = zip(
        *(climb(c) for c in (0.25, 0.5, 0.75, 1.0)), strict=True
    )

    # More command is more stroke is more height, with no exceptions.
    assert all(b > a for a, b in zip(amplitudes[:-1], amplitudes[1:], strict=True))
    assert all(b > a for a, b in zip(heights[:-1], heights[1:], strict=True))

    # And the throttle crosses weight between half and full command.
    assert heights[0] < 0 and heights[-1] > 0
    assert heights[1] < 0 < heights[3]


@needs_model
def test_no_command_is_no_flight(rigid, wing, tmp_path):
    """The control for the test above: the rail only goes up when asked.

    Without this, a climb would be evidence of nothing -- a body that rises
    whatever the brain says is not being flown by it.
    """
    from wingloop.body.power import PowerOscillator, PowerStroke, aerodynamic_load
    from wingloop.brain.readout import PowerReadout

    stored = dict(np.load(Path(__file__).parent / "power_command.npz"))
    readout = PowerReadout(
        command=stored["command"], activation=stored["activation"]
    )
    assert readout.drive(0.0) == pytest.approx(0.0, abs=1e-9)

    rail = add_free_base(rigid[0], tmp_path / "idle.xml", dofs="z")
    oscillator = PowerOscillator(drive=readout.drive(0.0), load=aerodynamic_load(wing))
    oscillator.angle = np.deg2rad(1.0)
    body = FlightBody(rail, wing, timestep=2e-5)
    generator = PowerStroke(oscillator)
    for _ in range(int(0.05 / body.model.opt.timestep)):
        angles, rates = generator(float(body.model.opt.timestep))
        body.set_wings(angles, rates)
        body.apply_aerodynamics()
        body._mj.mj_step(body.model, body.data)

    assert np.degrees(abs(oscillator.angle)) < 1.0, "an undriven muscle decays"
    assert body.data.qpos[0] < 0.0, "and the fly falls"


@needs_model
def test_the_throttle_loop_holds_a_height_that_no_fixed_command_holds(
    rigid, wing, tmp_path
):
    """Closing the second loop: how high it is, back onto how hard it flies.

    The control is the point. A *held* command cannot hold a height, even the
    right one: at the measured hover command the animal still sinks 64 mm over
    600 ms, because it spends the first wingbeats falling while the muscle
    spins up and proportional-to-nothing never makes that back. Below it sinks
    412 mm and above it climbs 324. Closed, the same body holds zero to within
    0.04 mm.
    """
    from wingloop.body.control import HOVER_COMMAND, HaltereController, Throttle
    from wingloop.body.power import PowerOscillator, PowerStroke, aerodynamic_load
    from wingloop.brain.readout import PowerReadout

    stored = dict(np.load(Path(__file__).parent / "power_command.npz"))
    readout = PowerReadout(command=stored["command"], activation=stored["activation"])
    rail = add_free_base(rigid[0], tmp_path / "throttle_loop.xml", dofs="z")
    load = aerodynamic_load(wing)

    def fly(seconds=0.6, **throttle):
        oscillator = PowerOscillator(drive=0.0, load=load)
        oscillator.angle = np.deg2rad(1.0)
        body = FlightBody(rail, wing, timestep=2e-5)
        controller = HaltereController(
            stroke=PowerStroke(oscillator),
            throttle=Throttle(readout=readout, oscillator=oscillator, **throttle),
        )
        return controller.fly(body, seconds)

    # Open loop, including at the command that hovers: height is whatever the
    # command made it.
    assert fly(held=0.60)["z"][-1] < -300.0
    assert fly(held=0.80)["z"][-1] > 250.0
    assert fly(held=HOVER_COMMAND)["z"][-1] < -50.0

    # Closed, on the same body.
    trace = fly(target=0.0)
    late = trace["t"] > 0.5
    assert np.abs(trace["z"][late]).max() < 1.0, np.abs(trace["z"][late]).max()
    assert trace["z"].min() > -10.0, "and it barely dips while the muscle starts"

    # The loop really is working the connectome channel, not a constant.
    assert trace["throttle"].max() > trace["throttle"].min() + 0.1
    assert trace["throttle"][-1] == pytest.approx(HOVER_COMMAND, abs=0.02)


@needs_model
def test_it_climbs_to_a_commanded_height_without_overshooting(rigid, wing, tmp_path):
    """Critically damped on purpose: an altitude loop that overshoots down has
    a floor to hit. Asked for 30 mm from a standing start it arrives in 219 ms
    and goes past by 0.03."""
    from wingloop.body.control import HaltereController, Throttle
    from wingloop.body.power import PowerOscillator, PowerStroke, aerodynamic_load
    from wingloop.brain.readout import PowerReadout

    stored = dict(np.load(Path(__file__).parent / "power_command.npz"))
    readout = PowerReadout(command=stored["command"], activation=stored["activation"])
    rail = add_free_base(rigid[0], tmp_path / "step.xml", dofs="z")

    oscillator = PowerOscillator(drive=0.0, load=aerodynamic_load(wing))
    oscillator.angle = np.deg2rad(1.0)
    body = FlightBody(rail, wing, timestep=2e-5)
    controller = HaltereController(
        stroke=PowerStroke(oscillator),
        throttle=Throttle(readout=readout, oscillator=oscillator, target=30.0),
    )
    trace = controller.fly(body, 0.8)

    assert trace["z"].max() < 31.0, "overshoot"
    assert np.abs(trace["z"][trace["t"] > 0.7] - 30.0).max() < 1.0
    # It got there by flying harder and then backing off, not by drifting up.
    rising = trace["throttle"][trace["t"] < 0.1].max()
    assert rising > 0.9 and trace["throttle"][-1] < 0.75


@needs_model
def test_a_mis_measured_hover_point_costs_exactly_what_it_should(
    rigid, wing, tmp_path
):
    """What the loop leans on, stated as a number rather than trusted.

    There is no integral term, so the loop is only as accurate as
    ``HOVER_COMMAND``: a trim that is wrong by ``d`` settles at
    ``d * CLIMB_PER_COMMAND / bandwidth**2`` away from the target, below it if
    the trim is low. At 0.65 against a true 0.695 that is -2.37 mm, and the
    body settles at -2.40.

    A test that only checked "it holds a height" would pass with the trim
    wrong; this one says how wrong, so the constant cannot rot quietly.
    """
    from wingloop.body.control import (
        ALTITUDE_BANDWIDTH,
        CLIMB_PER_COMMAND,
        HOVER_COMMAND,
        HaltereController,
        Throttle,
    )
    from wingloop.body.power import PowerOscillator, PowerStroke, aerodynamic_load
    from wingloop.brain.readout import PowerReadout

    stored = dict(np.load(Path(__file__).parent / "power_command.npz"))
    readout = PowerReadout(command=stored["command"], activation=stored["activation"])
    rail = add_free_base(rigid[0], tmp_path / "trim.xml", dofs="z")
    load = aerodynamic_load(wing)

    for assumed in (0.65, 0.75):
        oscillator = PowerOscillator(drive=0.0, load=load)
        oscillator.angle = np.deg2rad(1.0)
        body = FlightBody(rail, wing, timestep=2e-5)
        controller = HaltereController(
            stroke=PowerStroke(oscillator),
            throttle=Throttle(
                readout=readout, oscillator=oscillator, target=0.0, hover=assumed
            ),
        )
        settled = float(controller.fly(body, 0.6)["z"][-1])
        predicted = (assumed - HOVER_COMMAND) * CLIMB_PER_COMMAND / ALTITUDE_BANDWIDTH**2
        assert settled == pytest.approx(predicted, abs=0.15), (assumed, settled, predicted)


@needs_model
def test_stroke_sensing_buys_bandwidth_and_bandwidth_buys_flight(
    rigid, wing, tmp_path
):
    """What the phase margin was actually spent on, and what buying it back got.

    A first-order filter rejects the stroke beat by lagging everything, and
    the lag is inside the loop. A boxcar over exactly one wingbeat has an
    exact null at the stroke frequency and at every harmonic of it -- 56 dB
    down at 218 Hz for the 229-sample window this timestep gives -- with a
    group delay of half a period, 2.29 ms against 5.

    On the sinusoid the two were level at the bandwidth the old default used
    and separated above it, which is what a phase margin looks like when it is
    spent on gain instead of lag: 236 against 237 at bandwidth 40, and 163
    against 320 at 60.

    **On the stroke this animal now flies the boxcar wins everywhere**, and by
    much more:

    ======  =================  ==============
    rad/s   first-order 6 ms   stroke boxcar
    ======  =================  ==============
    40            366 ms          1162 ms
    60            779 ms          1227 ms
    ======  =================  ==============

    That is the same effect grown rather than a different one. A sharper sweep
    puts more of its disturbance at the wingbeat and its harmonics, which is
    exactly what a one-period boxcar nulls and a first-order filter can only
    smear -- and the first-order filter's own optimum moved to 3.5 ms for the
    same reason, so 6 ms is further off its best than it used to be.
    """
    from wingloop.body.control import HaltereController, gains_for

    free = add_free_base(rigid[0], tmp_path / "sensing.xml", dofs="free")

    def holds_until(bandwidth: float, **kwargs) -> float:
        body = FlightBody(free, wing, timestep=2e-5)
        trace = HaltereController(**kwargs, **{**gains_for(bandwidth), **YAW_OFF}).fly(
            body, 1.60
        )
        bad = np.degrees(np.maximum(np.abs(trace["pitch"]), np.abs(trace["roll"])))
        over = np.flatnonzero(bad > 30.0)
        return float(trace["t"][over[0]] * 1000) if len(over) else 1600.0

    low = dict(sensing="lowpass", tau=0.006)
    box = dict(sensing="stroke")

    assert holds_until(40.0, **box) > 2.0 * holds_until(40.0, **low)
    assert holds_until(60.0, **box) > 1.3 * holds_until(60.0, **low)
    assert holds_until(60.0, **box) > holds_until(40.0, **box)


@needs_model
def test_the_boxcar_nulls_the_wingbeat_rather_than_smearing_it(rigid, wing):
    """The property the choice rests on, checked on the filter itself.

    Not a flight test: a one-period boxcar either has its null at the stroke
    frequency or it does not, and that is arithmetic on the window length.
    The first-order filter it replaces is 26 dB down at the same frequency
    while costing nearly twice the group delay.
    """
    from wingloop.body.control import HaltereController

    dt, frequency = 2e-5, 218.0
    n = int(round(1.0 / (frequency * dt)))
    assert n == 229, n

    # Response of the running mean at the wingbeat, and at its third harmonic.
    def boxcar_gain(f):
        x = np.pi * f * dt
        return abs(np.sin(n * x) / (n * np.sin(x)))

    assert boxcar_gain(frequency) < 0.01, boxcar_gain(frequency)
    assert boxcar_gain(3 * frequency) < 0.01, boxcar_gain(3 * frequency)

    # The first-order filter it replaces, for comparison, at tau = 5 ms.
    first_order = 1.0 / np.hypot(1.0, 2 * np.pi * frequency * 0.005)
    assert first_order > 10 * boxcar_gain(frequency)

    # And it really is what the controller runs: a constant survives it and a
    # signal at the wingbeat does not.
    c = HaltereController(sensing="stroke", frequency=frequency)
    steady = [c._filtered(0.3, 0.0, 0.0, np.zeros(3), dt)[0] for _ in range(3 * n)][-1]
    assert steady == pytest.approx(0.3, abs=1e-12)

    c = HaltereController(sensing="stroke", frequency=frequency)
    out = [
        c._filtered(
            np.sin(2 * np.pi * frequency * i * dt), 0.0, 0.0, np.zeros(3), dt
        )[0]
        for i in range(4 * n)
    ]
    assert max(abs(v) for v in out[2 * n :]) < 0.02


@needs_model
def test_the_pitch_tether_pins_through_the_centre_of_mass(rigid, wing, tmp_path):
    """A rig failure that reads exactly like a control failure.

    The centre of mass is 1.07 mm above the model origin and 0.30 mm behind
    it. Pinned at the origin, the net aerodynamic force stops accelerating the
    animal and starts torquing it about the pin with an arm the trim knob
    cannot reach, and the tethered fly spins -- continuously, at 80-107 Hz, at
    every filter setting and every gain that was tried. It took a sweep that
    failed everywhere to notice the rig rather than the loop.

    So this asserts where the pin is, which is the thing that was wrong.
    """
    from wingloop.body.hinge import _centre_of_mass

    com = _centre_of_mass(Path(rigid[0]), "FlyBody")
    assert com[2] == pytest.approx(1.0666, abs=0.01), com
    assert com[0] == pytest.approx(-0.3040, abs=0.01), com

    tether = add_free_base(rigid[0], tmp_path / "tether.xml", dofs="pitch")
    body = FlightBody(tether, wing, timestep=2e-5)
    assert body.model.nv == 1, "a pitch tether leaves exactly one freedom"
    assert body.root_body is not None, "and the wrench still has a body to act on"
    assert body.root_translation == 0, "but there is no position to read"

    # The hinge is the pitch axis and only the pitch axis.
    body.data.qpos[body.root_dof] = 0.1
    mujoco.mj_forward(body.model, body.data)
    pitch, roll = attitude(body)
    assert np.degrees(pitch) == pytest.approx(5.73, abs=0.01)
    assert np.degrees(roll) == pytest.approx(0.0, abs=1e-9)


@needs_model
def test_holding_yaw_helps_and_much_less_than_it_first_appeared(
    rigid, wing, tmp_path
):
    """The loop that was missing, and what it is actually worth.

    Yaw was the one axis with nothing on it, and the light one -- inertia
    0.000591 against 0.002014 in pitch -- so it ran away fastest: a flight
    reached +40 degrees of heading by 100 ms and -86 by 300, at up to 1587
    deg/s, while pitch and roll stayed inside ten. Closing it took the flight
    from 320 ms to 1004, and this test said it tripled the flight.

    **That factor was mostly two other things.** The wings had never been told
    the body was rotating, so the uncontrolled animal had no yaw damping to
    fall back on; and the standing roll offset was making a yaw torque through
    sideslip that nothing was answering. With both fixed the uncontrolled
    animal flies 1020-1140 ms on its own, and the loop is worth **1.15 to
    1.29** on top -- consistently, at every amplitude, and nothing like a
    tripling.

    A loop measured against a broken plant flatters itself. The claim that
    survives is that it helps and that the heading stays a heading.
    """
    from wingloop.body.control import HaltereController

    free = add_free_base(rigid[0], tmp_path / "yaw.xml", dofs="free")

    def fly(seconds, **kw):
        """Returns the trace cut to the samples that are still a flight.

        Cut by **index**, not by time. MuJoCo resets the clock when it
        diverges, so post-divergence samples carry small times and sail
        through a ``t <= held`` mask while the recorded heading keeps
        accumulating -- which reported 9000 degrees of spin inside a flight
        whose heading never left ten. ``fly`` stops at the reset, so the last
        good index is the first place time goes backwards.
        """
        body = FlightBody(free, wing, timestep=2e-5)
        controller = HaltereController(**kw)
        trace = controller.fly(body, seconds)
        backwards = np.flatnonzero(np.diff(trace["t"]) < 0)
        end = int(backwards[0]) + 1 if len(backwards) else len(trace["t"])
        trace = {k: v[:end] for k, v in trace.items()}
        bad = np.degrees(np.maximum(np.abs(trace["pitch"]), np.abs(trace["roll"])))
        over = np.flatnonzero(bad > 30.0)
        end = int(over[0]) + 1 if len(over) else end
        trace = {k: v[:end] for k, v in trace.items()}
        return trace, float(trace["t"][-1] * 1000)

    _, loose = fly(2.5, **YAW_OFF)
    held_trace, held = fly(2.5)
    assert held > 1.1 * loose, (loose, held)
    assert held > 1100.0, held

    # And the heading it holds is a heading, not a slow spin: inside ten
    # degrees at 200, 400, 600 and 800 ms, departing only in the last fifty
    # milliseconds as the attitude goes.
    for ms in (200, 400, 600, 800):
        i = int(np.argmin(np.abs(held_trace["t"] - ms / 1000.0)))
        assert abs(held_trace["heading"][i]) < 20.0, (ms, held_trace["heading"][i])


@needs_model
def test_only_the_phase_knob_yaws_and_it_does_so_linearly(rigid, wing):
    """The authority the yaw gains are derived from, and the knob's isolation.

    Rotation phase is the yaw control and amplitude is the roll control -- a
    division the steering work found and this measures directly. Amplitude
    asymmetry makes exactly zero yaw; the symmetric phase shift makes 0.0004;
    the phase *asymmetry* makes 0.5666 per radian, odd in the knob and linear
    to 3.6% of full scale.
    """
    from wingloop.body.control import TRIM_BIAS, YAW_PER_PHASE
    from wingloop.body.flight import harmonic_stroke

    body = FlightBody(
        add_free_base(rigid[0], Path(MESH).parent / "yawauth.xml", dofs="free"),
        wing,
        timestep=2e-5,
    )

    def mean_torque(**kw):
        acc = []
        for t in np.linspace(0, 1 / 218.0, 240, endpoint=False):
            angles, rates = harmonic_stroke(
                t, amplitude=np.deg2rad(75.0), frequency=218.0,
                bias=TRIM_BIAS, rates=True, **kw,
            )
            body.set_wings(angles, rates)
            body.apply_aerodynamics()
            acc.append(body.wrench_about_com()[3:6])
        return np.asarray(acc).mean(axis=0)

    assert mean_torque()[2] == pytest.approx(0.0, abs=1e-6)
    # Amplitude is the roll knob and makes no yaw whatever.
    assert mean_torque(asymmetry=0.2)[2] == pytest.approx(0.0, abs=1e-6)
    assert abs(mean_torque(asymmetry=0.2)[0]) > 5.0
    # The symmetric phase shift is not a yaw control either.
    assert abs(mean_torque(phase=0.2)[2]) < 0.001

    values = np.array([-0.3, -0.15, 0.15, 0.3])
    yaw = np.array([mean_torque(phase_asymmetry=v)[2] for v in values])
    assert np.allclose(yaw, -yaw[::-1], atol=1e-6), "should be odd in the knob"
    fit = np.polyfit(values, yaw, 1)[0]
    assert fit == pytest.approx(YAW_PER_PHASE, rel=0.05), (fit, YAW_PER_PHASE)


@needs_model
def test_the_phase_knob_means_the_same_thing_in_both_stroke_generators(
    rigid, wing, tmp_path
):
    """A bug the yaw loop found, because it was the first thing to use the knob.

    ``PowerStroke`` added ``phase_asymmetry`` to a saturating velocity proxy
    where ``harmonic_stroke`` rotates it inside the cosine. Same name, a
    different physical quantity: **opposite in sign and thirty times larger**.
    Nothing had noticed, because until there was a yaw loop nothing read the
    knob's sign -- and then a loop with gains measured on one generator was
    positive feedback on the other, and the muscle-driven flight dropped from
    338 ms to 54.
    """
    from wingloop.body.control import TRIM_BIAS
    from wingloop.body.flight import harmonic_stroke
    from wingloop.body.power import PowerOscillator, PowerStroke, aerodynamic_load

    free = add_free_base(rigid[0], tmp_path / "convention.xml", dofs="free")
    body = FlightBody(free, wing, timestep=2e-5)

    oscillator = PowerOscillator(drive=3.0, load=aerodynamic_load(wing))
    oscillator.angle = np.deg2rad(1.0)
    generator = PowerStroke(oscillator)
    for _ in range(int(0.25 / 2e-5)):
        generator(2e-5)

    def muscle_yaw(phase_asymmetry):
        acc = []
        for _ in range(int(3 / oscillator.frequency / 2e-5)):
            angles, rates = generator(2e-5, phase_asymmetry=phase_asymmetry)
            body.set_wings(angles, rates)
            body.apply_aerodynamics()
            acc.append(body.wrench_about_com()[5])
        return float(np.mean(acc))

    def harmonic_yaw(phase_asymmetry):
        acc = []
        for t in np.linspace(0, 1 / 218.0, 240, endpoint=False):
            angles, rates = harmonic_stroke(
                t, amplitude=np.deg2rad(75.0), frequency=218.0, bias=TRIM_BIAS,
                rates=True, phase_asymmetry=phase_asymmetry,
            )
            body.set_wings(angles, rates)
            body.apply_aerodynamics()
            acc.append(body.wrench_about_com()[5])
        return float(np.mean(acc))

    for knob in (-0.3, -0.15, 0.15, 0.3):
        a, b = muscle_yaw(knob), harmonic_yaw(knob)
        assert np.sign(a) == np.sign(b), (knob, a, b)
        assert 0.5 < abs(a) / abs(b) < 2.0, (knob, a, b)


@needs_model
def test_a_spinning_body_damps_itself_through_its_own_wings(rigid, wing, tmp_path):
    """Flapping counter-torque, and the hole it was filling.

    A wing sits out at the radius of gyration, so when the body rotates that
    point moves at ``omega x r`` on top of the body's translation: one wing
    advances into the air and the other retreats, the forces stop balancing,
    and the difference opposes the spin. It is the dominant passive damping on
    insect yaw.

    This model did not have it. The wings were told the body's *translational*
    velocity and nothing else, so an imposed spin of 2000 deg/s produced
    **0.0000** of yaw torque -- measured, not estimated. Yaw is also the only
    axis with no restoring term of its own, pitch and roll having gravity and
    the stroke plane, so it was the one axis where the omission was fatal: the
    animal flew for a second and then entered a flat spin that nothing at all
    resisted.
    """
    from wingloop.body.control import TRIM_BIAS, YAW_INERTIA
    from wingloop.body.flight import harmonic_stroke

    free = add_free_base(rigid[0], tmp_path / "fct.xml", dofs="free")
    body = FlightBody(free, wing, timestep=2e-5)

    def yaw_torque(spin: float) -> float:
        body.data.qvel[body.root_dof + 5] = spin
        mujoco.mj_forward(body.model, body.data)
        acc = []
        for t in np.linspace(0, 1 / 218.0, 240, endpoint=False):
            angles, rates = harmonic_stroke(
                t, amplitude=np.deg2rad(75.0), frequency=218.0,
                bias=TRIM_BIAS, rates=True,
            )
            body.set_wings(angles, rates)
            body.apply_aerodynamics()
            acc.append(body.wrench_about_com()[5])
        body.data.qvel[body.root_dof + 5] = 0.0
        return float(np.mean(acc))

    assert yaw_torque(0.0) == pytest.approx(0.0, abs=1e-6), "no spin, no torque"

    spins = np.array([np.deg2rad(d) for d in (250.0, 500.0, 1000.0, 2000.0)])
    torques = np.array([yaw_torque(s) for s in spins])
    # It opposes, always.
    assert np.all(torques < 0.0), torques
    assert np.all(np.array([yaw_torque(-s) for s in spins]) > 0.0)
    # And it is linear in the spin rate, which is what makes it a damping
    # coefficient rather than a nonlinearity that happens to point the right
    # way.
    coefficient = torques / spins
    assert np.allclose(coefficient, coefficient[0], rtol=0.02), coefficient

    # The time constant it gives yaw. Tens of milliseconds is the right order
    # for a fly, and the number is what makes the spin self-arresting.
    tau = YAW_INERTIA / abs(float(coefficient.mean()))
    assert 0.005 < tau < 0.05, tau


@needs_model
def test_the_wings_are_told_about_rotation_only_through_their_own_offset(
    rigid, wing, tmp_path
):
    """Where the damping comes from, so it cannot be mistaken for a fudge.

    There is no damping coefficient anywhere in this: the torque falls out of
    putting the wing's own position into the oncoming air. With the body still
    the two agree exactly; with it spinning they differ by ``omega x r``, and
    the two wings differ from each other in sign.
    """
    free = add_free_base(rigid[0], tmp_path / "offset.xml", dofs="free")
    body = FlightBody(free, wing, timestep=2e-5)
    body.set_wings(*harmonic_stroke(0.0, frequency=FREQUENCY, rates=True))

    poses = {w: body.wing_pose(w, body.wing_angles) for w in WINGS}

    # Still: the air the wing meets is the air the body meets.
    for wingname, (rot, hinge) in poses.items():
        assert np.allclose(
            body.air_velocity_at(wingname, rot, hinge), body.body_velocity()
        ), wingname

    # Spinning: each wing sees something different, and it splits into exactly
    # the two parts the geometry predicts.
    body.data.qvel[body.root_dof + 5] = np.deg2rad(1000.0)
    mujoco.mj_forward(body.model, body.data)
    extra = {
        w: body.air_velocity_at(w, *poses[w]) - body.body_velocity() for w in WINGS
    }
    assert np.linalg.norm(extra["LWing"]) > 1.0

    differential = extra["LWing"] - extra["RWing"]
    common = 0.5 * (extra["LWing"] + extra["RWing"])
    # The wings are separated across the body, so a yaw spin sweeps one
    # forward and the other back. That difference is the counter-torque, and
    # it lies along the fore-aft axis.
    assert abs(differential[0]) > 10.0 * abs(differential[1]), differential
    assert np.sign(extra["LWing"][0]) != np.sign(extra["RWing"][0])
    # They also share a part, because the centre of mass is 0.30 mm behind the
    # wing line and a spin about it carries both wings sideways together. That
    # part is lateral and it cancels between the sides, which is why it is not
    # what damps the spin.
    assert abs(common[1]) > 10.0 * abs(common[0]), common


@needs_model
def test_a_rolled_animal_that_is_climbing_yaws(rigid, wing, tmp_path):
    """The standing disturbance that was eating the yaw loop's authority.

    Neither roll nor climb does much on its own. Together they make a large
    standing yaw torque, because a rolled animal moving through air has
    sideslip. This fly flies permanently rolled -- the roll loop is
    proportional-derivative against the +0.6 the pitch trim cross-couples into
    it -- so the torque is always there, and the yaw knob was holding a
    standing 25 degrees against it, over half its range, before any
    disturbance arrived.

    That is why ``roll_integral_gain`` is no longer zero, and the docstring
    there is the rest of the story.
    """
    from wingloop.body.control import TRIM_BIAS
    from wingloop.body.flight import harmonic_stroke

    free = add_free_base(rigid[0], tmp_path / "sideslip.xml", dofs="free")
    body = FlightBody(free, wing, timestep=2e-5)

    def yaw_torque(roll: float, climb: float) -> float:
        quat = np.zeros(4)
        mujoco.mju_euler2Quat(quat, np.array([roll, 0.0, 0.0]), "xyz")
        body.data.qpos[body.root_dof + 3 : body.root_dof + 7] = quat
        body.data.qvel[body.root_dof : body.root_dof + 3] = [0.0, 0.0, climb]
        mujoco.mj_forward(body.model, body.data)
        acc = []
        for t in np.linspace(0, 1 / 218.0, 240, endpoint=False):
            angles, rates = harmonic_stroke(
                t, amplitude=np.deg2rad(75.0), frequency=218.0,
                bias=TRIM_BIAS, rates=True,
            )
            body.set_wings(angles, rates)
            body.apply_aerodynamics()
            acc.append(body.wrench_about_com()[5])
        return float(np.mean(acc))

    rolled = np.deg2rad(5.0)
    assert abs(yaw_torque(0.0, 0.0)) < 1e-6
    assert abs(yaw_torque(rolled, 0.0)) < 0.01, "roll alone does almost nothing"
    assert abs(yaw_torque(0.0, 1500.0)) < 1e-6, "climbing alone does nothing"
    together = yaw_torque(rolled, 1500.0)
    assert abs(together) > 0.1, together
    # It is the product that matters, not either factor: the pair is more than
    # twenty times the sum of the two on their own.
    alone = abs(yaw_torque(rolled, 0.0)) + abs(yaw_torque(0.0, 1500.0))
    assert abs(together) > 20.0 * alone, (together, alone)
    # And it reverses with the roll, so it is a sideslip term and not an
    # offset that happens to be there.
    assert np.sign(yaw_torque(-rolled, 1500.0)) != np.sign(together)


def test_the_yaw_knob_saturates_early_and_that_is_a_known_cost():
    """The yaw loop is bang-bang for most of a flight, on purpose.

    Yaw carries a third of pitch's inertia, so the shared bandwidth puts the
    gain high enough that twelve degrees of heading error already asks for the
    whole knob. That was investigated as the cause of the flat spin and is
    not: giving yaw its own lower bandwidth widens the linear range exactly as
    the arithmetic says and buys no flight at all -- 1206 ms at bandwidth 20
    where the knob stays linear to 108 degrees, against 1218 at the shared 60
    where it saturates at 12. Below 60 the loop stops saturating and starts
    drifting, and the heading wanders further for it.

    No simulation here: this pins the number so the trade stays visible.
    """
    from wingloop.body.control import (
        BANDWIDTH,
        MAX_PHASE,
        YAW_INERTIA,
        YAW_PER_PHASE,
        HaltereController,
        gains_for,
    )

    controller = HaltereController()
    assert controller.yaw_gain == pytest.approx(gains_for(BANDWIDTH)["yaw_gain"])
    saturates_at = np.degrees(MAX_PHASE / controller.yaw_gain)
    assert saturates_at == pytest.approx(12.0, abs=0.5), saturates_at

    # And it is the light inertia that does it, not the knob being weak: at
    # pitch's inertia the same bandwidth would stay linear three times further.
    as_heavy = YAW_INERTIA * BANDWIDTH**2 / abs(YAW_PER_PHASE) * (0.002014 / YAW_INERTIA)
    assert np.degrees(MAX_PHASE / as_heavy) < saturates_at / 3.0


@needs_model
def test_the_yaw_knob_drags_roll_with_it_and_the_loop_is_told_in_advance(
    rigid, wing, tmp_path
):
    """The cross-coupling, and the compensation that was expected not to matter.

    Deflecting the rotation-phase asymmetry to yaw the animal also rolls it,
    and not symmetrically: +1.51 of roll torque at +45 degrees against +0.33
    at -45, so the even part does not cancel between the sides. In the roll
    loop's own units that is 0.039 of amplitude asymmetry against a limit of
    0.45 -- one tenth of the authority available, which is why cancelling it
    looked pointless.

    It is worth 1218 ms of flight to 1377, at every stroke amplitude tried.
    The size was never the point: the yaw knob saturates for long stretches,
    so the roll it drags arrives as a step the roll loop can only answer after
    its own filter has seen it, and a known disturbance fed forward skips that
    delay for free.

    Flight times are too noisy here to assert a 13% difference from one run,
    so what is asserted is the mechanism: the curve is real, and the
    controller subtracts it.
    """
    from wingloop.body.control import (
        ROLL_FROM_PHASE,
        ROLL_PER_ASYMMETRY,
        TRIM_BIAS,
        HaltereController,
    )
    from wingloop.body.flight import harmonic_stroke

    free = add_free_base(rigid[0], tmp_path / "xc.xml", dofs="free")
    body = FlightBody(free, wing, timestep=2e-5)

    def roll_torque(phase_asymmetry: float) -> float:
        acc = []
        for t in np.linspace(0, 1 / 218.0, 240, endpoint=False):
            angles, rates = harmonic_stroke(
                t, amplitude=np.deg2rad(75.0), frequency=218.0, bias=TRIM_BIAS,
                rates=True, phase_asymmetry=phase_asymmetry,
            )
            body.set_wings(angles, rates)
            body.apply_aerodynamics()
            acc.append(body.wrench_about_com()[3])
        return float(np.mean(acc))

    # The coupling is real, large, and lopsided.
    base = roll_torque(0.0)
    plus = roll_torque(np.deg2rad(45.0)) - base
    minus = roll_torque(np.deg2rad(-45.0)) - base
    assert plus > 1.4, plus
    assert 0.2 < minus < 0.5, minus
    assert plus > 3.0 * minus, "the even part is what does not cancel"

    # And the stored curve is the measurement, so it matches at every knot.
    for degrees in (-45.0, -22.5, 0.0, 22.5, 45.0):
        p = np.deg2rad(degrees)
        predicted = float(np.interp(p, *ROLL_FROM_PHASE))
        assert predicted == pytest.approx(roll_torque(p) - base, abs=0.01), degrees

    # The controller subtracts exactly that, and only when asked to.
    class Fixed(HaltereController):
        def _filtered(self, pitch, roll, yaw, rate, dt):
            return 0.0, 0.0, np.deg2rad(20.0), np.zeros(3)

    on = Fixed(compensate_yaw_roll=True).knobs(body)
    off = Fixed(compensate_yaw_roll=False).knobs(body)
    assert on["phase_asymmetry"] == pytest.approx(off["phase_asymmetry"])
    p = on["phase_asymmetry"]
    expected = float(np.interp(p, *ROLL_FROM_PHASE)) / ROLL_PER_ASYMMETRY
    assert off["asymmetry"] - on["asymmetry"] == pytest.approx(expected, rel=1e-9)
    # It vanishes where there is nothing to cancel.
    assert float(np.interp(0.0, *ROLL_FROM_PHASE)) == 0.0
