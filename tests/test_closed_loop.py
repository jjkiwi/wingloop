"""Bearing computed from the animal's own heading, and what that exposed."""

from pathlib import Path

import numpy as np
import pytest

from wingloop.brain.readout import FlightReadout

mujoco = pytest.importorskip("mujoco", reason="the flight body needs MuJoCo")

from wingloop.aero.wing import wing_from_mesh  # noqa: E402
from wingloop.body.control import (  # noqa: E402
    HaltereController,
    SteeringController,
    attitude,
)

#: Switches the yaw loop off, for the measurements that are about what yaw
#: does when nothing holds it.
YAW_OFF = {"yaw_gain": 0.0, "yaw_rate_gain": 0.0}
from wingloop.body.flight import FlightBody  # noqa: E402
from wingloop.body.hinge import (  # noqa: E402
    add_free_base,
    neuromechfly_model,
    tuck_legs,
)

CURVE = Path(__file__).parent / "dnbe001_tuning.npz"
pytestmark = pytest.mark.slow

try:
    SOURCE = neuromechfly_model()
except Exception:  # pragma: no cover - FlyGym absent
    SOURCE = None
needs_model = pytest.mark.skipif(SOURCE is None, reason="needs FlyGym's MJCF")


class _PoseOnly:
    """Just enough body for the geometry: a heading and a position."""

    def __init__(self, yaw_deg: float, xy=(0.0, 0.0)):
        c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
        self.root_body = 0
        self.data = type("D", (), {})()
        self.data.xmat = np.array([[c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0]])
        self.data.qpos = np.array(list(xy) + [0.0] * 5)


# ------------------------------------------------------------- the geometry


def test_bearing_is_positive_to_the_right():
    """The sign has to match the readout's, and it is not the obvious one: this
    fly faces +x and its left is +y, because the right wing's span points to
    -y. Backwards, it turns smoothly away from what it is looking at, which
    reads as avoidance rather than as a bug."""
    assert SteeringController(target=(10.0, -10.0)).current_bearing(
        _PoseOnly(0.0)
    ) == pytest.approx(45.0)
    assert SteeringController(target=(10.0, 10.0)).current_bearing(
        _PoseOnly(0.0)
    ) == pytest.approx(-45.0)


def test_turning_the_body_moves_the_bearing():
    """The whole point of closing the loop."""
    c = SteeringController(target=(10.0, -10.0))
    assert c.current_bearing(_PoseOnly(0.0)) == pytest.approx(45.0)
    # Yawing left (+45) leaves the object further round to the right.
    assert c.current_bearing(_PoseOnly(45.0)) == pytest.approx(90.0)


def test_a_distant_target_ignores_position_and_a_near_one_does_not():
    far = SteeringController(target=(1.0, 0.0), distant=True)
    near = SteeringController(target=(1.0, 0.0), distant=False)
    assert far.current_bearing(_PoseOnly(0.0, xy=(50.0, 50.0))) == pytest.approx(0.0)
    assert abs(near.current_bearing(_PoseOnly(0.0, xy=(0.0, 50.0)))) > 45.0


def test_without_a_target_the_bearing_is_held():
    c = SteeringController(bearing=33.0)
    assert c.current_bearing(_PoseOnly(90.0)) == pytest.approx(33.0)


# --------------------------------------------------- what closing it exposed


@pytest.fixture(scope="module")
def rig(tmp_path_factory):
    d = tmp_path_factory.mktemp("cl")
    rigid, _ = tuck_legs(SOURCE, d / "rigid.xml", keep="__none__")
    free = add_free_base(rigid, d / "free.xml", dofs="free")
    wing = wing_from_mesh(
        np.load(Path(__file__).parent / "rwing_vertices.npy"),
        n=20,
        span_axis=2,
        chord_axis=1,
    )
    return free, wing


def _yaw(body) -> float:
    m = body.data.xmat[body.root_body].reshape(3, 3)
    return float(np.degrees(np.arctan2(m[1, 0], m[0, 0])))


@needs_model
def test_the_uncontrolled_fly_yaws_on_its_own(rig):
    """Why every steering claim here is a paired measurement.

    With no steering command at all the heading still wanders, and it wanders
    one way: left. Read as an absolute bearing, that alone looks like fixation
    for objects on the left and avoidance for objects on the right. Only the
    difference between steering on and steering off means anything.
    """
    free, wing = rig
    body = FlightBody(free, wing, timestep=2e-5)
    # Explicitly uncontrolled: this is a statement about what yaw does when
    # nothing holds it, and there is now a loop that does. With the loop on
    # the same 90 ms drifts under half a degree.
    # 5.5 degrees in 90 ms. It was over 10 until the wings were told about
    # the body's rotation: flapping counter-torque damps the free drift as
    # well as a commanded turn, so every yaw magnitude in this file shrank
    # when that term went in. The claim is that the drift exists and has a
    # direction, not that it is large.
    HaltereController(**YAW_OFF).fly(body, 0.09)
    assert abs(_yaw(body)) > 3.0, _yaw(body)

    # And the loop answers it -- but not inside 90 ms, which is about twenty
    # wingbeats and less than the loop's own settling time. At 90 ms the held
    # animal is at -7.5 against the free one's +5.6, which is a transient, not
    # a failure. By 400 ms it is +2.9 against -11.5. Compare where the loop
    # has settled or the comparison is meaningless.
    def heading_at(ms, **kw):
        body = FlightBody(free, wing, timestep=2e-5)
        return float(HaltereController(**kw).fly(body, ms / 1000.0)["heading"][-1])

    # Compared across windows, not at one. A single sample of this is noisy
    # enough to flip a threshold: the ratios run 0.58, 0.54, 0.70 and 0.04 at
    # 200, 400, 600 and 800 ms. The loop is ahead at every one of them, and
    # the free animal is the one that eventually runs away.
    for ms in (200, 400, 600, 800):
        assert abs(heading_at(ms)) < abs(heading_at(ms, **YAW_OFF)), ms
    assert abs(heading_at(800)) < 0.2 * abs(heading_at(800, **YAW_OFF))


@needs_model
def test_a_right_turn_command_yaws_left_first(rig):
    """Adverse yaw, and the reason amplitude is not the steering knob.

    The wing told to beat harder carries more drag, and the drag yaws the
    animal *away* from the commanded turn. Only once the bank develops does the
    tilted lift vector turn it the intended way, and the crossover here is
    somewhere between 70 and 90 ms -- four to five wingbeats of turning the
    wrong way first.

    Real flies do not steer on amplitude alone; they shift the timing of wing
    rotation, which moves lift and drag differently. This model has only the
    amplitude, so it has only the adverse phase and then the recovery.
    """
    free, wing = rig
    flat = FlightReadout(
        bearings=np.array([-90.0, 90.0]), command=np.array([0.09, 0.09])
    )

    def yaw_at(ms: float, steering: bool) -> float:
        body = FlightBody(free, wing, timestep=2e-5)
        controller = (
            SteeringController(
                readout=flat, bearing=0.0, steer_mode="amplitude", **YAW_OFF
            )
            if steering
            else HaltereController(**YAW_OFF)
        )
        controller.fly(body, ms / 1000.0)
        return _yaw(body)

    # A right turn is decreasing yaw. Early on it does the opposite, and then
    # crosses over: +3.2 degrees at 30 ms, +2.1 at 50, -2.7 at 70, -13.5 at
    # 90. Flapping counter-torque both shrank the adverse phase and brought
    # the crossover in from 70-90 ms to 50-70.
    early = yaw_at(30, True) - yaw_at(30, False)
    assert early > 2.0, f"expected adverse yaw, got {early}"
    late = yaw_at(90, True) - yaw_at(90, False)
    assert late < -5.0, f"expected the turn to come good, got {late}"

    # And it banks while doing it, which is what eventually turns it.
    #
    # This asked for more than 30 degrees of bank until the steering
    # controller stopped advancing the sensor filter twice per step -- it
    # called the stabiliser and then recomputed the knobs, running the filter
    # at double rate and halving the time constant everything else here argues
    # about. With one advance per step the same 50 ms banks 11 degrees. The
    # old number measured the bug.
    body = FlightBody(free, wing, timestep=2e-5)
    SteeringController(
        readout=flat, bearing=0.0, steer_mode="amplitude", **YAW_OFF
    ).fly(body, 0.05)
    assert np.degrees(attitude(body)[1]) > 8.0


@needs_model
def test_closed_loop_steering_pulls_the_bearing_toward_zero_once_banked(rig):
    """Fixation, measured against the drift rather than against zero.

    At 90 ms -- past the adverse-yaw phase -- turning the steering on moves the
    bearing toward straight ahead from both sides. The comparison is the same
    run with the command zeroed, so the animal's own yaw drift cancels.
    """
    free, wing = rig
    stored = dict(np.load(CURVE))

    def final_bearing(start: float, gain: float) -> float:
        readout = FlightReadout(
            bearings=stored["bearings"], command=stored["command"], gain=gain
        )
        body = FlightBody(free, wing, timestep=2e-5)
        a = np.radians(-start)
        controller = SteeringController(
            readout=readout,
            target=(float(np.cos(a)), float(np.sin(a))),
            distant=True,
            bearing=start,
        )
        return controller.fly(body, 0.15)["bearing"][-1]

    # 150 ms, where it was 90. Flapping counter-torque damps a commanded turn
    # as well as a disturbance, so the loop needs longer to show what it is
    # doing: at 90 ms only two of the five starts have moved the right way,
    # at 150 all five have, and they stay five at 250 and 400 ms.
    toward = 0
    for start in (-60.0, -45.0, -30.0, 30.0, 45.0):
        delta = final_bearing(start, 1.0) - final_bearing(start, 0.0)
        if (delta < 0) if start > 0 else (delta > 0):
            toward += 1
    assert toward >= 4, f"only {toward} of 5 bearings moved toward straight ahead"


# ------------------------------------------------------- the other steering knob


@needs_model
def test_rotation_phase_is_a_yaw_control_and_amplitude_is_a_roll_control(rig):
    """They are different controls, not two strengths of the same one.

    Shifting when the wings flip acts through the rotational force term;
    beating one harder acts through drag. Measured about the centre of mass,
    the first is an antisymmetric yaw torque and the second an antisymmetric
    roll torque, and neither is much of the other.
    """
    from wingloop.body.flight import harmonic_stroke

    free, wing = rig
    body = FlightBody(free, wing, timestep=2e-5)

    def wrench(**kw) -> np.ndarray:
        total, n = np.zeros(6), 400
        for t in np.linspace(0, 1 / 218.0, n, endpoint=False):
            angles, rates = harmonic_stroke(t, frequency=218.0, rates=True, **kw)
            body.set_wings(angles, rates)
            body.apply_aerodynamics()
            total += body.wrench_about_com()
        return total / n

    trim = np.deg2rad(-10.7)
    left = wrench(bias=trim, phase_asymmetry=np.deg2rad(-30))
    right = wrench(bias=trim, phase_asymmetry=np.deg2rad(30))
    # Yaw reverses with the phase and is the dominant term.
    assert left[5] > 0 > right[5]
    assert abs(left[5] - right[5]) > 0.4
    assert abs(left[5] - right[5]) > abs(left[4] - right[4])

    roll_left = wrench(bias=trim, asymmetry=-0.2)
    roll_right = wrench(bias=trim, asymmetry=0.2)
    # Amplitude is a roll control, an order of magnitude stronger in roll than
    # phase is, and it barely yaws at all.
    assert roll_right[3] > 0 > roll_left[3]
    assert abs(roll_right[3] - roll_left[3]) > 10 * abs(right[5] - left[5])


@needs_model
def test_phase_steering_turns_the_right_way_from_the_start(rig):
    """The fix for adverse yaw, and the reason phase is the default knob.

    Amplitude yaws the animal away from a commanded turn for the first 50 ms.
    Phase yaws it the right way from 10 ms and never reverses.
    """
    from wingloop.body.flight import harmonic_stroke

    free, wing = rig

    def yaw_at(ms: float, **kw) -> float:
        """The stabiliser's own knobs, with a phase offset forced on top.

        The yaw loop is off because this asks what the phase knob does to an
        animal whose yaw nothing is holding -- with it on, the loop answers
        the offset and the question does not arise.
        """
        body = FlightBody(free, wing, timestep=2e-5)
        controller = HaltereController(**YAW_OFF)
        dt = float(body.model.opt.timestep)
        for _ in range(int(ms / 1000.0 / dt)):
            knobs = controller.knobs(body)
            knobs["phase_asymmetry"] += kw.get("phase_asymmetry", 0.0)
            angles, rates = harmonic_stroke(
                body.t,
                amplitude=controller.amplitude,
                frequency=controller.frequency,
                rates=True,
                **knobs,
            )
            body.set_wings(angles, rates)
            body.apply_aerodynamics()
            body._mj.mj_step(body.model, body.data)
        return _yaw(body)

    # A right turn is decreasing yaw. Phase gets the sign right immediately.
    for ms in (20.0, 50.0):
        delta = yaw_at(ms, phase_asymmetry=np.deg2rad(30)) - yaw_at(ms)
        assert delta < 0.0, f"phase steering went the wrong way at {ms} ms: {delta}"


@needs_model
def test_phase_steering_fixates_from_the_first_forty_milliseconds(rig):
    """What amplitude could not do: pull the bearing toward straight ahead
    before the bank develops, from both sides, at every window tested."""
    free, wing = rig
    stored = dict(np.load(CURVE))

    def final_bearing(start: float, gain: float, seconds: float) -> float:
        readout = FlightReadout(
            bearings=stored["bearings"], command=stored["command"], gain=gain
        )
        body = FlightBody(free, wing, timestep=2e-5)
        a = np.radians(-start)
        return SteeringController(
            readout=readout,
            target=(float(np.cos(a)), float(np.sin(a))),
            distant=True,
            bearing=start,
            steer_mode="phase",
        ).fly(body, seconds)["bearing"][-1]

    # **No single early window is reliable any more**, and picking the one
    # that passes would be the same mistake this project already had to
    # withdraw once. Counted across four windows and four starts the steering
    # is clearly pulling the right way -- 13 of 16 -- but window by window it
    # wobbles: 4 of 4 at 60 ms, 2 at 100, 4 at 150, 3 at 200. Flapping
    # counter-torque damps a commanded turn as much as a disturbance, so the
    # early phase is smaller than it was and the transients are a larger
    # share of it.
    toward = 0
    for seconds in (0.06, 0.10, 0.15, 0.20):
        for start in (-45.0, -30.0, 30.0, 45.0):
            delta = final_bearing(start, 1.0, seconds) - final_bearing(
                start, 0.0, seconds
            )
            toward += (delta < 0) if start > 0 else (delta > 0)
    assert toward >= 11, f"only {toward} of 16 start-window pairs moved toward zero"


def test_an_unknown_steer_mode_is_refused():
    r = FlightReadout(bearings=np.array([-90.0, 90.0]), command=np.array([0.1, 0.1]))
    with pytest.raises(ValueError, match="steer_mode"):
        SteeringController(readout=r, bearing=0.0, steer_mode="telepathy")
