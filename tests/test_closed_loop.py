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
    HaltereController().fly(body, 0.09)
    assert abs(_yaw(body)) > 10.0


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
            SteeringController(readout=flat, bearing=0.0, steer_mode="amplitude")
            if steering
            else HaltereController()
        )
        controller.fly(body, ms / 1000.0)
        return _yaw(body)

    # A right turn is decreasing yaw. Early on it does the opposite.
    early = yaw_at(50, True) - yaw_at(50, False)
    assert early > 10.0, f"expected adverse yaw, got {early}"

    # And it banks hard while doing it, which is what eventually turns it.
    body = FlightBody(free, wing, timestep=2e-5)
    SteeringController(readout=flat, bearing=0.0, steer_mode="amplitude").fly(body, 0.05)
    assert np.degrees(attitude(body)[1]) > 30.0


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
        return controller.fly(body, 0.09)["bearing"][-1]

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
        body = FlightBody(free, wing, timestep=2e-5)
        controller = HaltereController()
        dt = float(body.model.opt.timestep)
        from wingloop.body.control import angular_rate

        for _ in range(int(ms / 1000.0 / dt)):
            pitch, roll = attitude(body)
            pitch, roll, rate = controller._filtered(
                pitch, roll, angular_rate(body), dt
            )
            bias = (
                controller.trim_bias
                + controller.pitch_gain * pitch
                + controller.pitch_rate_gain * rate[1]
            )
            asym = -controller.roll_gain * roll - controller.roll_rate_gain * rate[0]
            angles, rates = harmonic_stroke(
                body.t,
                amplitude=controller.amplitude,
                frequency=controller.frequency,
                bias=float(np.clip(bias, -controller.max_bias, controller.max_bias)),
                asymmetry=float(np.clip(asym, -0.45, 0.45)),
                rates=True,
                **kw,
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

    for seconds in (0.04, 0.06):
        for start in (-45.0, -30.0, 30.0, 45.0):
            delta = final_bearing(start, 1.0, seconds) - final_bearing(
                start, 0.0, seconds
            )
            toward = (delta < 0) if start > 0 else (delta > 0)
            assert toward, f"{start} deg at {seconds * 1000:.0f} ms moved away: {delta}"


def test_an_unknown_steer_mode_is_refused():
    r = FlightReadout(bearings=np.array([-90.0, 90.0]), command=np.array([0.1, 0.1]))
    with pytest.raises(ValueError, match="steer_mode"):
        SteeringController(readout=r, bearing=0.0, steer_mode="telepathy")
