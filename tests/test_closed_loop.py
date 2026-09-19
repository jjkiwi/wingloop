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
    """Adverse yaw, and the reason amplitude alone is the wrong steering knob.

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
            SteeringController(readout=flat, bearing=0.0)
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
    SteeringController(readout=flat, bearing=0.0).fly(body, 0.05)
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
