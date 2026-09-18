"""The hinge, and whether the force model still works once it is in a body."""

from pathlib import Path

import numpy as np
import pytest

from wingloop.aero.blade_element import hover_check, lift_coefficient
from wingloop.aero.wing import wing_from_mesh

mujoco = pytest.importorskip("mujoco", reason="the flight body needs MuJoCo")

from wingloop.body.flight import FlightBody, harmonic_stroke  # noqa: E402
from wingloop.body.hinge import (  # noqa: E402
    HINGE_DOFS,
    SPAN_LOCAL,
    WINGS,
    add_wing_hinges,
    neuromechfly_model,
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
        body.prescribe(angles, rates)
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
        body.prescribe(angles, rates)
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
        body.prescribe(angles, rates)
        body.apply_aerodynamics()
        per_step.append((angles["joint_RWing_rotation"], body.total_applied_force()[2]))

    upstroke = [v for a, v in per_step if a < -0.1]
    assert upstroke, "the test stroke never pitched the wing negative"
    assert np.mean(upstroke) > 0, "the upstroke is producing downward force"
