"""flylab: the scene, the drone, the eye, recognition, learning and the report."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from wingloop.lab.drone import FlyPilot, Quadrotor, wrap
from wingloop.lab.learning import MushroomBody, learning_index, load, save, train
from wingloop.lab.recognition import evaluate
from wingloop.lab.scene import Camera, Thing, World, render

FEATURES = Path(__file__).parent / "visual_features.npz"


def _flyvis_ready() -> bool:
    try:
        import flyvis

        return (flyvis.results_dir / "flow/0000/000").exists()
    except Exception:
        return False


needs_flyvis = pytest.mark.skipif(
    not _flyvis_ready(), reason="needs flyvis and its pretrained models"
)


# ------------------------------------------------------------------- the world


def test_an_object_on_the_left_is_seen_on_the_left():
    world = World([Thing("sphere", [5.0, 1.5, 1.5], 0.6)])
    image = render(world, [0.0, 0.0, 1.5], (0.0, 0.0, 0.0), Camera(size=64))
    rows, cols = np.nonzero(image < 0.1)
    assert len(cols) > 0 and cols.mean() < 32  # column 0 is the animal's left
    # Sky above the horizon, ground below it.
    assert image[2].mean() > 0.8 and image[-2].mean() < 0.6


def test_every_kind_of_object_is_drawn():
    for kind in ("sphere", "bar", "box"):
        z = 0.0 if kind == "bar" else 1.5
        world = World([Thing(kind, [4.0, 0.0, z], 0.5)])
        image = render(world, [0.0, 0.0, 1.5], (0.0, 0.0, 0.0), Camera(size=48))
        assert (image < 0.1).mean() > 0.005, kind


# ------------------------------------------------------------------- the drone


def _fly(drone, pilot, seconds, speed=0.0, turn=0.0, dt=0.002):
    for _ in range(int(seconds / dt)):
        drone.step(pilot.control(drone, speed, turn, dt), dt)


def test_the_throttle_law_holds_the_drone_at_its_height():
    drone, pilot = Quadrotor(), FlyPilot(altitude=1.5)
    drone.state.position[:] = [0.0, 0.0, 0.5]
    _fly(drone, pilot, 4.0)
    assert drone.state.position[2] == pytest.approx(1.5, abs=0.02)
    assert np.degrees(np.abs(drone.state.attitude[:2])).max() < 0.5


def test_it_flies_forward_by_tilting_and_turns_by_its_heading_setpoint():
    drone, pilot = Quadrotor(), FlyPilot(altitude=1.5)
    drone.state.position[2] = 1.5
    _fly(drone, pilot, 3.0, speed=1.0)
    assert drone.state.velocity[0] > 0.7
    assert np.degrees(drone.state.attitude[1]) > 0.5  # nose down to go forward
    _fly(drone, pilot, 2.0, speed=0.0, turn=0.5)
    assert np.degrees(wrap(drone.state.attitude[2])) == pytest.approx(57.3, abs=8.0)
    assert drone.state.position[2] == pytest.approx(1.5, abs=0.05)


def test_the_attitude_gains_are_the_flys_formula():
    pilot = FlyPilot()
    kp, kd = pilot.gains()
    inertia = np.asarray(pilot.frame.inertia)
    assert kp[0] == pytest.approx(inertia[0] * pilot.attitude_bandwidth**2)
    assert kd[2] == pytest.approx(inertia[2] * 2 * pilot.yaw_bandwidth)


# ----------------------------------------------------- recognition and learning


@pytest.fixture(scope="module")
def recorded():
    d = np.load(FEATURES)
    return d["X"], d["R"], d["y"]


def test_the_optic_lobe_recognises_objects_better_than_its_photoreceptors(recorded):
    X, R, y = recorded
    result = evaluate(X, R, y)
    assert result["network"][0] > result["photoreceptors"][0] + 0.10, result
    assert result["network"][0] > 0.70, result
    assert result["shuffled"][0] < 0.45, result


def test_dopamine_teaches_bar_against_sphere_and_unpaired_dopamine_does_not(recorded):
    X, _, y = recorded
    paired, unpaired = [], []
    for seed in (1, 2, 3):
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(y))
        tr, te = idx[:240], idx[240:]
        for labels, store in ((y, paired), (rng.permutation(y), unpaired)):
            mb = MushroomBody(X.shape[1], seed=seed)
            mb.calibrate(X[tr])
            train(mb, X[tr], labels[tr], {1: 1.0, 0: -1.0}, seed=seed)
            choices = [
                (mb.valence(x) > 0) == (c == 1)
                for x, c in zip(X[te], y[te], strict=True)
                if c != 2
            ]
            store.append(np.mean(choices))
    assert min(paired) > 0.85, paired
    assert max(unpaired) < 0.70, unpaired


def test_the_rule_itself_learns_perfectly_given_ideal_object_channels():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 300)
    X = np.eye(3)[y].repeat(12, axis=1) + 0.01 * rng.standard_normal((300, 36))
    mb = MushroomBody(36)
    mb.calibrate(X[:200])
    train(mb, X[:200], y[:200], {0: 1.0, 2: -1.0})
    v = learning_index(mb, X[200:], y[200:])
    assert v[0] > 0.8 and v[2] < -0.8 and abs(v[1]) < 0.2, v


def test_a_memory_survives_being_saved(tmp_path, recorded):
    X, _, y = recorded
    mb = MushroomBody(X.shape[1])
    mb.calibrate(X)
    train(mb, X[:60], y[:60], {1: 1.0, 0: -1.0})
    save(mb, tmp_path / "m.npz")
    again = load(tmp_path / "m.npz")
    assert [again.valence(x) for x in X[:10]] == pytest.approx([mb.valence(x) for x in X[:10]])


# ----------------------------------------------------------------- the eye


@pytest.fixture(scope="module")
def eye():
    if not _flyvis_ready():
        pytest.skip("needs flyvis and its pretrained models")
    from wingloop.lab.vision import FlyEye

    return FlyEye()


@needs_flyvis
def test_a_grey_world_reads_grey_on_every_column(eye):
    eye.reset()
    eye.step(np.full((96, 96), 0.5, np.float32))
    assert eye.photoreceptors.min() == pytest.approx(0.5, abs=1e-3)
    assert eye.photoreceptors.max() == pytest.approx(0.5, abs=1e-3)


@needs_flyvis
def test_the_eye_finds_the_object_and_sees_it_loom(eye):
    for lateral, sign in ((1.5, 1), (-1.5, -1)):
        eye.reset()
        world = World([Thing("sphere", [5.0, lateral, 1.5], 0.6)])
        sizes = []
        for k in range(12):
            world.things[0].position[0] = 5.0 - 0.25 * k
            activity = eye.step(render(world, [0, 0, 1.5], (0, 0, 0)), frames=2)
            sizes.append(eye.dark()[0])
        bearing, _ = eye.bearing(activity)
        assert np.sign(bearing) == sign and abs(bearing) > 0.3, bearing
        assert sizes[-1] > 2.5 * sizes[1], sizes


# ----------------------------------------------------------------- the report


def test_the_report_builds_from_results(tmp_path):
    from wingloop.lab.report import build

    runs = {
        "recognition": {
            "network": (0.78, 0.04),
            "photoreceptors": (0.58, 0.05),
            "shuffled": (0.36, 0.05),
            "chance": 1 / 3,
        },
        "learning": {
            "reward": {"bar": 1.0, "sphere": -1.0},
            "valence": {"bar": 0.8, "sphere": -0.7},
        },
    }
    page = build(json.loads(json.dumps(runs)), tmp_path / "r.html").read_text()
    assert "Object recognition" in page and "mushroom body" in page and "78%" in page


# ------------------------------------------------------------ learning in the drone

FIXATIONS = Path(__file__).parent / "drone_fixations.npz"


def test_the_drone_learns_from_its_own_flights_and_generalises():
    """Twenty training flights, twenty new ones: what the mushroom body takes away.

    Each fixation is the drone hovering and looking at one object, recorded as
    the foveal optic-lobe features. 80% correct choices on the new flights;
    unpaired dopamine stays near chance.
    """
    from wingloop.lab.drone import teach

    d = np.load(FIXATIONS)
    F, K, batch = d["F"], d["K"], d["batch"]
    train_, test_ = batch == 0, batch == 1
    paired, unpaired = [], []
    for seed in range(5):
        rng = np.random.default_rng(seed)
        for labels, store in ((K[train_], paired), (rng.permutation(K[train_]), unpaired)):
            mb = MushroomBody(F.shape[1], seed=seed)
            teach(mb, F[train_], labels, {"bar": 1.0, "sphere": -1.0}, seed=seed)
            v = np.array([mb.valence(f) for f in F[test_]])
            store.append(np.mean((v > 0) == (K[test_] == "bar")))
    assert np.mean(paired) > 0.75, paired
    assert np.mean(unpaired) < 0.55, unpaired


def _mission(reward):
    from wingloop.lab.drone import Mission, place, teach
    from wingloop.lab.vision import FlyEye

    d = np.load(FIXATIONS)
    mb = MushroomBody(d["F"].shape[1])
    teach(mb, d["F"], d["K"], reward)
    world = World([place("bar", 120, 5), place("sphere", -100, 5)])
    mission = Mission(world, FlyEye(), mb)
    mission.run(40.0)
    return mission


@needs_flyvis
def test_the_drone_flies_to_what_it_was_rewarded_for():
    """Look around, judge, go: the whole loop, vision to motors."""
    m = _mission({"bar": 1.0, "sphere": -1.0})
    assert m.phase == "arrived", m.phase
    d = m.distances()
    assert d["bar"] < 2.5 and d["sphere"] > 5.0, d
    valence = {round(float(np.degrees(mem.heading))) % 360: mem.valence for mem in m.memories}
    assert len(m.memories) == 2, valence


@needs_flyvis
def test_and_reversing_the_reward_reverses_the_choice():
    """The control: same eye, same world, opposite dopamine."""
    m = _mission({"bar": -1.0, "sphere": 1.0})
    assert m.phase == "arrived", m.phase
    d = m.distances()
    assert d["sphere"] < 2.5 and d["bar"] > 5.0, d


def test_the_cache_writer_never_deletes_a_file_it_holds_open(tmp_path, monkeypatch):
    """The Windows failure: datamate deleted an HDF5 file while its handle was open.

    Linux allows that and Windows refuses, so flyvis could not build its
    connectome cache on Windows at all. The replacement removes any old file
    before opening, so no unlink can happen with a handle open -- checked here
    by making every unlink fail while any file is open.
    """
    pytest.importorskip("h5py")
    pytest.importorskip("datamate")
    import h5py
    from datamate.io import H5Reader

    from wingloop.lab import vision

    opened = []
    real_file, real_unlink = h5py.File, Path.unlink

    class Tracked(real_file):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            opened.append(self)

    def unlink(self, *a, **k):
        assert not any(f.id.valid for f in opened), "unlinked a file that is still open"
        return real_unlink(self, *a, **k)

    monkeypatch.setattr(h5py, "File", Tracked)
    monkeypatch.setattr(Path, "unlink", unlink)

    path = tmp_path / "deep" / "unique_cell_types.h5"
    vision._write_h5(path, np.array([b"T4a", b"T5b"]))
    vision._write_h5(path, np.arange(5))  # over an existing file
    monkeypatch.undo()
    assert H5Reader(path)[()].tolist() == [0, 1, 2, 3, 4]
