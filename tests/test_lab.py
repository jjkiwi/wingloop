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


def _body_model_ready() -> bool:
    try:
        from wingloop.body.hinge import neuromechfly_model

        return neuromechfly_model().exists()
    except Exception:
        return False


needs_model_lab = pytest.mark.skipif(not _body_model_ready(), reason="needs FlyGym's MJCF")
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


def test_a_camouflaged_object_wears_the_floors_checker():
    """At 1 it is painted like the ground, tile for tile; nothing on it is dark any more."""
    from wingloop.lab.scene import GROUND

    def look(camo):
        world = World([Thing("box", [4.0, 0.0, 1.5], 0.6, camo=camo)])
        return render(world, [0.0, 0.0, 1.5], (0.0, 0.0, 0.0), Camera(size=64))

    plain, hidden = look(0.0), look(1.0)
    on_it = plain < 0.1
    assert on_it.mean() > 0.01 and (hidden < 0.1).sum() == 0
    assert np.abs(hidden[on_it][:, None] - np.array(GROUND)).min(axis=1).max() < 1e-3
    half = look(0.5)[on_it]
    assert np.all((half > 0.1) & (half < GROUND[1]))


def test_a_camouflaged_arena_is_the_same_arena():
    from wingloop.lab import arena

    plain, hidden = arena.mission(3, per_kind=2), arena.mission(3, per_kind=2, camo=0.7)
    assert [(i.kind, i.x, i.y) for i in plain] == [(i.kind, i.x, i.y) for i in hidden]
    assert all(i.camo == 0.7 for i in hidden)
    assert all(t.camo == 0.7 for t in arena.eye_things(hidden))


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


# ------------------------------------------------------------------ the 3-D view


def test_the_pilot_keys_do_what_the_help_says():
    from wingloop.lab.watch import Pilot

    p = Pilot()
    for key in (265, 265, 263, 87):  # up, up, left, W
        p.key(key)
    assert p.pitch_deg == 6.0 and p.heading_deg == 15.0 and p.height_mm == 8.0
    p.key(72)  # H: hover
    assert p.pitch_deg == 0.0
    p.key(32)
    assert p.paused
    for _ in range(20):
        p.key(265)
    assert p.pitch_deg == 20.0, "the nose-down setpoint is capped"


@needs_model_lab
def test_the_drawn_fly_is_where_the_physics_fly_is():
    """The view is posed from the physics, wings included, and never stepped."""
    from wingloop.lab.watch import Flight, FlyView

    flight, view = Flight(), FlyView()
    flight.step(500)
    view.pose(flight.body)
    i = flight.body.root_dof
    assert np.allclose(
        view.data.qpos[view.root_qpos : view.root_qpos + 7], flight.body.data.qpos[i : i + 7]
    )
    assert len(view.wing_qpos) == 6
    for name, angle in flight.body.wing_angles.items():
        assert view.data.qpos[view.wing_qpos[name]] == pytest.approx(angle)
    # Both wings are drawn somewhere other than their rest pose mid-stroke.
    assert any(abs(flight.body.wing_angles[n]) > 0.1 for n in flight.body.wing_angles)


@needs_model_lab
def test_a_nose_down_setpoint_flies_the_fly_forward():
    """Forward flight is the hover loop with a pitch setpoint: 8 degrees, about 0.2 m/s."""
    from wingloop.lab.watch import Flight

    hover, forward = Flight(), Flight()
    forward.command(8.0, 0.0, 5.0)
    hover.command(0.0, 0.0, 5.0)
    for f in (hover, forward):
        f.step(int(0.3 / 2e-5))
    assert forward.state()["x"] > hover.state()["x"] + 15.0, (forward.state(), hover.state())
    assert abs(forward.state()["roll"]) < 10 and forward.upright()


def test_the_brain_panel_scales_its_signals_and_draws(tmp_path):
    """The bars stay inside the axes and the panel renders without a display."""
    pytest.importorskip("matplotlib")
    from wingloop.lab.brainpanel import BARS, BrainPanel, signals

    state = {
        "t": 0.1,
        "speed": 250.0,
        "z": 5.0,
        "heading": 10.0,
        "neurons": {
            "DNa08/DNp31 drive": 0.7,
            "power MNs (DLMn, DVMn)": 0.27,
            "steering MNs": 0.013,
            "DNbe001 L-R": 0.0,
        },
    }
    knobs = {"bias": -0.1, "asymmetry": 0.5, "phase_asymmetry": -2.0}
    values = signals(state, knobs, np.array([10.0, -10.0, 0.5]))
    assert set(values) == {name for name, _ in BARS}
    assert all(-1.0 <= v <= 1.0 for v in values.values()), values
    assert values["wing: amplitude L-R"] == 1.0 and values["haltere: roll rate"] == 1.0

    panel = BrainPanel(None, live=False)
    for k in range(3):
        panel.update(dict(state, t=0.1 + 0.01 * k), knobs, np.zeros(3))
    image = panel.image()
    assert image.ndim == 3 and image.shape[2] == 3 and image.shape[0] > 300


@needs_flyvis
@needs_model_lab
def test_the_optic_lobe_runs_on_what_the_flying_fly_sees():
    """The fly's view is rendered from the physics pose and stepped through flyvis."""
    from wingloop.lab.watch import Flight, FlysEye

    flight, eye = Flight(), FlysEye()
    flight.command(10.0, 0.0, 5.0)
    images = []
    for _ in range(4):
        flight.step(250)
        eye.look(flight.body)
        images.append(eye.image.copy())
    assert eye.activity is not None and eye.activity.shape == (eye.eye.n_neurons,)
    assert eye.image.shape == (64, 64)
    # Ground below, sky above, and the view changes as the fly moves.
    assert eye.image[:5].mean() > eye.image[-5:].mean()
    assert np.abs(images[-1] - images[0]).mean() > 1e-3


def test_the_brain_page_is_served_with_live_state():
    """The page, its layout and a snapshot come back from the local server."""
    import json
    import urllib.request

    from wingloop.lab.livebrain import LiveBrain

    live = LiveBrain(None)
    try:
        assert live.url.startswith("http://127.0.0.1:")
        page = urllib.request.urlopen(live.url, timeout=5).read().decode()
        assert "Mózg muchy w locie" in page and "/state.json" in page
        assert json.load(urllib.request.urlopen(live.url + "layout.json", timeout=5)) == {
            "eye": False
        }
        assert (
            json.load(urllib.request.urlopen(live.url + "state.json", timeout=5))["ready"]
            is False
        )
        state = {
            "t": 0.25,
            "speed": 210.0,
            "z": 5.0,
            "heading": 12.0,
            "pitch": 8.0,
            "roll": -1.0,
            "neurons": {
                "DNa08/DNp31 drive": 0.7,
                "power MNs (DLMn, DVMn)": 0.27,
                "steering MNs": 0.013,
                "DNbe001 L-R": 0.0,
            },
        }
        live.update(
            state, {"bias": -0.2, "asymmetry": 0.01, "phase_asymmetry": 0.1}, np.zeros(3), 40.0
        )
        snap = json.load(urllib.request.urlopen(live.url + "state.json", timeout=5))
        assert snap["ready"] and snap["speed"] == 210.0 and snap["slowdown"] == 40.0
        assert snap["eye"] is None and len(snap["history"]["t"]) == 1
        assert -1 <= snap["bars"]["wing: rotation phase L-R"] <= 1
    finally:
        live.close()


def test_depth_camera_measures_through_camouflage():
    """The depth camera sees an object's size, whatever it is painted like."""
    from wingloop.lab import arena
    from wingloop.lab.autopilot import TARGET_SIZES
    from wingloop.lab.navigate import Navigator
    from wingloop.lab.scene import depth
    from wingloop.lab.watch import FlysEye

    eye = FlysEye(items=())
    heading = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # facing +y
    assert eye.measure() is None  # nothing looked at yet
    nav = Navigator(None)
    for kind in ("sphere", "slab", "bar"):
        sizes = []
        for camo in (0.0, 1.0):
            eye.set_items((arena.Item(kind, 0.0, 50.0, camo=camo),))
            eye.pose = (np.array([0.0, 0.0, 5.0]), heading)
            sizes.append(eye.measure(np.random.default_rng(0)))
        assert sizes[0] == sizes[1]
        m = sizes[0]
        assert 30 < m["distance"] < 50 and not m["top"]
        # Each fits its own size, and no other's.
        for other, size in TARGET_SIZES.items():
            nav.sizes = size
            assert nav._fits(m) == (other == kind), (kind, other, m)
    # Straight down from 17 mm above the floor, the floor is 17 mm away.
    down = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    eye.set_items(())
    from wingloop.lab.scene import Camera

    t = depth(
        eye.world,
        (0.0, 0.0, eye.world.ground_height + 17.0),
        Camera(size=3, fov=1.0),
        rotation_matrix=down,
    )
    assert np.allclose(t, 17.0, atol=0.1)


def test_thermal_camera_and_rangefinder_measure_from_far():
    """Paint is not in a thermal picture, and with the rangefinder it gives sizes from far."""
    from wingloop.lab import arena
    from wingloop.lab.scene import Camera, thermal
    from wingloop.lab.watch import LRF_RANGE, TOF_RANGE, FlysEye

    eye = FlysEye(items=())
    heading = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # facing +y
    assert eye.measure_far() is None  # nothing looked at yet
    far = 150.0
    assert TOF_RANGE < far < LRF_RANGE
    seen = {}
    for kind in ("sphere", "slab", "bar"):
        pictures, sizes = [], []
        for camo in (0.0, 1.0):
            eye.set_items((arena.Item(kind, 0.0, far, camo=camo),))
            eye.pose = (np.array([0.0, 0.0, 5.0]), heading)
            pictures.append(
                thermal(eye.world, eye.pose[0], Camera(32, 60.0), rotation_matrix=heading)
            )
            sizes.append(eye.measure_far(np.random.default_rng(0)))
        assert np.array_equal(pictures[0], pictures[1]) and sizes[0] == sizes[1]
        assert abs(sizes[0]["distance"] - far) < 15 and eye.measure() is None  # beyond the ToF
        seen[kind] = sizes[0]
    # The ball and the slab stand as tall, the pole much taller; the slab is the widest.
    assert abs(seen["sphere"]["height"] - seen["slab"]["height"]) < 4
    assert seen["bar"]["height"] > seen["sphere"]["height"] + 10
    assert seen["slab"]["width"] > seen["sphere"]["width"] + 8
    # Nothing in the beam's way: no distance, and no measure.
    eye.set_items(())
    eye.pose = (np.array([0.0, 0.0, 5.0]), heading)
    assert not np.isfinite(eye.laser_range()) and eye.measure_far() is None


def test_far_size_decides_what_is_worth_flying_to():
    """Measured from far, the target's size makes a sighting worth a look; another size not."""
    from wingloop.lab.navigate import Navigator, Sighting

    nav = Navigator(None)
    liked, disliked = Sighting(0.0, 0.1, 0.01, (0, 0)), Sighting(0.0, -0.1, 0.01, (0, 0))
    assert nav._worth(liked) and not nav._worth(disliked)  # not measured: the brain decides
    nav.far_bar = -0.2
    liked.fits, disliked.fits = False, True
    assert nav._worth(disliked) and not nav._worth(liked)
    nav.far_veto = False
    assert nav._worth(liked)  # another size: the brain decides again


def test_moving_objects_go_round_and_the_eye_follows():
    """Moving objects keep their circle, the eye sees them where they are, the arena stays."""
    from wingloop.lab import arena
    from wingloop.lab.navigate import nearest
    from wingloop.lab.watch import FlysEye

    still, moving = arena.mission(3, per_kind=2), arena.mission(3, per_kind=2, moving=10.0)
    assert [(it.x, it.y) for it in still] == [(it.x, it.y) for it in moving]
    assert {np.sign(it.speed) for it in moving} == {-1.0, 1.0}
    it = moving[0]
    x, y = it.at(2.0)
    r = np.hypot(it.x, it.y)
    assert np.hypot(x, y) == pytest.approx(r)
    assert np.hypot(x - it.x, y - it.y) == pytest.approx(20.0, rel=0.02)  # 2 s at 10 mm/s
    assert nearest(moving, x, y, 2.0) == (it, pytest.approx(0.0, abs=1e-9))
    eye = FlysEye(items=moving)
    eye.move_to(2.0)
    thing = eye.world.things[len(eye.background)]
    assert np.allclose(thing.position[:2], (x, y))


def test_near_infrared_camera_sees_through_paint_by_its_match():
    """Through the NIR camera, camouflage counts only as much as the paint matches."""
    from wingloop.lab import arena
    from wingloop.lab.watch import FlysEye

    heading = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    pictures = {}
    for camo, nir in ((0.0, None), (1.0, None), (1.0, 0.0), (1.0, 1.0)):
        eye = FlysEye(items=(arena.Item("bar", 0.0, 40.0, camo=camo),))
        eye.nir = nir
        eye.look_from(np.array([0.0, 0.0, 5.0]), heading)
        pictures[camo, nir] = eye.image.copy()
    assert np.array_equal(pictures[1.0, 0.0], pictures[0.0, None])  # paint not matched at all
    assert np.array_equal(pictures[1.0, 1.0], pictures[1.0, None])  # matched as for the eye
    assert not np.array_equal(pictures[1.0, None], pictures[0.0, None])
