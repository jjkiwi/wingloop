"""The fly flying to its targets on its own: arena, teaching, navigator, page."""

from __future__ import annotations

import json
import urllib.request

import numpy as np
import pytest
from test_lab import needs_flyvis, needs_model_lab

from wingloop.lab import arena
from wingloop.lab.autopilot import POLISH, kind_of, map_image
from wingloop.lab.learning import MushroomBody
from wingloop.lab.navigate import Navigator, nearest, teach

# ------------------------------------------------------------------- the arena


@pytest.mark.parametrize("seed", range(6))
def test_a_mission_spreads_its_objects_around_the_start(seed):
    items = arena.mission(seed, per_kind=2)
    assert sorted(it.kind for it in items) == sorted(2 * list(arena.KINDS))
    headings = np.sort([it.bearing_from(0, 0) % 360 for it in items])
    gaps = np.diff(np.r_[headings, headings[0] + 360])
    assert gaps.min() >= 360 / len(items) - 20 - 1e-6
    for it in items:
        assert 50 <= it.distance_from(0, 0) <= 80


def test_one_list_of_items_draws_both_scenes():
    import xml.etree.ElementTree as ET

    items = arena.mission(0)
    world = ET.Element("worldbody")
    arena.add_to_mjcf(world, items)
    geoms = world.findall("geom")
    assert len(geoms) == len(items)
    assert {g.get("type") for g in geoms} == {"sphere", "box", "cylinder"}
    assert all(g.get("contype") == "0" for g in geoms)  # scenery, nothing to hit
    things = arena.eye_things(items)
    assert [t.label for t in things] == [it.name for it in items]
    # Every object is darker than the floor's darkest tile, so the eye sees it as an object.
    assert all(t.brightness < 0.25 for t in things)


def test_the_names_people_type_are_the_kinds():
    assert kind_of("kula") == "sphere" and kind_of("Słupek") == "bar"
    assert kind_of("plytka") == "slab" and kind_of("bar") == "bar"
    with pytest.raises(SystemExit):
        kind_of("kot")


# ----------------------------------------------------------------- teaching


def test_balanced_dopamine_makes_the_target_the_only_liked_kind():
    """One target against two punished kinds still ends with the target liked.

    Unbalanced, punishment depresses the approach output twice as often as
    reward depresses avoidance and every kind ends up disliked.
    """
    rng = np.random.default_rng(0)
    kinds = ("sphere", "slab", "bar")
    centres = {k: rng.normal(0, 1, 60) for k in kinds}
    F = np.array([centres[k] + rng.normal(0, 0.5, 60) for k in kinds for _ in range(30)])
    K = np.array([k for k in kinds for _ in range(30)])
    for target in kinds:
        mb = MushroomBody(60, seed=1)
        teach(mb, F, K, {k: 1.0 if k == target else -1.0 for k in kinds})
        for k in kinds:
            views = centres[k] + rng.normal(0, 0.5, (20, 60))
            v = np.mean([mb.valence(x) for x in views])
            assert (v > 0) == (k == target), (target, k, v)


def test_the_tethered_views_are_looked_at_once_and_any_target_taught_from_them(
    tmp_path, monkeypatch
):
    from wingloop.lab import autopilot

    calls = []
    rng = np.random.default_rng(2)
    centres = {k: rng.normal(0, 1, 40) for k in arena.KINDS}

    def look(eye, kinds, samples, seed=0, progress=None, **kw):
        calls.append(samples)
        K = np.array([kinds[i % len(kinds)] for i in range(samples)])
        return np.array([centres[k] + rng.normal(0, 0.4, 40) for k in K]), K

    monkeypatch.setattr(autopilot, "tethered_training", look)
    monkeypatch.setattr(autopilot, "flown_close_looks", look)
    monkeypatch.setattr(autopilot, "SHIPPED", tmp_path / "not-shipped.npz")
    for target in ("sphere", "bar"):
        mb, valence = autopilot.learned(None, {target}, folder=tmp_path, progress=None)
        assert max(valence, key=valence.get) == target
    # Looked once -- scanning on the tether, and close up in flight -- taught twice.
    assert calls == [90, autopilot.CLOSE_LOOKS]
    # A change to the objects makes the saved views stale.
    monkeypatch.setitem(arena.KINDS["bar"], "size", 9.0)
    autopilot.learned(None, {"bar"}, folder=tmp_path, progress=None)
    assert calls == [90, autopilot.CLOSE_LOOKS] * 2


@needs_flyvis
def test_the_close_look_lesson_is_flown_by_the_body_itself():
    """Turned away, it flies to the object, stops, looks: that view is the lesson."""
    from wingloop.lab.autopilot import EYE_MS, flown_close_looks
    from wingloop.lab.watch import FlysEye

    eye = FlysEye(items=(arena.Item("bar", 60.0, 0.0),), every_ms=EYE_MS[True])
    world = eye.world
    F, K = flown_close_looks(eye, tuple(arena.KINDS), 3)
    assert F.shape == (3, 325) and K.tolist() == list(arena.KINDS)
    assert np.all(np.isfinite(F)) and eye.world is world  # its own arena given back


def test_the_shipped_views_are_the_ones_the_current_arena_would_produce():
    """Change an object's size and this fails: run ``flylab cel --od-nowa`` and ship that."""
    from wingloop.lab import autopilot

    for every_ms in (5.0, 20.0):  # the full simulation's eye, and the fast mode's
        for close in (False, True):  # scanning, and close up
            for drone in (False, True):  # the fly's way of flying, and the drone's
                path = autopilot.shipped(every_ms, close, drone)
                n = autopilot.CLOSE_LOOKS if close else 90
                with np.load(path) as d:
                    sig = autopilot._signature(n, 0, every_ms, close, drone)
                    assert str(d["signature"]) == sig, path.name
                    assert d["F"].shape == (n, 325)
                    assert sorted(set(d["K"].tolist())) == sorted(arena.KINDS)
    # And the drone's camouflage lessons, in the fast mode, with the landscape's columns.
    from wingloop.lab.navigate import LANDMARK

    for close in (False, True):
        path = autopilot.shipped(20.0, close, drone=True, camo=True)
        n = autopilot.close_looks(camo=True) if close else 90
        with np.load(path) as d:
            assert str(d["signature"]) == autopilot._signature(n, 0, 20.0, close, True, True)
            assert sorted(set(d["K"].tolist())) == sorted([*arena.KINDS, LANDMARK])
    # And its looks from missions, which a stale signature would fly again.

    class Eye:
        every_ms = 20.0

    def stale(*a, **kw):
        raise AssertionError("the shipped mission looks are stale: fly them again")

    with pytest.MonkeyPatch.context() as m:
        m.setattr(autopilot, "mission_looks", stale)
        F, K = autopilot.mission_views(Eye(), "/nonexistent", progress=None)
    assert F.shape[1] == 325 and len(F) == len(K) > 500
    assert sorted(set(K.tolist())) == sorted([*arena.KINDS, LANDMARK])


# --------------------------------------------------------------- navigating


class _Eye:
    """A stand-in for the fly's eye: dark objects, their bearing and size."""

    class camera:  # noqa: N801 - mirrors FlysEye.camera
        fov = 120.0

    def __init__(self, world, value):
        self.world, self.value = world, value
        self.activity = 0
        self.eye = self
        self.pose = (0.0, 0.0, 0.0)

    def _visible(self, within=1.0, centre=0.0):
        x, y, h = self.pose
        out = []
        for ox, oy, kind in self.world:
            d = max(np.hypot(ox - x, oy - y), 1.0)
            off = (np.degrees(np.arctan2(oy - y, ox - x)) - h + 180) % 360 - 180
            # The real eye's numbers for the ball: 0.017 of the eye at 48 mm,
            # and nothing it can make out beyond about 80.
            size = min(1.0, 39.0 / d**2)
            if abs(off) < 60 and abs(off - 60 * centre) < 60 * within and size > 0.006:
                out.append((size, off, kind))
        return out

    def dark(self, threshold, within=1.0, centre=0.0):
        seen = self._visible(within, centre)
        if not seen:
            return 0.0, 0.0
        total = sum(s for s, _, _ in seen)
        return total, sum(s * o for s, o, _ in seen) / total / 60.0

    def fovea_shape(self, activity):
        seen = self._visible()
        if not seen:
            return np.zeros(1)
        size, _, kind = min(seen, key=lambda v: abs(v[1]))
        closer = getattr(self, "closer_value", {})
        if kind in closer and size > 0.035:  # and from nearer still
            return np.array([closer[kind]])
        close = getattr(self, "close_value", {})
        if kind in close and size > 0.015:  # what it really is, seen from near
            return np.array([close[kind]])
        return np.array([self.value[kind]])


class _Judge:
    def __init__(self):
        self.taught = []

    def valence(self, f):
        return float(f[0])

    def learn(self, f, dopamine):
        self.taught.append((float(f[0]), dopamine))


def _fly(world, value, seconds=40.0, reward=None):
    """Kinematic stand-in for the fly: heading as asked, speed from the pitch."""
    eye, nav, dt = _Eye(world, value), Navigator(_Judge(), reward=reward), 0.005
    x = y = h = 0.0
    nav.start(h)
    t, path = 0.0, []
    while t < seconds and nav.phase != "done":
        eye.pose = (x, y, h)
        pitch, heading, _ = nav.decide(eye, dict(t=t, x=x, y=y, heading=h), dt)
        h += np.clip(heading - h, -3.0, 3.0)  # a few degrees per look, like the fly
        v = max(pitch, 0.0) * 20.0
        x, y = x + v * dt * np.cos(np.deg2rad(h)), y + v * dt * np.sin(np.deg2rad(h))
        path.append((x, y))
        t += dt
    return nav, np.array(path)


def test_the_navigator_flies_to_each_liked_object_and_not_the_others():
    world = [(65.0, 0.0, "sphere"), (-35.0, 55.0, "bar"), (-10.0, -70.0, "sphere")]
    nav, path = _fly(world, {"sphere": 0.3, "bar": -0.3})
    assert nav.phase == "done"

    def closest(ox, oy):
        return np.hypot(path[:, 0] - ox, path[:, 1] - oy).min()

    assert closest(65, 0) < 30 and closest(-10, -70) < 30
    assert closest(-35, 55) > 40
    assert len(nav.visited) == 2


def test_turning_to_its_target_it_does_not_latch_onto_what_it_turns_past():
    """The pole at 45 degrees crosses the middle of the eye on the way to the ball at 125."""
    world = [(40.0, 40.0, "bar"), (-35.0, 50.0, "sphere")]
    nav, path = _fly(world, {"sphere": 0.3, "bar": -0.3})
    assert np.hypot(path[:, 0] + 35, path[:, 1] - 50).min() < 30
    assert np.hypot(path[:, 0] - 40, path[:, 1] - 40).min() > 40


def test_a_closer_look_saves_a_landing_on_what_only_looked_right_from_afar():
    """Liked from 65 mm, disliked from 45: it stops, looks, and does not land."""
    world = [(65.0, 0.0, "bar")]

    def fly(inspect):
        eye, nav, dt = _Eye(world, {"bar": 0.3}), Navigator(_Judge(), inspect=inspect), 0.005
        eye.close_value = {"bar": -0.3}
        nav.curious = False
        x = y = h = 0.0
        nav.start(h)
        t = 0.0
        while t < 20.0 and nav.phase != "done":
            eye.pose = (x, y, h)
            pitch, heading, _ = nav.decide(eye, dict(t=t, x=x, y=y, heading=h), dt)
            h += np.clip(heading - h, -3.0, 3.0)
            v = max(pitch, 0.0) * 20.0
            x, y = x + v * dt * np.cos(np.deg2rad(h)), y + v * dt * np.sin(np.deg2rad(h))
            t += dt
        return nav

    careful, hasty = fly(True), fly(False)
    assert not careful.visited and [(round(v, 6), ok) for v, ok in careful.inspections] == [
        (-0.3, False)
    ]
    assert len(hasty.visited) == 1


def test_a_landing_teaches_the_far_view_it_chose_on_and_the_close_look():
    """And landing on the same place again teaches nothing new."""
    world = [(65.0, 0.0, "bar")]
    eye, dt = _Eye(world, {"bar": 0.3}), 0.005
    eye.close_value = {"bar": 0.1}
    nav = Navigator(_Judge(), inspect=True, land_to_learn=True, reward=lambda x, y: -1.0)
    nav.curious = False
    x = y = h = t = 0.0
    nav.start(h)
    while t < 12.0 and nav.phase != "done":
        eye.pose = (x, y, h)
        pitch, heading, _ = nav.decide(eye, dict(t=t, x=x, y=y, heading=h), dt)
        h += np.clip(heading - h, -3.0, 3.0)
        v = max(pitch, 0.0) * 20.0
        x, y = x + v * dt * np.cos(np.deg2rad(h)), y + v * dt * np.sin(np.deg2rad(h))
        t += dt
    taught = [(round(f, 6), d) for f, d in nav.mushroom_body.taught]
    assert (0.1, -1.0) in taught and (0.3, -1.0) in taught  # close look, far view
    assert len(nav.lessons) == 1


def test_two_liked_objects_close_together_are_both_landed_on():
    """Its places fall short of the objects; the one it came from is behind it, not here."""
    world = [(60.0, 0.0, "slab"), (60.0, 40.0, "slab")]
    nav, path = _fly(world, {"slab": 0.3})
    assert len(nav.visited) == 2 and nav.phase == "done"
    assert np.hypot(path[:, 0] - 60, path[:, 1] - 40).min() < 30


def test_with_nothing_liked_in_sight_it_goes_home_and_looks_from_there():
    """From the first target the second is out of sight; from home it is not."""
    world = [(70.0, 0.0, "sphere"), (-60.0, 5.0, "sphere")]
    nav, path = _fly(world, {"sphere": 0.3})
    phases = [entry["phase"] for entry in nav.log]
    assert "home" in phases
    assert np.hypot(path[:, 0] + 60, path[:, 1] - 5).min() < 30
    assert nav.phase == "done" and len(nav.visited) == 2


def test_where_it_lands_it_is_rewarded_or_punished_for_what_it_chose():
    """The dopamine goes onto the view the choice was made on."""
    world = [(65.0, 0.0, "sphere"), (-35.0, 55.0, "bar")]

    def sugar(x, y):  # only the ball is sweet
        return 1.0 if np.hypot(x - 65, y) < 30 else -1.0

    nav, _ = _fly(world, {"sphere": 0.3, "bar": 0.2}, reward=sugar)
    assert sorted(d for _, d, _ in nav.lessons) == [-1.0, 1.0]
    # The ball's view (0.3) was rewarded, the pole's (0.2) punished.
    taught = sorted((round(v, 6), d) for v, d in nav.mushroom_body.taught)
    assert taught == [(0.2, -1.0), (0.3, 1.0)]


def test_a_fly_that_likes_nothing_tries_the_most_promising_thing_once():
    """Curiosity: with nothing liked and nowhere landed, it checks the best, then stops."""
    world = [(65.0, 0.0, "bar"), (-40.0, 50.0, "slab")]
    nav, path = _fly(world, {"bar": -0.2, "slab": -0.1})
    assert nav.phase == "done" and len(nav.visited) == 1
    assert np.hypot(path[:, 0] + 40, path[:, 1] - 50).min() < 30  # the slab, valued higher
    assert np.hypot(path[:, 0] - 65, path[:, 1]).min() > 40
    # Without curiosity it stays put.
    eye, nav = _Eye(world, {"bar": -0.2, "slab": -0.1}), Navigator(_Judge(), curious=False)
    nav.start(0.0)
    for k in range(4000):
        eye.pose = (0.0, 0.0, nav.heading)
        nav.decide(eye, dict(t=k * 0.005, x=0.0, y=0.0, heading=nav.heading), 0.005)
        if nav.phase == "done":
            break
    assert nav.phase == "done" and not nav.visited


# -------------------------------------------------------------- the display


def _slow(world, value, close, seconds=30.0, speed=4.0, **kw):
    """A body with more momentum than a fly's: a fifth of its speed per degree."""
    eye, dt = _Eye(world, value), 0.005
    eye.close_value = close
    nav = Navigator(_Judge(), inspect=True, curious=False, **kw)
    x = y = h = t = 0.0
    nav.start(h)
    while t < seconds and nav.phase != "done":
        eye.pose = (x, y, h)
        pitch, heading, _ = nav.decide(eye, dict(t=t, x=x, y=y, heading=h), dt)
        h += np.clip(heading - h, -3.0, 3.0)
        v = max(pitch, 0.0) * speed
        x, y = x + v * dt * np.cos(np.deg2rad(h)), y + v * dt * np.sin(np.deg2rad(h))
        t += dt
    return nav


def test_a_close_look_it_likes_gives_the_rest_of_the_way_in_its_own_time():
    """Slow to stop and to look, a drone gave up a few centimetres short of what it liked."""
    nav = _slow([(65.0, 0.0, "sphere")], {"sphere": 0.3}, {"sphere": 0.2}, give_up=1.2)
    assert [ok for _, ok in nav.inspections] == [True]
    assert len(nav.visited) == 1 and np.hypot(*np.subtract(nav.visited[0], (65.0, 0.0))) < 30


def test_the_drone_goes_back_to_where_it_saw_something_it_never_looked_at():
    """Out of sight from home, a ball it saw from elsewhere: it goes there, and looks."""
    from wingloop.lab.navigate import Sighting

    world = [(60.0, 75.0, "sphere")]  # 96 mm from home, 75 from where it was seen
    there = Sighting(90.0, -0.01, 0.007, (60.0, 0.0), np.array([-0.01]))
    value, close = {"sphere": -0.01}, {"sphere": 0.2}
    kw = dict(speed=20.0, look_before_leaving=True, seen=[there])
    assert not _slow(world, value, close, **kw).visited  # the fly does not
    nav = _slow(world, value, close, go_back=True, **kw)
    assert len(nav.visited) == 1 and nav.phase == "done"


def test_with_nothing_left_it_explores_the_widest_stretch_it_has_seen_nothing_in():
    """A ball 90 mm out, too small for the eye from home, in a half-circle of nothing."""
    world = [(60.0, 0.0, "bar"), (30.0, 52.0, "bar"), (-30.0, 52.0, "bar"), (-60.0, 0.0, "bar")]
    world.append((0.0, -90.0, "sphere"))
    value, close = {"sphere": -0.01, "bar": -0.3}, {"sphere": 0.2, "bar": -0.3}
    kw = dict(speed=20.0, look_before_leaving=True, seconds=60.0)
    assert not _slow(world, value, close, **kw).visited
    nav = _slow(world, value, close, explore=40.0, **kw)
    assert len(nav.visited) == 1 and np.hypot(*np.subtract(nav.visited[0], (0.0, -90.0))) < 30


def test_turned_down_by_a_hair_from_afar_it_goes_closer_and_looks_again():
    """The drone's far close looks were its only wrong ones: a near-zero no means look again."""
    world = [(70.0, 0.0, "slab")]

    def fly(unsure):
        eye, dt = _Eye(world, {"slab": 0.3}), 0.005
        eye.close_value, eye.closer_value = {"slab": -0.01}, {"slab": 0.2}
        nav = Navigator(_Judge(), inspect=True, curious=False, unsure=unsure)
        x = y = h = t = 0.0
        nav.start(h)
        while t < 20.0 and nav.phase != "done":
            eye.pose = (x, y, h)
            pitch, heading, _ = nav.decide(eye, dict(t=t, x=x, y=y, heading=h), dt)
            h += np.clip(heading - h, -3.0, 3.0)
            v = max(pitch, 0.0) * 20.0
            x, y = x + v * dt * np.cos(np.deg2rad(h)), y + v * dt * np.sin(np.deg2rad(h))
            t += dt
        return nav

    assert not fly(0.0).visited  # the first close look's no is final
    nav = fly(0.06)
    assert [round(v, 6) for v, _ in nav.inspections] == [0.2] and len(nav.visited) == 1


def test_a_camouflaged_object_is_looked_at_from_three_sides_and_judged_on_all_three():
    """Disliked from the side it came from, liked from the two others: it lands."""
    world = [(65.0, 0.0, "slab")]

    class Sides(_Eye):
        def fovea_shape(self, activity):
            x, y, _ = self.pose
            if np.hypot(x - 65.0, y) > 50.0:
                return np.array([0.3])  # liked from afar
            side = np.degrees(np.arctan2(y, x - 65.0))  # where it looks from
            return np.array([-0.2 if abs(abs(side) - 180) < 20 else 0.2])

    def fly(looks):
        eye, dt = Sides(world, {"slab": 0.3}), 0.005
        nav = Navigator(_Judge(), inspect=True, curious=False, looks=looks)
        x = y = h = t = 0.0
        nav.start(h)
        phases = set()
        while t < 40.0 and nav.phase != "done":
            eye.pose = (x, y, h)
            pitch, heading, _ = nav.decide(eye, dict(t=t, x=x, y=y, heading=h), dt)
            phases.add(nav.phase)
            h += np.clip(heading - h, -3.0, 3.0)
            v, side, r = max(pitch, 0.0) * 20.0, nav.sideways * 20.0, np.deg2rad(h)
            x += (v * np.cos(r) - side * np.sin(r)) * dt  # and sideways, to the left
            y += (v * np.sin(r) + side * np.cos(r)) * dt
            t += dt
        return nav, phases

    once, _ = fly(1)
    assert not once.visited and [round(v, 6) for v, _ in once.inspections] == [-0.2]
    nav, phases = fly(3)
    assert "around" in phases
    assert [round(v, 6) for v, _ in nav.inspections] == [round(0.2 / 3, 6)]
    assert len(nav.visited) == 1


def test_peering_it_takes_its_close_look_flying_sideways_and_facing_the_object():
    world = [(65.0, 0.0, "bar")]
    eye, dt = _Eye(world, {"bar": 0.3}), 0.005
    nav = Navigator(_Judge(), inspect=True, curious=False, peer=10.0, peer_time=0.4)
    x = y = h = t = 0.0
    nav.start(h)
    looking = []
    while t < 20.0 and not nav.inspections:
        eye.pose = (x, y, h)
        pitch, heading, _ = nav.decide(eye, dict(t=t, x=x, y=y, heading=h), dt)
        if nav._stopped:
            off = (np.degrees(np.arctan2(-y, 65.0 - x)) - h + 180) % 360 - 180
            looking.append((nav.sideways, abs(off)))
        h += np.clip(heading - h, -3.0, 3.0)
        v, side, r = max(pitch, 0.0) * 20.0, nav.sideways * 20.0, np.deg2rad(h)
        x += (v * np.cos(r) - side * np.sin(r)) * dt
        y += (v * np.sin(r) + side * np.cos(r)) * dt
        t += dt
    assert nav.inspections and nav.sideways == 0.0  # and still again once it has looked
    assert len(looking) * dt >= 0.4 and all(s == 10.0 for s, _ in looking[1:])
    assert max(off for _, off in looking) < 10.0 and abs(y) > 5.0  # moved, facing it


def test_judged_against_the_rest_it_goes_back_to_what_stood_out_though_disliked():
    """Every close look below zero, the target least so: it lands there, and only there."""
    world = [(65.0, 0.0, "sphere"), (-40.0, 55.0, "bar"), (-10.0, -70.0, "bar")]

    def fly(relative):
        eye, dt = _Eye(world, {"sphere": -0.05, "bar": -0.1}), 0.005
        eye.close_value = {"sphere": -0.06, "bar": -0.3}
        nav = Navigator(
            _Judge(), inspect=True, curious=False, look_before_leaving=True, relative=relative
        )
        x = y = h = t = 0.0
        nav.start(h)
        phases = set()
        while t < 90.0 and nav.phase != "done":
            eye.pose = (x, y, h)
            pitch, heading, _ = nav.decide(eye, dict(t=t, x=x, y=y, heading=h), dt)
            phases.add(nav.phase)
            h += np.clip(heading - h, -3.0, 3.0)
            v = max(pitch, 0.0) * 20.0
            x, y = x + v * dt * np.cos(np.deg2rad(h)), y + v * dt * np.sin(np.deg2rad(h))
            t += dt
        return nav, phases

    plain, _ = fly(0.0)
    assert not plain.visited and len(plain.inspections) == 3
    nav, phases = fly(0.1)
    assert "back" in phases  # turned down first, with nothing yet to compare it to
    assert len(nav.visited) == 1 and np.hypot(*np.subtract(nav.visited[0], (65.0, 0.0))) < 30


def test_it_looks_closely_only_once_it_faces_the_object_and_has_stopped_turning():
    """Starting its approach near enough to look, the drone first turns to face it."""
    from wingloop.lab.dronebrain import NAVIGATOR

    facing = {k: NAVIGATOR[k] for k in ("centred", "still", "face_time")}
    world = [(30.0, 30.0, "bar")]  # 42 mm away, 45 degrees to the left: near enough already
    eye, dt = _Eye(world, {"bar": 0.3}), 0.005
    nav = Navigator(_Judge(), inspect=True, curious=False, **facing)
    x = y = h = t = 0.0
    nav.start(h)
    offs = []
    while t < 10.0 and not nav.inspections:
        eye.pose = (x, y, h)
        pitch, heading, _ = nav.decide(eye, dict(t=t, x=x, y=y, heading=h), dt)
        h += np.clip(heading - h, -0.4, 0.4)  # 80 deg/s at most: a drone's turn
        v = max(pitch, 0.0) * 4.0
        x, y = x + v * dt * np.cos(np.deg2rad(h)), y + v * dt * np.sin(np.deg2rad(h))
        t += dt
        offs.append(abs((np.degrees(np.arctan2(30.0 - y, 30.0 - x)) - h + 180) % 360 - 180))
    assert nav.inspections and offs[-1] < 5.0
    # Without the wait it looked while still turning, the object well off the middle.
    eye, nav = _Eye(world, {"bar": 0.3}), Navigator(_Judge(), inspect=True, curious=False)
    nav.centred, nav.still = 360.0, 1e9  # no waiting
    x = y = h = t = 0.0
    nav.start(h)
    while t < 10.0 and not nav.inspections:
        eye.pose = (x, y, h)
        pitch, heading, _ = nav.decide(eye, dict(t=t, x=x, y=y, heading=h), dt)
        h += np.clip(heading - h, -0.4, 0.4)
        v = max(pitch, 0.0) * 4.0
        x, y = x + v * dt * np.cos(np.deg2rad(h)), y + v * dt * np.sin(np.deg2rad(h))
        t += dt
    off = (np.degrees(np.arctan2(30.0 - y, 30.0 - x)) - h + 180) % 360 - 180
    assert nav.inspections and abs(off) > 5.0


def test_before_leaving_it_looks_closely_at_what_it_has_not_been_to():
    """Valued just below zero from afar, a target is still worth a close look."""
    world = [(65.0, 0.0, "sphere"), (-40.0, 50.0, "sphere"), (-20.0, -60.0, "bar")]
    value = {"sphere": -0.01, "bar": -0.3}
    close = {"sphere": 0.2, "bar": -0.3}
    assert not _slow(world, value, close, speed=20.0).visited
    nav = _slow(world, value, close, speed=20.0, look_before_leaving=True)
    assert nav.phase == "done" and len(nav.visited) == 2
    # Both balls landed on, the pole turned down.
    assert sorted(ok for _, ok in nav.inspections) == [False, True, True]
    # Not liked from afar, a close look must like it clearly to overrule that.
    faint = _slow(world, value, {**close, "sphere": 0.01}, speed=20.0, look_before_leaving=True)
    assert not faint.visited and len(faint.inspections) == 3


def _status():
    return {
        "phase": "approach",
        "phase_pl": "leci do celu",
        "valence_now": 0.21,
        "learned": {"kula": 0.25, "płytka": -0.4, "słupek": -0.2},
        "items": [
            dict(kind="sphere", name="kula", x=60.0, y=5.0, reached=True, target=True),
            dict(kind="bar", name="słupek", x=-40.0, y=-50.0, reached=False, target=False),
        ],
        "sightings": [dict(heading=5.0, valence=0.21, name="kula", chosen=True)],
        "fly": dict(x=30.0, y=2.0, heading=4.0),
        "trail": [(0.0, 0.0), (15.0, 1.0), (30.0, 2.0)],
        "visited": [[62.0, 5.0]],
        "reached": ["kula"],
        "targets_left": 0,
    }


def test_the_map_draws_the_arena_the_track_and_the_fly():
    img = map_image(_status(), size=200)
    assert img.shape == (200, 200, 3)
    # The fly (black) is drawn where it is: 30 mm right of the start.
    fx, fy = int(100 + 30 / 110 * 100), int(100 - 2 / 110 * 100)
    assert img[fy - 2 : fy + 3, fx - 2 : fx + 3].min() < 40
    # And the reached object is ringed in green.
    assert (np.abs(img.astype(int) - (12, 163, 12)).sum(axis=2) < 30).any()


def test_the_brain_page_shows_the_autopilot():
    from wingloop.lab.livebrain import LiveBrain

    live = LiveBrain(None)
    try:
        page = urllib.request.urlopen(live.url, timeout=5).read().decode()
        assert "Autopilot" in page and "renderAuto" in page
        state = {
            "t": 1.0,
            "speed": 100.0,
            "z": 5.0,
            "heading": 4.0,
            "pitch": 8.0,
            "roll": 0.0,
            "neurons": {
                "DNa08/DNp31 drive": 0.7,
                "power MNs (DLMn, DVMn)": 0.27,
                "steering MNs": 0.01,
                "DNbe001 L-R": 0.0,
            },
        }
        live.update(state, {}, np.zeros(3), 50.0, autopilot=_status())
        snap = json.load(urllib.request.urlopen(live.url + "state.json", timeout=5))
        assert snap["autopilot"]["phase"] == "approach"
        assert snap["autopilot"]["items"][0]["name"] == POLISH["sphere"]
    finally:
        live.close()


# ----------------------------------------------------------- the real thing


@needs_model_lab
def test_the_fly_turns_all_the_way_round_and_keeps_flying():
    """Half a turn and more, in saccades, without losing the attitude.

    Before the haltere rates were read about the fly's own axes every turn
    past about 90 degrees ended the flight: the pitch rate went to the roll
    loop. Here it turns 225 degrees in one second.
    """
    from wingloop.lab.watch import Flight

    flight = Flight()
    flight.command(0.0, 0.0, 5.0)
    flight.step(5000)
    worst = 0.0
    while flight.t < 1.2:
        k = int((flight.t - 0.1) / 0.2) + 1
        flight.command(0.0, 45.0 * k, 5.0)
        flight.step(250)
        s = flight.state()
        worst = max(worst, abs(s["roll"]), abs(s["pitch"]))
    assert flight.upright() and worst < 20.0
    assert flight.state()["heading"] > 200.0


@needs_flyvis
@needs_model_lab
def test_the_fly_flies_to_its_target_by_itself(tmp_path):
    """Taught on the tether that balls are good, it flies to a ball."""
    from wingloop.lab.autopilot import Autopilot, learned
    from wingloop.lab.watch import FlysEye

    items = (arena.Item("sphere", 70.0, 0.0), arena.Item("bar", -40.0, 55.0))
    eye = FlysEye(items=items)
    mb, valence = learned(eye, {"sphere"}, folder=tmp_path, progress=None)
    assert valence["sphere"] > 0 > max(valence["slab"], valence["bar"])
    pilot = Autopilot(items, mb, eye, valence, {"sphere"})
    while pilot.flight.t < 4.0 and not pilot.reached():
        pilot.step()
        assert pilot.flight.upright()
    assert [it.kind for it in pilot.reached()] == ["sphere"]


def test_practice_memory_is_what_the_next_mission_flies_with(tmp_path, monkeypatch):
    """A memory left by practice is loaded, for its targets, body and eye rate only."""
    from wingloop.lab import autopilot
    from wingloop.lab.learning import save

    rng = np.random.default_rng(3)
    centres = {k: rng.normal(0, 1, 40) for k in arena.KINDS}

    def look(eye, kinds, samples, seed, progress, **kw):
        K = np.array([kinds[i % len(kinds)] for i in range(samples)])
        return np.array([centres[k] + rng.normal(0, 0.4, 40) for k in K]), K

    monkeypatch.setattr(autopilot, "tethered_training", look)
    monkeypatch.setattr(autopilot, "flown_close_looks", look)
    monkeypatch.setattr(autopilot, "SHIPPED", tmp_path / "not-shipped.npz")

    class Eye:
        every_ms = 20.0

    tether, _ = autopilot.learned(Eye(), {"sphere"}, folder=tmp_path, progress=None)
    practised = autopilot.teach_targets(
        *look(None, tuple(arena.KINDS), 90, 0, None), {"sphere"}
    )[0]
    practised.approach[:] = 0.5  # what flights taught it, recognisably
    path = autopilot.memory_path({"sphere"}, False, 20.0, tmp_path)
    save(practised, path)
    mb, _ = autopilot.learned(Eye(), {"sphere"}, folder=tmp_path, progress=None)
    assert np.allclose(mb.approach, 0.5) and not np.allclose(tether.approach, 0.5)
    # Not for the drone, not for another target, not when told to start again.
    for kw, targets in (
        ({"drone": True}, {"sphere"}),
        ({}, {"bar"}),
        ({"retrain": True}, {"sphere"}),
    ):
        other, _ = autopilot.learned(Eye(), targets, folder=tmp_path, progress=None, **kw)
        assert not np.allclose(other.approach, 0.5)
    assert path.name == "pamiec_kula_20ms.npz"


def test_the_drones_expert_brains_are_shipped_and_flown_with_by_default(tmp_path, monkeypatch):
    """With no practice of its own, the drone flies with the brain practised here."""
    from wingloop.lab import autopilot
    from wingloop.lab.learning import load, save

    for kind in arena.KINDS:
        path = autopilot.shipped_memory({kind}, drone=True, every_ms=20.0)
        with np.load(path) as d:
            assert int(d["missions"]) == 50
        load(path)
    assert not autopilot.shipped_memory({"sphere"}, drone=False, every_ms=20.0).exists()

    rng = np.random.default_rng(4)
    centres = {k: rng.normal(0, 1, 40) for k in arena.KINDS}

    def look(eye, kinds, samples, seed=0, progress=None, **kw):
        K = np.array([kinds[i % len(kinds)] for i in range(samples)])
        return np.array([centres[k] + rng.normal(0, 0.4, 40) for k in K]), K

    monkeypatch.setattr(autopilot, "tethered_training", look)
    monkeypatch.setattr(autopilot, "flown_close_looks", look)
    monkeypatch.setattr(autopilot, "SHIPPED", tmp_path / "data" / "not-shipped.npz")
    (tmp_path / "data").mkdir()

    class Eye:
        every_ms = 20.0

    expert = autopilot.teach_targets(*look(None, tuple(arena.KINDS), 90), {"bar"})[0]
    expert.approach[:] = 0.25
    save(expert, autopilot.shipped_memory({"bar"}, drone=True, every_ms=20.0))
    here = tmp_path / "here"
    here.mkdir()
    mb, _ = autopilot.learned(Eye(), {"bar"}, folder=here, progress=None, drone=True)
    assert np.allclose(mb.approach, 0.25)
    # Its own practice comes first, and --od-nowa goes back to the tether.
    own = autopilot.teach_targets(*look(None, tuple(arena.KINDS), 90), {"bar"})[0]
    own.approach[:] = 0.5
    save(own, autopilot.memory_path({"bar"}, True, 20.0, here))
    mb, _ = autopilot.learned(Eye(), {"bar"}, folder=here, progress=None, drone=True)
    assert np.allclose(mb.approach, 0.5)
    mb, _ = autopilot.learned(
        Eye(), {"bar"}, folder=here, progress=None, drone=True, retrain=True
    )
    assert not np.allclose(mb.approach, 0.5) and not np.allclose(mb.approach, 0.25)


def test_in_the_camouflage_world_a_landscape_column_is_a_kind_to_avoid():
    """Seen by what breaks the skyline, a column is taught as a non-target, and landing on one
    is punished and counted as a mistake."""
    from types import SimpleNamespace

    from wingloop.lab import autopilot
    from wingloop.lab.navigate import LANDMARK

    rng = np.random.default_rng(5)
    kinds = (*arena.KINDS, LANDMARK)
    centres = {k: rng.normal(0, 1, 40) for k in kinds}
    K = np.array([kinds[i % 4] for i in range(120)])
    F = np.array([centres[k] + rng.normal(0, 0.4, 40) for k in K])
    mb, valence = autopilot.teach_targets(F[:60], K[:60], {"sphere"}, close=(F[60:], K[60:]))
    column = np.mean([mb.valence(f) for f in F[K == LANDMARK]])
    ball = np.mean([mb.valence(f) for f in F[K == "sphere"]])
    assert column < 0 < ball

    items = (arena.Item("sphere", 60.0, 0.0), arena.Item("bar", -60.0, 0.0))
    pilot = SimpleNamespace(
        items=items,
        targets={"sphere"},
        landmarks=((0.0, 70.0),),
        punish=1.0,
        flight=SimpleNamespace(t=0.0),
    )
    pilot.nav = SimpleNamespace(visited=[(55.0, 0.0), (0.0, 65.0), (0.0, -40.0)])
    pilot.landed = [nearest(items, x, y) for x, y in pilot.nav.visited]
    reward = autopilot.Autopilot.reward
    assert reward(pilot, 55.0, 0.0) > 0 > reward(pilot, 0.0, 65.0)
    assert reward(pilot, 0.0, -40.0) == 0.0  # nowhere in particular
    assert autopilot.Autopilot.on_landmarks(pilot) == 1


@needs_flyvis
def test_the_exam_grades_what_it_knows_now(tmp_path):
    from wingloop.lab.autopilot import GRADES, exam

    result = exam({"slab"}, arenas=1, folder=tmp_path, progress=None)
    found, mistakes, wanted = result["rows"][0]
    assert wanted == 2 and 0 <= found <= 2 and mistakes >= 0
    assert result["grade"] in [g for g, _, _ in GRADES]


def test_taught_with_a_prediction_error_it_learns_only_what_it_still_gets_wrong():
    """A view already valued as aimed teaches nothing; the rest is pulled to +-aim."""
    rng = np.random.default_rng(0)
    centres = {k: rng.normal(0, 1, 30) for k in ("bar", "slab", "sphere")}
    K = np.array(["bar", "slab", "sphere"] * 40)
    F = np.array([centres[k] + rng.normal(0, 0.5, 30) for k in K])
    reward = {"bar": 1.0, "slab": -1.0, "sphere": -1.0}
    mb = MushroomBody(30, on_off=True, sparseness=0.03)
    teach(mb, F, K, reward, aim=0.3, epochs=3)
    v = np.array([mb.valence(f) for f in F])
    assert np.median(v[K == "bar"]) > 0.2 and np.median(v[K != "bar"]) < -0.2
    # Taught once more, nothing it already values as aimed is touched.
    before = mb.approach.copy(), mb.avoid.copy()
    there = (K == "bar") & (v >= 0.3) | (K != "bar") & (v <= -0.3)
    teach(mb, F[there], K[there], reward, calibrate=False, aim=0.3, epochs=1)
    assert np.array_equal(before[0], mb.approach) and np.array_equal(before[1], mb.avoid)
