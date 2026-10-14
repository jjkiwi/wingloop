"""The fly's brain flying a drone: the body, the drawing, the page, the mission."""

from __future__ import annotations

import json
import urllib.request

import numpy as np
from test_lab import needs_flyvis

from wingloop.lab import arena
from wingloop.lab.dronebrain import NAVIGATOR, SCALE, DroneFlight


def test_the_drone_takes_the_flys_setpoints():
    """A pitch is a speed, a heading is a heading, a height is a height -- in arena units."""
    f = DroneFlight()
    f.command(0.0, 0.0, 5.0)
    f.step(2000)
    s = f.state()
    assert abs(s["z"] - 5.0) < 0.2 and np.hypot(s["x"], s["y"]) < 0.5
    assert abs(s["altitude_m"] - (5.0 - arena.FLOOR) * SCALE / 1000) < 0.01
    # The fly's cruising pitch: forward at about the drone's cruise speed.
    f.command(12.0, 0.0, 5.0)
    f.step(3000)
    s = f.state()
    assert s["x"] > 40.0 and abs(s["y"]) < 1.0
    assert 1000.0 < s["speed"] < 1600.0  # real mm/s
    # Braking is a pitch below zero: the speed goes to nothing.
    f.command(-6.0, 0.0, 5.0)
    f.step(3000)
    assert f.state()["speed"] < 100.0
    # And it turns to a heading past half a turn, the heading unwrapped.
    for k in range(600):
        f.command(0.0, min(270.0, k * 0.5), 5.0)
        f.step(5)
    f.step(3000)
    assert abs(f.state()["heading"] - 270.0) < 5.0 and f.upright()


def test_the_camera_sits_where_the_drone_is_in_the_arena():
    f = DroneFlight()
    f.command(12.0, 30.0, 8.0)
    f.step(4000)
    s = f.state()
    position, R = f.pose()
    assert np.allclose(position, [s["x"], s["y"], s["z"]])
    forward = R @ np.array([1.0, 0.0, 0.0])
    assert abs(np.degrees(np.arctan2(forward[1], forward[0])) - s["heading"]) < 1.0


def test_the_drone_is_drawn_in_the_arena():
    from wingloop.lab.dronebrain import DroneView

    items = arena.mission(0)
    view = DroneView(items=items)
    assert view.model.ngeom > len(items) + 12  # scenery, objects, and the airframe
    f = DroneFlight()
    f.command(12.0, 45.0, 5.0)
    f.step(3000)
    view.show(f)
    assert np.allclose(view.data.xpos[view.root_body], f.pose()[0])


def test_the_brain_page_shows_the_drones_body():
    from wingloop.lab.livebrain import LiveBrain

    f = DroneFlight()
    f.step(100)
    live = LiveBrain(None)
    try:
        page = urllib.request.urlopen(live.url, timeout=5).read().decode()
        assert "Żyroskop" in page and "renderDrone" in page
        live.update(f.state(), f.knobs, f.rates(), 7.0)
        snap = json.load(urllib.request.urlopen(live.url + "state.json", timeout=5))
        assert snap["ready"] and len(snap["drone"]["motors"]) == 4
        assert snap["z"] == round((5.0 - arena.FLOOR) * SCALE)  # mm above the floor
    finally:
        live.close()


@needs_flyvis
def test_the_drone_flies_to_what_the_fly_learned_to_like(tmp_path):
    """The fly's tethered lesson, unchanged, steering a quadrotor to a ball."""
    from wingloop.lab.autopilot import Autopilot, learned
    from wingloop.lab.watch import FlysEye

    items = (arena.Item("sphere", 65.0, 10.0), arena.Item("bar", -40.0, 55.0))
    eye = FlysEye(items=items)
    mb, valence = learned(eye, {"sphere"}, folder=tmp_path, progress=None)
    pilot = Autopilot(items, mb, eye, valence, {"sphere"}, vehicle=DroneFlight(), **NAVIGATOR)
    while pilot.flight.t < 12.0 and not pilot.reached():
        pilot.step()
        assert pilot.flight.upright()
    assert [it.kind for it in pilot.reached()] == ["sphere"]
    assert pilot.who == "Dron"
