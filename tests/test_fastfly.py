"""The fast fly against the full physics it was fitted to."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from test_lab import needs_flyvis, needs_model_lab

from wingloop.lab import arena
from wingloop.lab.fastfly import FastFly

#: Full-physics flights, recorded every 5 ms after 0.2 s of settling: columns
#: t, x, y, z, vx, vy, heading, pitch, roll, and the pitch, heading and
#: height setpoints. Pitch steps of 4/8/12/15 degrees (then a brake), heading
#: steps hovering and cruising, a 240 deg/s ramp, and a height step.
TRACES = Path(__file__).with_name("fastfly_traces.npz")


def _replay(trace):
    f = FastFly()
    out = []
    for row in trace:
        f.command(*row[9:12])
        f.step(5)
        s = f.state()
        out.append([s["x"], s["y"], s["z"], s["vx"], s["vy"], s["heading"]])
    return np.asarray(out)


@pytest.mark.parametrize(
    "case, speed, heading",
    [
        ("pitch4", 15, 4),
        ("pitch8", 20, 4),
        ("pitch12", 20, 5),
        ("pitch15", 25, 5),
        ("ramp", 25, 8),
        ("cruiseturn", 20, 6),
        ("height", 25, 4),
        # The big reversal (135 degrees back) is where the model is worst: the
        # full fly overshoots a few degrees and the model does not.
        ("turn", 35, 20),
    ],
)
def test_the_fast_fly_answers_its_setpoints_as_the_full_physics_does(case, speed, heading):
    full = np.load(TRACES)[case]
    fast = _replay(full)
    v_full, v_fast = np.hypot(full[:, 4], full[:, 5]), np.hypot(fast[:, 3], fast[:, 4])
    assert np.sqrt(np.mean((v_full - v_fast) ** 2)) < speed  # mm/s
    assert np.sqrt(np.mean((full[:, 6] - fast[:, 5]) ** 2)) < heading  # degrees
    assert np.sqrt(np.mean((full[:, 3] - fast[:, 2]) ** 2)) < 1.2  # mm of height
    # And it ends up where the fly did: within 20% of the way flown, or a few
    # body lengths. Turning at speed the full fly has a slow tail the model
    # leaves out, which is most of the miss on the cruising turn.
    start = full[0, 1:3]
    flown = np.hypot(*(full[-1, 1:3] - start))
    assert np.hypot(*(full[-1, 1:3] - start - fast[-1, :2])) < max(20.0, 0.2 * flown)


def test_the_fast_fly_runs_far_faster_than_the_animal():
    import time

    f = FastFly()
    f.command(12.0, 90.0, 5.0)
    t0 = time.perf_counter()
    f.step(2000)  # two seconds of flight
    assert time.perf_counter() - t0 < 1.0
    assert f.state()["neurons"]["DNa08/DNp31 drive"] > 0


@needs_model_lab
def test_the_fast_fly_is_drawn_where_it_is_with_its_wings_beating():
    from wingloop.lab.watch import FlyView

    view, f = FlyView(), FastFly()
    f.command(12.0, 40.0, 7.0)
    f.step(500)
    view.show(f)
    assert np.allclose(view.data.xpos[view.root_body], f.pose()[0], atol=1e-6)
    wings = [view.data.qpos[i] for i in view.wing_qpos.values()]
    f.step(1)
    view.show(f)
    assert not np.allclose(wings, [view.data.qpos[i] for i in view.wing_qpos.values()])


@needs_flyvis
def test_in_the_fast_mode_the_fly_still_flies_to_what_it_learned_to_like(tmp_path):
    from wingloop.lab.autopilot import EYE_MS, Autopilot, learned
    from wingloop.lab.watch import FlysEye

    items = (arena.Item("sphere", 70.0, 0.0), arena.Item("bar", -40.0, 55.0))
    eye = FlysEye(items=items, every_ms=EYE_MS[True])
    mb, valence = learned(eye, {"sphere"}, folder=tmp_path, progress=None)
    assert valence["sphere"] > 0 > max(valence["slab"], valence["bar"])
    pilot = Autopilot(items, mb, eye, valence, {"sphere"}, vehicle=FastFly())
    while pilot.flight.t < 5.0 and not pilot.reached():
        pilot.step()
    assert [it.kind for it in pilot.reached()] == ["sphere"]
