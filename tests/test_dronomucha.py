"""The drone's jobs, kept apart from the lab: a job changes nothing in the lab."""

from __future__ import annotations

import numpy as np
import pytest

from wingloop import dronomucha
from wingloop.lab import arena, autopilot, watch


def test_a_job_is_flown_in_its_own_world_and_the_lab_is_put_back():
    """While a job flies, the arena is its world; after, the lab's -- even after an error."""
    lab = (dict(arena.KINDS), dict(autopilot.POLISH), dict(watch.SCENERY), watch.THERMAL_SPAN)
    z = dronomucha.zadanie("ratunek_gory")
    with dronomucha.swiat(z):
        assert set(arena.KINDS) == set(z.kinds) and not watch.SCENERY == lab[2]
    with pytest.raises(RuntimeError), dronomucha.swiat(z):
        raise RuntimeError("a job that fails")
    assert lab == (
        dict(arena.KINDS),
        dict(autopilot.POLISH),
        dict(watch.SCENERY),
        watch.THERMAL_SPAN,
    )


def test_a_new_job_is_one_description():
    """Registering a job is all a new use of the drone needs; a target must be in its world."""
    kinds = {"cel": dict(pl="cel", shape="sphere", size=5.0, heat=(5.0, 9.0), rgb=(1, 0, 0))}
    Zadanie = dronomucha.Zadanie
    z = dronomucha.dodaj(Zadanie(name="proba", pl="próba", target="cel", kinds=kinds))
    try:
        assert dronomucha.zadanie("proba") is z
        assert {it.kind for it in dronomucha.obszar(z, 3)} == {"cel"}
        with pytest.raises(ValueError):
            dronomucha.dodaj(Zadanie(name="zle", pl="złe", target="nic", kinds=kinds))
    finally:
        dronomucha.ZADANIA.pop("proba")


def test_rescue_sees_a_person_by_heat():
    """Through the thermal camera a person lying in the snow is dark: warm."""
    from wingloop.lab.scene import rotation

    z = dronomucha.zadanie("ratunek_gory")
    eye = dronomucha.oko(z)
    assert eye.thermal
    with dronomucha.swiat(z):
        eye.set_items((arena.Item("czlowiek", 25.0, 0.0),))
        lo, hi = z.kinds["czlowiek"]["heat"]
        assert lo <= eye.world.things[-1].heat <= hi
        eye.look_from(np.array([0.0, 0.0, 5.0]), rotation(0, 0, 0))
    assert eye.eye.dark(0.25)[0] > 0.05


def test_the_rescue_lessons_ship():
    """The rescue jobs' lessons are on disk, so a search starts without flying them."""
    for name in ("ratunek_gory", "ratunek_morze"):
        z = dronomucha.zadanie(name)
        (F, K), (CF, CK) = dronomucha.lekcje(None, z, progress=None)
        assert len(K) == z.lessons[0] and len(CK) == z.lessons[1]
        assert set(K) == set(z.kinds) and F.shape[1] == CF.shape[1]


def test_the_rescue_scene_is_drawn_with_models_that_move_with_their_items():
    """The 3-D view of a job: models of parts, mountains, snow; a moving item's model moves."""
    pytest.importorskip("mujoco")
    from wingloop.lab.dronebrain import DroneView

    z = dronomucha.zadanie("ratunek_gory")
    with dronomucha.swiat(z):
        items = dronomucha.obszar(z, 3, moving=10.0)
        view = DroneView(items=items)
        body = view.model.body("flylab_item0").id
        assert view.model.body_geomnum[body] == len(z.kinds[items[0].kind]["parts"])
        assert view.model.hfield("dronomucha_gory").id >= 0
        arena.move_geoms(view.model, items, 2.0)
        assert np.allclose(view.model.body_pos[body][:2], items[0].at(2.0))


def test_the_operators_thermal_screen_shows_a_person_warm_part_by_part():
    """Full screen, a person's drawn model is warm part by part: jacket warmer than boots."""
    from wingloop.dronomucha import termo
    from wingloop.dronomucha.wyglad import CIEPLO, czlowiek_lezacy
    from wingloop.lab.scene import World, thermal_along

    parts = czlowiek_lezacy()
    solids = termo.kule(parts, (0.0, 0.0, 0.0), 20.0)
    boots = [t for t in solids if t.heat == 20.0 * CIEPLO[(0.2, 0.12, 0.08, 1)]]
    assert boots and max(t.heat for t in solids) == 20.0 * CIEPLO[(0.86, 0.66, 0.52, 1)]
    # Grouped, the picture is the one drawn without grouping.
    o = np.array([-40.0, 0.0, 10.0])
    rays = termo.promienie((64, 36))
    G = termo.gimbal(np.eye(3), tilt=10.0)
    d = rays @ G.T
    world = World([], ground_height=-3.0)
    plain = thermal_along(World(solids, ground_height=-3.0), o, d, clutter=0.3)
    grouped = thermal_along(world, o, d, clutter=0.3, groups=[((0, 0, 0), 30.0, solids)])
    assert np.array_equal(plain, grouped) and plain.max() > 15.0


def test_the_senses_are_coded_as_lines_each_tuned_to_a_value():
    """A reading is a bump of activity round its value; nothing measured is silence."""
    from wingloop.dronomucha.zmysly import POLA, STROJENIE, kod

    r = np.array([[10.0, 50.0, np.nan, 20.0, 0.0], [1.0, 200.0, 5.0, 3.0, 1.0]])
    lines = kod(r)
    per = [STROJENIE[k][2] if k in STROJENIE else 1 for k in POLA]
    assert lines.shape == (2, sum(per)) and kod(r, gain=3).shape == (2, 3 * sum(per))
    height = slice(per[0] + per[1], per[0] + per[1] + per[2])
    assert not lines[0, height].any() and lines[1, height].max() > 0.5
    temp = slice(0, per[0])
    assert lines[0, temp].argmax() > lines[1, temp].argmax()  # warmer, further along


def test_the_gimbal_reads_a_person_lying_below_the_horizon():
    """The eye looks level; the person is below it; the gimbal finds them, warm and near."""
    from wingloop.dronomucha.zmysly import odczyt, polozenie
    from wingloop.lab.scene import rotation

    z = dronomucha.zadanie("ratunek_gory")
    eye = dronomucha.oko(z)
    with dronomucha.swiat(z):
        eye.set_items((arena.Item("czlowiek", 45.0, 0.0),))
        heat = eye.world.things[-1].heat
        eye.look_from(np.array([0.0, 0.0, 5.0]), rotation(0, 0, 0))
        r = odczyt(eye, np.random.default_rng(0))
        xy = polozenie(eye, np.random.default_rng(0))
    assert abs(r[0] - heat) < 2.0 and 25.0 < r[1] < 50.0 and 3.0 < r[2] < 15.0
    # Its near end: the person is 34 mm long, lying along the way the drone looks.
    assert np.hypot(xy[0] - 28.0, xy[1]) < 5.0


def test_the_search_map_sends_the_drone_where_it_has_not_looked():
    """Looked around from the start, the map is lower there and its next place is out."""
    from wingloop.dronomucha.mapa import Mapa

    m = Mapa()
    assert np.isclose(m.poa0.sum(), 1.0)

    class Nav:
        _home = (0.0, 0.0)
        log = [dict(phase="scan", x=0.0, y=0.0, heading=h) for h in range(0, 360, 5)]

    spot = m(Nav(), (0.0, 0.0))
    assert m.zostalo() < 0.7 and 30.0 <= np.hypot(*spot) < 110.0
    # Looked only east, it is sent west.
    m = Mapa()
    Nav.log = [dict(phase="scan", x=40.0, y=0.0, heading=h) for h in range(0, 360, 5)]
    assert m(Nav(), (40.0, 0.0))[0] < 0


def test_the_object_memory_follows_what_it_turned_down():
    """An object turned down while moving is avoided where it has gone, not where it was."""
    from wingloop.dronomucha.pamiec import Pamiec, Slad

    class Nav:
        log = [dict(t=10.0)]
        visited, rejected = [], []

    mem = Pamiec(None, None)
    track = Slad(0.0, (60.0, 0.0))
    for t in (1.0, 2.0, 3.0):
        track.t.append(t)
        track.xy.append((60.0, 10.0 * t))
    mem.slady, mem._last = [track], track
    Nav.rejected = [(60.0, 30.0)]
    assert np.allclose(track.predkosc(), (0.0, 10.0), atol=0.1)
    # At t = 10 it is about (60, 90): that way is turned down, where it was is not.
    assert mem(Nav, (0.0, 0.0), np.degrees(np.arctan2(90, 60)))
    assert not mem(Nav, (0.0, 0.0), -30.0)


def test_the_drone_asks_the_operator_only_when_it_is_not_sure():
    """A close look near the line is the operator's to decide; a sure one, the brain's."""
    from wingloop.dronomucha.operator import Operator

    class Brain:
        v = 0.02

        def valence(self, f):
            return self.v

    class Nav:
        phase = "approach"
        log = [dict(t=5.0, x=0.0, y=0.0, heading=0.0)]

    items = (arena.Item("czlowiek", 50.0, 0.0), arena.Item("kamien", -50.0, 0.0))
    brain = Brain()
    op = Operator(brain, None, items, "czlowiek", prog=0.1, trafnosc=1.0, zwloka=4.0)
    op.nav = Nav
    assert op.valence(None) == 0.5 and op.czekanie == 4.0
    brain.v = -0.3
    assert op.valence(None) == -0.3 and len(op.pytania) == 1 and len(op.zgloszenia) == 2
    Nav.phase = "scan"
    brain.v = 0.01
    assert op.valence(None) == 0.01 and len(op.pytania) == 1
