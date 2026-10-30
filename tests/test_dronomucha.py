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
