"""The search at real scale: a Tatra valley, a thermal camera looking down, two platforms."""

from __future__ import annotations

import numpy as np
import pytest

from wingloop.dronomucha.platformy import KAMERY, PLATFORMY
from wingloop.dronomucha.teren import kamera as K
from wingloop.dronomucha.teren import lot as LOT
from wingloop.dronomucha.teren import obiekty as O
from wingloop.dronomucha.teren import teren as T
from wingloop.dronomucha.teren import wykrywanie as W


@pytest.fixture(scope="module")
def dolina():
    t = T.dolina(krok=4.0)
    pg = T.Pogoda()
    return t, pg, t.temperatura(pg)


def test_the_valley_is_a_tatra_valley(dolina):
    """A floor about 1550-1750 m, ridges about 2100-2300 m; snow, rock, dwarf pine, a tarn."""
    t, _, temp = dolina
    floor = t.wysokosc(t.baza[0], t.baza[1])
    assert 1500 < floor < 1700 and 2050 < np.percentile(t.z, 95) < 2500
    for kind in (T.SNIEG, T.SKALA, T.KOSO, T.WODA, T.SZLAK):
        assert (t.pokrycie == kind).any()
    assert temp[t.pokrycie == T.SNIEG].max() <= 0.5  # snow is not warmer than 0 C (+ texture)


def test_a_ray_from_the_air_meets_the_ground(dolina):
    t, _, _ = dolina
    o = np.array([30.0, -40.0, float(t.wysokosc(30.0, -40.0)) + 80.0])
    d = K.promienie(KAMERY["t256s"], 0.4, 90.0)
    hit = t.trafienie(o, d)
    assert np.allclose(hit[..., 2], t.wysokosc(hit[..., 0], hit[..., 1]), atol=0.5)


def test_a_person_fades_into_their_pixels_from_higher_up(dolina):
    """Mixed into a bigger pixel, a person is less warm above the ground round them."""
    t, pg, temp = dolina
    rng = np.random.default_rng(0)
    person = O.Obiekt("czlowiek_lezy", 40.0, 300.0, 0.0, 0.5, O.czlowiek("lezy", -2.0, rng))
    person.z = float(t.wysokosc(person.x, person.y))
    peaks = []
    for h in (30.0, 60.0, 120.0):
        v = K.zdjecie(
            t, temp, [person], KAMERY["t256s"], (40.0, 300.0, person.z + h), 0.0, rng=rng
        )
        spots = [s for s in W.plamy(v) if np.hypot(*(s.gdzie[:2] - (40.0, 300.0))) < 3]
        peaks.append(max(s.cechy[1] for s in spots) if spots else 0.0)
    assert peaks[0] > peaks[1] > peaks[2] and peaks[0] > 8.0
    mlx = K.zdjecie(
        t, temp, [person], KAMERY["mlx90640_55"], (40.0, 300.0, person.z + 60.0), 0.0
    )
    assert not [s for s in W.plamy(mlx) if np.hypot(*(s.gdzie[:2] - (40.0, 300.0))) < 3]


def test_both_platforms_fly_lanes_over_the_square_kilometre(dolina):
    """The plane flies every lane, skipping to turn; the quadcopter the trail's first."""
    plane = LOT.pasy(PLATFORMY["samolot"], KAMERY["t256s"], 60.0)
    quad = LOT.pasy(PLATFORMY["quad"], KAMERY["t256s"], 40.0)
    xs = sorted(lane[0][0] for lane in plane)
    assert xs[0] < -470 and xs[-1] > 470 and len(set(xs)) == len(xs)
    assert abs(plane[1][0][0] - plane[0][0][0]) > 60  # it skips lanes to turn
    assert abs(quad[0][0][0] - 60.0) < 20  # the lane nearest the trail first


def test_the_plane_holds_its_height_and_never_goes_below_its_floor(dolina):
    t, _, _ = dolina
    lot = LOT.Lot(PLATFORMY["samolot"], t, (0.0, 400.0), 60.0)
    heights = []
    for _ in range(600):
        lot.krok(0.2, (0.0, -480.0), 60.0)
        heights.append(lot.nad_ziemia)
    assert min(heights[100:]) >= LOT.NAJNIZEJ["samolot"] - 0.5 and np.median(heights[100:]) > 40
